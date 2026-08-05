"""Export the cross-encoder to ONNX, and verify it against torch before trusting it.

    uv run python scripts/export_cross_encoder_onnx.py --verify

Why: on this host torch returns NaN in float32 for realistic cross-encoder input,
so the torch path escalates to float64 -- exact, and ~2.3x slower, which is most
of per-query latency. The NaN is a kernel problem, not a weights problem, so a
different runtime may not have it. That is a hypothesis; `--verify` is what turns
it into a measurement.

The gate is deliberately strict on *ordering* rather than only on absolute error.
A backend whose logits differ slightly but rank identically is a drop-in; one
that reorders the shortlist is a different ranker wearing the same name.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from reportfinder.config import DEFAULT, from_mapping
from reportfinder.corpus import authoritative_text, build_corpus_model
from reportfinder.ingest import build_corpus

DEFAULT_OUT = Path("artifacts/onnx/cross_encoder.onnx")

# Real business questions, not lorem ipsum: the failure being investigated is
# length- and content-dependent.
PROBE_QUERIES = [
    "why are we losing people faster than we can backfill",
    "worker headcount by supervisory organization",
    "gross to net payroll results by pay group",
    "which training assignments are overdue",
    "compensation amounts and salary ranges by employee",
]


def export(out_path: Path, cfg) -> Path:
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    out_path.parent.mkdir(parents=True, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(
        cfg.rerank.checkpoint, revision=cfg.rerank.revision,
        trust_remote_code=cfg.security.trust_remote_code,
    )
    model = AutoModelForSequenceClassification.from_pretrained(
        cfg.rerank.checkpoint, revision=cfg.rerank.revision,
        trust_remote_code=cfg.security.trust_remote_code,
    ).eval()

    sample = tokenizer(
        [("query text", "document text")], return_tensors="pt",
        padding=True, truncation=True, max_length=cfg.rerank.max_length,
    )
    names = [n for n in ("input_ids", "attention_mask", "token_type_ids") if n in sample]
    # `dynamo=False` selects the legacy TorchScript exporter deliberately. The
    # dynamo exporter ignores `dynamic_axes`, and the graph it produced here baked
    # in batch=1 -- ONNX Runtime then failed at the first batch of 32 with a
    # LayerNormalization buffer-shape mismatch. It also wrote a 0.1 MB file, i.e.
    # a graph without its weights.
    with torch.inference_mode():
        torch.onnx.export(
            model,
            tuple(sample[n] for n in names),
            str(out_path),
            input_names=names,
            output_names=["logits"],
            dynamic_axes={
                **{n: {0: "batch", 1: "sequence"} for n in names},
                "logits": {0: "batch"},
            },
            opset_version=17,
            do_constant_folding=True,
            dynamo=False,
        )

    size_mb = out_path.stat().st_size / 1e6
    if size_mb < 10:
        raise RuntimeError(
            f"exported graph is only {size_mb:.1f} MB; the checkpoint is ~90 MB, so "
            "the weights did not make it into the file"
        )
    return out_path


def verify(out_path: Path, cfg, *, tolerance: float) -> dict:
    """Compare ONNX against the torch backend on real report text."""
    from reportfinder.pipeline.onnx_scorer import OnnxPairScorer
    from reportfinder.pipeline.rerank import TorchCrossEncoder

    built, _ = build_corpus(cfg.with_overrides(corpus_granularity="report_row"),
                            verbose=False)
    corpus = build_corpus_model(
        built.frame, ingest_mode=cfg.ingest_mode, source_file=str(cfg.data_path),
    )
    texts = [
        authoritative_text(instance, max_chars=cfg.rerank.max_report_chars)
        for instance in corpus.instances[:40]
    ]

    torch_scorer = TorchCrossEncoder(
        cfg.rerank.checkpoint, cfg.rerank.revision,
        max_length=cfg.rerank.max_length, batch_size=cfg.rerank.batch_size,
        trust_remote_code=cfg.security.trust_remote_code, dtype=cfg.rerank.dtype,
    )
    onnx_scorer = OnnxPairScorer(
        out_path, cfg.rerank.checkpoint, cfg.rerank.revision,
        max_length=cfg.rerank.max_length, batch_size=cfg.rerank.batch_size,
        trust_remote_code=cfg.security.trust_remote_code,
    )

    max_delta, order_mismatches, pairs = 0.0, 0, 0
    torch_ms, onnx_ms = [], []
    onnx_non_finite = 0
    for query in PROBE_QUERIES:
        started = time.perf_counter()
        reference = torch_scorer.score_pairs(query, texts)
        torch_ms.append((time.perf_counter() - started) * 1000)

        started = time.perf_counter()
        candidate = onnx_scorer.score_pairs(query, texts)
        onnx_ms.append((time.perf_counter() - started) * 1000)

        onnx_non_finite += int((~np.isfinite(candidate)).sum())
        finite = np.isfinite(reference) & np.isfinite(candidate)
        pairs += int(finite.sum())
        if finite.any():
            max_delta = max(max_delta, float(np.abs(reference[finite] - candidate[finite]).max()))
        if list(np.argsort(-reference)) != list(np.argsort(-candidate)):
            order_mismatches += 1

    passed = (
        max_delta <= tolerance
        and order_mismatches == 0
        and onnx_non_finite == 0
        and pairs > 0
    )
    return {
        "passed": passed,
        "pairs_compared": pairs,
        "max_abs_logit_delta": round(max_delta, 6),
        "tolerance": tolerance,
        "ordering_mismatches": order_mismatches,
        "queries": len(PROBE_QUERIES),
        "onnx_non_finite_scores": onnx_non_finite,
        "torch_dtype": torch_scorer.dtype,
        "torch_dtype_escalated": torch_scorer.dtype_escalated,
        "latency_ms_per_40_pairs": {
            "torch_p50": round(float(np.median(torch_ms)), 1),
            "onnx_p50": round(float(np.median(onnx_ms)), 1),
        },
        "measured_on": "single CPU host; not a production latency claim",
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="export_cross_encoder_onnx")
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--verify", action="store_true")
    parser.add_argument("--tolerance", type=float, default=1e-3)
    parser.add_argument("--report", type=Path,
                        default=Path("artifacts/onnx_parity.json"))
    args = parser.parse_args(argv)

    cfg = DEFAULT
    if args.config:
        import yaml
        cfg = from_mapping(yaml.safe_load(args.config.read_text()) or {})

    export(args.out, cfg)
    print(f"exported -> {args.out} ({args.out.stat().st_size / 1e6:.1f} MB)")

    if not args.verify:
        return 0

    report = verify(args.out, cfg, tolerance=args.tolerance)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))
    if not report["passed"]:
        print(
            "\nPARITY FAILED -- keep rerank.backend=torch. A backend that ranks "
            "differently is a different system, not a faster one."
        )
        return 1
    print("\nPARITY PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
