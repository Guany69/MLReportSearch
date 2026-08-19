"""Compile, verify, and benchmark the exported cross-encoder with TensorRT.

This is the GPU continuation of ``export_cross_encoder_onnx.py``: latency is
measured only after TensorRT passes an ordering-strict comparison with torch.
See ``docs/tensorrt_benchmark.md`` for the Colab T4 runbook.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import platform
import statistics
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np

try:
    from scripts.export_cross_encoder_onnx import DEFAULT_OUT, PROBE_QUERIES
except ModuleNotFoundError:  # Direct execution puts scripts/, not its parent, on sys.path.
    from export_cross_encoder_onnx import DEFAULT_OUT, PROBE_QUERIES
from reportfinder.config import DEFAULT, from_mapping
from reportfinder.corpus import authoritative_text, build_corpus_model
from reportfinder.evaluation.metrics import discordant_pairs
from reportfinder.ingest import build_corpus

DEFAULT_OUTPUT = Path("artifacts/evals/tensorrt_eval.json")
RUNBOOK = "docs/tensorrt_benchmark.md"
PROFILE_MIN = (1, 8)
PROFILE_MAX = (32, 256)
WARMUP_QUERIES = 5


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="benchmark_tensorrt",
        description="Build, verify, and benchmark a TensorRT cross-encoder engine.",
    )
    parser.add_argument("--onnx", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--queries", type=int, default=40)
    parser.add_argument("--shortlist", type=int, default=12)
    parser.add_argument(
        "--fp16", action=argparse.BooleanOptionalAction, default=True,
        help="build with TensorRT FP16 enabled (default: enabled)",
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--max-discordant", type=float, default=0.02)
    args = parser.parse_args(argv)
    if args.queries < 1:
        parser.error("--queries must be at least 1")
    if not 1 <= args.shortlist <= PROFILE_MAX[0]:
        parser.error(f"--shortlist must be between 1 and {PROFILE_MAX[0]}")
    if not 0.0 <= args.max_discordant <= 1.0:
        parser.error("--max-discordant must be between 0 and 1")
    return args


def require_gpu_environment() -> None:
    """Fail before artifact or model work when the GPU stack is unavailable."""
    problem = None
    if importlib.util.find_spec("tensorrt") is None:
        problem = "TensorRT is not importable"
    else:
        import torch

        if not torch.cuda.is_available():
            problem = "torch.cuda.is_available() is false"
    if problem:
        raise RuntimeError(f"{problem}. Run the Colab T4 runbook in {RUNBOOK}.")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def engine_path_for(onnx_path: Path, *, fp16: bool) -> Path:
    suffix = ".fp16.engine" if fp16 else ".fp32.engine"
    return onnx_path.with_suffix(suffix)


def build_engine(onnx_path: Path, engine_path: Path, *, shortlist: int, fp16: bool) -> None:
    import tensorrt as trt

    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = builder.create_network(
        1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    )
    parser = trt.OnnxParser(network, logger)
    if not parser.parse(onnx_path.read_bytes()):
        errors = "\n".join(str(parser.get_error(i)) for i in range(parser.num_errors))
        raise RuntimeError(f"TensorRT failed to parse {onnx_path}:\n{errors}")

    config = builder.create_builder_config()
    if fp16:
        config.set_flag(trt.BuilderFlag.FP16)
    profile = builder.create_optimization_profile()
    optimum = (shortlist, 128)
    for index in range(network.num_inputs):
        name = network.get_input(index).name
        if not profile.set_shape(name, PROFILE_MIN, optimum, PROFILE_MAX):
            raise RuntimeError(f"TensorRT rejected optimization profile for {name}")
    config.add_optimization_profile(profile)
    serialized = builder.build_serialized_network(network, config)
    if serialized is None:
        raise RuntimeError("TensorRT failed to build a serialized engine")
    engine_path.parent.mkdir(parents=True, exist_ok=True)
    engine_path.write_bytes(bytes(serialized))


def _torch_dtype(trt_dtype: Any):
    import tensorrt as trt
    import torch

    return torch.from_numpy(np.empty((), dtype=trt.nptype(trt_dtype))).dtype


class TensorRTRunner:
    def __init__(self, engine_path: Path) -> None:
        import tensorrt as trt
        import torch

        self._trt = trt
        self._torch = torch
        self._runtime = trt.Runtime(trt.Logger(trt.Logger.WARNING))
        self.engine = self._runtime.deserialize_cuda_engine(engine_path.read_bytes())
        if self.engine is None:
            raise RuntimeError(f"TensorRT could not deserialize {engine_path}")
        self.context = self.engine.create_execution_context()
        self.stream = torch.cuda.Stream()
        self.inputs = [
            self.engine.get_tensor_name(i)
            for i in range(self.engine.num_io_tensors)
            if self.engine.get_tensor_mode(self.engine.get_tensor_name(i))
            == trt.TensorIOMode.INPUT
        ]
        self.outputs = [
            self.engine.get_tensor_name(i)
            for i in range(self.engine.num_io_tensors)
            if self.engine.get_tensor_mode(self.engine.get_tensor_name(i))
            == trt.TensorIOMode.OUTPUT
        ]

    def prepare(self, encoded: dict[str, Any]) -> Callable[[], np.ndarray]:
        buffers: dict[str, Any] = {}
        for name in self.inputs:
            dtype = _torch_dtype(self.engine.get_tensor_dtype(name))
            value = encoded[name].to(device="cuda", dtype=dtype).contiguous()
            if not self.context.set_input_shape(name, tuple(value.shape)):
                raise RuntimeError(f"TensorRT rejected input shape {name}={tuple(value.shape)}")
            buffers[name] = value
        for name in self.outputs:
            shape = tuple(self.context.get_tensor_shape(name))
            if any(size < 0 for size in shape):
                raise RuntimeError(f"TensorRT left output shape unresolved: {name}={shape}")
            buffers[name] = self._torch.empty(
                shape, device="cuda", dtype=_torch_dtype(self.engine.get_tensor_dtype(name))
            )
        for name, value in buffers.items():
            self.context.set_tensor_address(name, value.data_ptr())

        def run() -> np.ndarray:
            for input_name in self.inputs:
                value = buffers[input_name]
                if not self.context.set_input_shape(input_name, tuple(value.shape)):
                    raise RuntimeError(
                        f"TensorRT rejected input shape {input_name}={tuple(value.shape)}"
                    )
            for tensor_name, value in buffers.items():
                self.context.set_tensor_address(tensor_name, value.data_ptr())
            if not self.context.execute_async_v3(self.stream.cuda_stream):
                raise RuntimeError("TensorRT execute_async_v3 failed")
            self.stream.synchronize()
            return buffers[self.outputs[0]].reshape(-1).float().cpu().numpy()

        return run


def timed(run: Callable[[], np.ndarray]) -> tuple[np.ndarray, float]:
    started = time.perf_counter()
    scores = run()
    return scores, (time.perf_counter() - started) * 1000.0


def percentile(values: Sequence[float], q: float) -> float:
    return float(np.percentile(np.asarray(values, dtype=float), q))


def make_result(
    *, args: argparse.Namespace, config: dict[str, Any], onnx_path: Path,
    engine_path: Path, versions: dict[str, str], gpu_name: str,
    latencies: dict[str, Sequence[float]], discordant_rates: Sequence[float],
    top1_flips: int,
) -> dict[str, Any]:
    return {
        "schema_version": "1",
        "kind": "tensorrt_cross_encoder_benchmark",
        "artifact": {
            "onnx_path": str(onnx_path), "onnx_sha256": sha256(onnx_path),
            "engine_path": str(engine_path), "engine_sha256": sha256(engine_path),
        },
        "config": {**config, "fp16": args.fp16},
        "dataset": {
            "label_source": "torch_fp32_cross_encoder_scores",
            "human_validated": False,
        },
        "label_basis": "identical real-report shortlists scored by torch FP32",
        "generated_at": datetime.now(UTC).isoformat(),
        "environment": {**versions, "gpu_name": gpu_name, "platform": platform.platform()},
        "workload": {
            "queries": args.queries, "shortlist": args.shortlist,
            "profile_min": list(PROFILE_MIN), "profile_opt": [args.shortlist, 128],
            "profile_max": list(PROFILE_MAX), "warmup_queries": WARMUP_QUERIES,
        },
        "metric_directions": {
            "latency_ms_per_query_p50": "lower",
            "latency_ms_per_query_p95": "lower",
            "mean_discordant_rate": "lower", "max_discordant_rate": "lower",
            "top1_flips": "lower",
        },
        "metrics": {
            "latency_ms_per_query": {
                name: {"p50": percentile(samples, 50), "p95": percentile(samples, 95)}
                for name, samples in latencies.items()
            },
            "agreement": {
                "mean_discordant_rate": statistics.fmean(discordant_rates),
                "max_discordant_rate": max(discordant_rates),
                "top1_flips": top1_flips,
                "max_discordant_threshold": args.max_discordant,
            },
        },
    }


def _load_config(path: Path | None):
    if path is None:
        return DEFAULT
    import yaml

    return from_mapping(yaml.safe_load(path.read_text()) or {})


def run_benchmark(args: argparse.Namespace) -> tuple[dict[str, Any], bool]:
    import onnxruntime as ort
    import tensorrt as trt
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    if not args.onnx.is_file():
        raise FileNotFoundError(
            f"ONNX artifact not found at {args.onnx}; run export_cross_encoder_onnx.py first"
        )
    cfg = _load_config(args.config)
    engine_path = engine_path_for(args.onnx, fp16=args.fp16)
    build_engine(args.onnx, engine_path, shortlist=args.shortlist, fp16=args.fp16)

    built, _ = build_corpus(
        cfg.with_overrides(corpus_granularity="report_row"), verbose=False
    )
    corpus = build_corpus_model(
        built.frame, ingest_mode=cfg.ingest_mode, source_file=str(cfg.data_path)
    )
    texts = [
        authoritative_text(item, max_chars=cfg.rerank.max_report_chars)
        for item in corpus.instances[: args.shortlist]
    ]
    if len(texts) != args.shortlist:
        raise RuntimeError(f"corpus has only {len(texts)} reports; need {args.shortlist}")
    queries = [PROBE_QUERIES[i % len(PROBE_QUERIES)] for i in range(args.queries)]

    tokenizer = AutoTokenizer.from_pretrained(
        cfg.rerank.checkpoint, revision=cfg.rerank.revision,
        trust_remote_code=cfg.security.trust_remote_code,
    )
    model = AutoModelForSequenceClassification.from_pretrained(
        cfg.rerank.checkpoint, revision=cfg.rerank.revision,
        trust_remote_code=cfg.security.trust_remote_code,
    ).eval().float().cuda()
    encoded_batches = [
        tokenizer(
            [(query, text) for text in texts], return_tensors="pt", padding=True,
            truncation=True, max_length=cfg.rerank.max_length,
        )
        for query in queries
    ]

    ort_session = ort.InferenceSession(
        str(args.onnx), providers=["CUDAExecutionProvider"]
    )
    if "CUDAExecutionProvider" not in ort_session.get_providers():
        raise RuntimeError("ONNX Runtime did not activate CUDAExecutionProvider")
    ort_names = [item.name for item in ort_session.get_inputs()]
    trt_runner = TensorRTRunner(engine_path)

    prepared: list[dict[str, Callable[[], np.ndarray]]] = []
    for encoded in encoded_batches:
        cuda_batch = {name: value.cuda() for name, value in encoded.items()}
        ort_feed = {
            name: encoded[name].numpy().astype(np.int64, copy=False) for name in ort_names
        }

        def torch_run(batch=cuda_batch) -> np.ndarray:
            with torch.inference_mode():
                values = model(**batch).logits.reshape(-1)
            torch.cuda.synchronize()
            return values.float().cpu().numpy()

        def ort_run(feed=ort_feed) -> np.ndarray:
            return ort_session.run(None, feed)[0].reshape(-1).astype(np.float32)

        prepared.append({
            "torch": torch_run, "onnxruntime": ort_run,
            "tensorrt": trt_runner.prepare(encoded),
        })

    for item in prepared[: min(WARMUP_QUERIES, len(prepared))]:
        for backend in ("torch", "onnxruntime", "tensorrt"):
            item[backend]()

    latencies: dict[str, list[float]] = {
        "torch": [], "onnxruntime": [], "tensorrt": []
    }
    discordant_rates: list[float] = []
    top1_flips = 0
    pair_count = args.shortlist * (args.shortlist - 1) // 2
    for item in prepared:
        scores: dict[str, np.ndarray] = {}
        for backend in ("torch", "onnxruntime", "tensorrt"):
            scores[backend], elapsed = timed(item[backend])
            latencies[backend].append(elapsed)
        reference, candidate = scores["torch"], scores["tensorrt"]
        if not np.isfinite(reference).all() or not np.isfinite(candidate).all():
            raise RuntimeError("non-finite torch or TensorRT verification scores")
        discordant_rates.append(discordant_pairs(reference, candidate) / max(1, pair_count))
        top1_flips += int(int(np.argmax(reference)) != int(np.argmax(candidate)))

    versions = {
        "python": platform.python_version(), "torch": torch.__version__,
        "onnxruntime": ort.__version__, "tensorrt": trt.__version__,
        "cuda": torch.version.cuda or "unknown",
    }
    result = make_result(
        args=args,
        config={
            "path": str(args.config) if args.config else None,
            "checkpoint": cfg.rerank.checkpoint,
            "revision": cfg.rerank.revision,
            "max_length": cfg.rerank.max_length,
            "max_report_chars": cfg.rerank.max_report_chars,
        },
        onnx_path=args.onnx, engine_path=engine_path, versions=versions,
        gpu_name=torch.cuda.get_device_name(0), latencies=latencies,
        discordant_rates=discordant_rates, top1_flips=top1_flips,
    )
    passed = max(discordant_rates) <= args.max_discordant and top1_flips == 0
    return result, passed


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        require_gpu_environment()
        result, passed = run_benchmark(args)
    except (RuntimeError, FileNotFoundError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))
    if not passed:
        print("ORDERING VERIFICATION FAILED", file=sys.stderr)
        return 1
    print("ORDERING VERIFICATION PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
