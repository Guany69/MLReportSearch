"""Measure candidate survival at every pruning boundary.

    uv run python scripts/measure_survival.py --queries 60 \
        --output artifacts/candidate_survival_generators.json

Final ranking metrics cannot diagnose a recall failure: if the right report was cut
at the shortlist, nDCG@5 reads low, and it reads *exactly the same* as if the report
had been retrieved and ranked badly -- two problems with opposite fixes. This runs
the real serving pipeline over judged queries and attributes every lost relevant
instance to the boundary that dropped it.

The labels are synthetic weak supervision. Every summary says so, and no number
produced here is production accuracy.
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

from reportfinder.auth import DEVELOPMENT_PRINCIPAL, SearchRequest
from reportfinder.config import DEFAULT, from_mapping
from reportfinder.evaluation.survival import observe, summarize
from reportfinder.model import ReportFinder
from reportfinder.represent import load_or_build
from reportfinder.training.datasets import load_labelled_queries


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="measure_survival")
    parser.add_argument("--config", default=None)
    parser.add_argument("--relevance-root", type=Path, default=Path("data/relevance"))
    parser.add_argument("--split", default="validation")
    parser.add_argument("--queries", type=int, default=60)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)

    cfg = DEFAULT
    if args.config:
        import yaml

        cfg = from_mapping(yaml.safe_load(Path(args.config).read_text()) or {})
    if cfg.retrieval_mode != "generators":
        raise SystemExit(
            f"survival measurement needs retrieval_mode=generators, got "
            f"{cfg.retrieval_mode}: the older paths have no per-stage trace"
        )

    finder = ReportFinder(load_or_build(cfg, verbose=False), cfg)
    corpus = finder.pipeline.corpus
    known = set(corpus.instance_ids)

    labelled = load_labelled_queries(args.relevance_root, split=args.split)
    # Deterministic selection: the first N judged queries that have at least one
    # relevant instance actually present in this corpus. A query whose positives
    # are all absent would score 0 recall for reasons that are not the pipeline's.
    selected = []
    for item in labelled:
        relevant = {r for r, grade in item.grades.items() if grade > 0} & known
        if relevant:
            selected.append((item, relevant))
        if len(selected) >= args.queries:
            break

    # One warm-up query outside the timed loop. Without it the first measurement
    # includes checkpoint loading and index warm-up, which are one-time costs and
    # not what a per-query latency figure claims to describe.
    if selected:
        finder.search(
            SearchRequest(selected[0][0].query, DEVELOPMENT_PRINCIPAL, top_k=args.top_k)
        )

    records, latencies = [], []
    for item, relevant in selected:
        started = time.perf_counter()
        outcome = finder.search(
            SearchRequest(item.query, DEVELOPMENT_PRINCIPAL, top_k=args.top_k)
        )
        latencies.append((time.perf_counter() - started) * 1000)
        records.append(observe(
            outcome, query_id=item.scenario_id, relevant_instances=relevant,
            family_of=lambda i: corpus.instance(i).family_id,
        ))

    # `recall@final` counts only what a user is actually shown, and
    # NO_CONFIDENT_MATCH deliberately shows nothing. That is the honest headline,
    # but it conflates two questions -- "did the pipeline find it" and "did the
    # pipeline choose to answer" -- so the answering subset is reported separately.
    answered = {
        r.query_id for r in records if r.decision != "NO_CONFIDENT_MATCH"
    }
    summary = summarize(records, slices={"answered_queries": answered})
    summary["latency_ms"] = {
        "p50": round(statistics.median(latencies), 1) if latencies else 0.0,
        "p95": round(
            sorted(latencies)[max(0, int(len(latencies) * 0.95) - 1)], 1
        ) if latencies else 0.0,
        "measured_on": "single CPU host; not a production latency claim",
    }
    reranker = finder.pipeline.reranker
    summary["cross_encoder_dtype"] = getattr(reranker, "dtype", "unknown")
    summary["retrieval_mode"] = cfg.retrieval_mode
    summary["family_expansion_enabled"] = cfg.aggregation.enabled

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")

    print(f"queries={summary['query_count']} relevant={summary['relevant_instances_total']}")
    for stage, value in summary["recall_by_stage"].items():
        print(f"  recall@{stage:<10} {value:.3f}")
    print(f"  instance_recall@1_within_family {summary['instance_recall@1_within_correct_family']:.3f}")
    print(f"  loss_by_boundary {summary['loss_by_boundary']}")
    print(f"-> {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
