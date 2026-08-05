"""Capture a reproducible behavior/latency snapshot for diagnostic queries."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import platform
import statistics
import time
from pathlib import Path

from reportfinder.config import DEFAULT
from reportfinder.model import ReportFinder
from reportfinder.represent import load_or_build

QUERIES = [
    "terminated workers by supervisory organization",
    "payroll gross pay by pay group",
    "new hires with start dates",
    "employees who transferred between departments",
    "monthly turnover by organization",
    "active headcount by company",
    "Why did people leave?",
    "Show voluntary turnover by boss.",
    "How big is each team?",
    "Show take-home pay and deductions by paycheck period.",
    "Who could replace each leader?",
    "Where does each employee work?",
    "Show applicants by recruiting stage.",
    "Which training assignments are overdue?",
    "show me attriton by manger",
    "reports with net pay",
    "terminations",
    "quantum submarine telemetry",
]


def _version(package: str) -> str:
    try:
        return importlib.metadata.version(package)
    except importlib.metadata.PackageNotFoundError:
        return "missing"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="artifacts/baseline_results.json")
    parser.add_argument("--dense-mode", choices=["auto", "local", "off"], default="auto")
    parser.add_argument(
        "--mode", choices=["legacy_single_file", "phase2_dual_file"],
        default="legacy_single_file",
        help="Ingest mode. Was hardcoded to phase2_dual_file, whose workbooks are "
             "not in this tree, so the script could not run at all.",
    )
    args = parser.parse_args()
    cfg = DEFAULT.with_overrides(
        ingest_mode=args.mode, dense_mode=args.dense_mode,
    )
    load_started = time.perf_counter()
    rep = load_or_build(cfg, verbose=False)
    load_ms = (time.perf_counter() - load_started) * 1000
    finder = ReportFinder(rep, cfg)
    records = []
    latencies = []
    for query in QUERIES:
        started = time.perf_counter()
        result = finder.query(query, top_k=10)
        latency = (time.perf_counter() - started) * 1000
        latencies.append(latency)
        records.append({
            "query": query, "latency_ms": latency, "status": result.status,
            "p1": result.p1, "p2": result.p2, "margin": result.margin,
            "normalized_entropy": result.normalized_entropy,
            "answerability_score": result.answerability.score,
            "answerability_reasons": list(result.answerability.reasons),
            "top_candidates": [
                {
                    "index": candidate.index,
                    "report_id": str(candidate.row.get("report_id", candidate.index)),
                    "title": str(candidate.row["title"]),
                    "ranking_score": candidate.ranking_score,
                    "retriever_ranks": candidate.retriever_ranks,
                    "features": candidate.features.values if candidate.features else {},
                } for candidate in result.candidates[:10]
            ],
            "intent": {
                "fields": [
                    {
                        "value": field.value, "evidence": field.evidence,
                        "confidence": field.confidence, "mandatory": field.mandatory,
                        "requirement": field.requirement.value, "origin": field.origin,
                    } for field in result.intent.fields
                ],
                "subqueries": result.intent.subqueries,
                "parser_confidence": result.intent.parser_confidence,
                "expanded_query": result.intent.expanded_query,
            },
        })
    # Fingerprint whichever workbook this mode actually read.
    source = (
        cfg.catalog_path if cfg.ingest_mode == "phase2_dual_file" else cfg.data_path
    )
    catalog_bytes = Path(source).read_bytes()
    artifact = {
        "configuration": {
            "ingest_mode": cfg.ingest_mode, "retrieval_mode": cfg.retrieval_mode,
            "dense_mode": cfg.dense_mode, "use_query_expansion": cfg.use_query_expansion,
        },
        "dataset_fingerprint": hashlib.sha256(catalog_bytes).hexdigest(),
        "environment": {
            "python": platform.python_version(),
            **{name: _version(name) for name in (
                "numpy", "pandas", "scikit-learn", "sentence-transformers",
            )},
        },
        "cache_load_ms": load_ms,
        "latency_p50_ms": statistics.median(latencies),
        "latency_p95_ms": sorted(latencies)[min(len(latencies) - 1, int(.95 * len(latencies)))],
        "queries": records,
        "label_limitation": (
            "Diagnostic behavior snapshot only. No human relevance judgments exist; "
            "this artifact is not a production-accuracy measurement."
        ),
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
