"""Reproducible weak/synthetic benchmark runner."""
from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

import numpy as np
import yaml

from reportfinder.config import DEFAULT, from_mapping, resolve_config_path
from reportfinder.model import ReportFinder
from reportfinder.represent import load_or_build

from .datasets import load_weak_qrels
from .metrics import (
    binary_nll,
    brier,
    expected_calibration_error,
    ranking_metrics,
    risk_coverage,
)
from .qrels import load_qrels
from .regression import assert_no_regression
from .reports import write_summary
from .slices import group_by_slice


def run(cfg, qrels):
    rep = load_or_build(cfg, verbose=False); finder = ReportFinder(rep, cfg)
    # Exclude one-time model/index initialization from query latency. Index build
    # latency is an operational metric of its own, not steady-state p95.
    if qrels:
        finder.query(qrels[0].query, top_k=50)
    samples, times, answer_scores, answer_labels, details = [], [], [], [], []
    no_answer_predictions, top_correct, observed_statuses = [], [], []
    for qrel in qrels:
        started = time.perf_counter(); result = finder.query(qrel.query, top_k=50)
        times.append((time.perf_counter()-started)*1000)
        relevant = qrel.resolve_relevant(rep.frame)
        report_key_namespace = any(
            key.startswith("R") and key[1:].isdigit() for key in relevant
        )
        ranked = [
            str(c.row.get("report_key") if report_key_namespace
                else c.row.get("report_id", c.index))
            for c in result.candidates
        ]
        samples.append(ranking_metrics(ranked, relevant))
        answer_scores.append(result.answerability.score if result.answerability else result.p1)
        answer_labels.append(float(qrel.answerable))
        no_answer_predictions.append(result.status == "NO_SATISFACTORY_REPORT")
        observed_statuses.append(result.status)
        top_correct.append(float((bool(ranked) and ranked[0] in relevant) if qrel.answerable else no_answer_predictions[-1]))
        first_rank = next((rank for rank, report_id in enumerate(ranked, 1)
                           if report_id in relevant), None)
        details.append({
            "query_id": qrel.query_id, "query": qrel.query,
            "label_source": qrel.label_source, "expected_status": qrel.expected_status,
            "status": result.status, "answerability_score": (
                result.answerability.score if result.answerability else result.p1
            ),
            "returned": ranked, "rank_of_first_relevant": first_rank,
            "relevance_definition": {
                "explicit": qrel.relevant,
                "required_fields": list(qrel.required_fields),
                "title_terms_any": list(qrel.title_terms_any),
                "title_terms_all": list(qrel.title_terms_all),
                "categories": list(qrel.categories),
                "excluded_fields": list(qrel.excluded_fields),
            },
            "parsed_fields": [concept.value for concept in result.intent.fields] if result.intent else [],
            "expansion_concepts": [
                {"canonical": concept.canonical, "origin": concept.origin,
                 "confidence": concept.confidence}
                for concept in (result.intent.expansion.concepts if result.intent else ())
            ],
            "reasons": list(result.answerability.reasons) if result.answerability else [],
            "decision": result.decision,
            "active_fallbacks": sorted(result.fallbacks),
            "model_bundle_version": (
                result.outcome.model_bundle_version if result.outcome else ""
            ),
            "latency_ms": times[-1],
        })
    keys = samples[0] if samples else {}
    summary = {k: statistics.fmean(s[k] for s in samples) for k in keys}

    # The generator path ships no calibration artifact and its decision fallback is
    # a deterministic policy, so `answerability.score` is a constant placeholder
    # there. Brier, ECE, NLL and risk-coverage over a constant are arithmetic, not
    # measurement -- they are reported as null with the reason, rather than as
    # numbers someone could quote.
    scored = cfg.retrieval_mode != "generators"
    unscored_reason = (
        None if scored else
        "retrieval_mode=generators emits no answerability probability; "
        "no calibration artifact ships"
    )
    summary.update({"ece": expected_calibration_error(answer_scores, answer_labels) if scored else None,
                    "brier": brier(answer_scores, answer_labels) if scored else None,
                    "nll": binary_nll(answer_scores, answer_labels) if scored else None,
                    "calibration_metrics_unavailable_reason": unscored_reason,
                    "latency_p50_ms": float(statistics.median(times)) if times else 0,
                    "latency_p95_ms": float(np.percentile(times, 95, method="linear")) if times else 0,
                    "queries": len(qrels), "label_basis": "weak/synthetic; not production accuracy",
                    "answerability_calibrated": False,
                    "retrieval_mode": cfg.retrieval_mode,
                    "catalog_version": getattr(rep, "catalog_version", ""),
                    "per_query": details})
    truth = [
        q.expected_status == "NO_SATISFACTORY_REPORT"
        if q.expected_status else not q.answerable
        for q in qrels
    ]
    tp = sum(p and y for p, y in zip(no_answer_predictions, truth, strict=True))
    summary["no_answer_precision"] = tp / max(1, sum(no_answer_predictions))
    summary["no_answer_recall"] = tp / max(1, sum(truth))
    expected_observed = [
        (q.expected_status, observed)
        for q, observed in zip(qrels, observed_statuses, strict=True)
        if q.expected_status
    ]
    labels = (
        "ANSWERABLE", "NEEDS_CLARIFICATION", "NO_SATISFACTORY_REPORT",
    )
    summary["status_confusion_matrix"] = {
        expected: {
            observed: sum(e == expected and o == observed for e, o in expected_observed)
            for observed in labels
        }
        for expected in labels
    }
    summary["status_accuracy"] = (
        sum(expected == observed for expected, observed in expected_observed)
        / len(expected_observed) if expected_observed else None
    )
    ambiguity_rows = [
        observed for expected, observed in expected_observed
        if expected == "NEEDS_CLARIFICATION"
    ]
    summary["ambiguity_accuracy"] = (
        sum(observed == "NEEDS_CLARIFICATION" for observed in ambiguity_rows)
        / len(ambiguity_rows) if ambiguity_rows else None
    )
    by_query_id = {
        qrel.query_id: sample for qrel, sample in zip(qrels, samples, strict=True)
    }
    summary["slices"] = {}
    for slice_name, slice_qrels in group_by_slice(qrels).items():
        slice_samples = [by_query_id[qrel.query_id] for qrel in slice_qrels]
        summary["slices"][slice_name] = {
            key: statistics.fmean(sample[key] for sample in slice_samples)
            for key in keys
        }
        summary["slices"][slice_name]["queries"] = len(slice_qrels)
    summary["dataset_versions"] = {}
    for version in sorted({qrel.version for qrel in qrels}):
        version_samples = [
            sample for qrel, sample in zip(qrels, samples, strict=True)
            if qrel.version <= version
        ]
        summary["dataset_versions"][f"v{version}"] = {
            key: statistics.fmean(sample[key] for sample in version_samples)
            for key in keys
        }
        summary["dataset_versions"][f"v{version}"]["queries"] = len(version_samples)
    summary["risk_coverage"] = (
        risk_coverage(answer_scores, top_correct) if scored else None
    )
    return summary

def main():
    parser = argparse.ArgumentParser(); parser.add_argument("--config", required=True)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--qrels")
    source.add_argument("--weak-root")
    parser.add_argument("--weak-split", choices=["train", "validation", "test"], default="validation")
    parser.add_argument("--output", default="artifacts/benchmark_summary.json")
    parser.add_argument(
        "--set", action="append", default=[], metavar="KEY=VALUE",
        help="Override a Config value; repeat for ablation runs.",
    )
    parser.add_argument("--baseline", help="Optional prior artifact used as a regression gate.")
    args = parser.parse_args(); raw = yaml.safe_load(Path(args.config).read_text()) or {}
    overrides = {}
    for item in args.set:
        if "=" not in item:
            parser.error(f"--set expects KEY=VALUE, got {item!r}")
        key, value = item.split("=", 1)
        try:
            _, _, current = resolve_config_path(DEFAULT, key)
        except ValueError as error:
            parser.error(str(error))
        # Collections cannot be expressed unambiguously on a command line, and
        # silently replacing a quota map from a CLI string would quietly disable
        # source preservation. Those belong in the config file.
        if isinstance(current, (tuple, list, dict)):
            parser.error(
                f"--set cannot express the collection-valued key {key!r}; "
                "set it in the config file instead"
            )
        # YAML 1.1 parses the strings "on"/"off" as booleans. Preserve raw
        # values for string-valued Config fields such as dense_mode.
        overrides[key] = value if isinstance(current, str) else yaml.safe_load(value)
    qrels = load_qrels(args.qrels) if args.qrels else load_weak_qrels(args.weak_root, args.weak_split)
    cfg = from_mapping(raw).with_path_overrides(overrides); summary = run(cfg, qrels)
    summary["config_overrides"] = dict(sorted(overrides.items()))
    if args.baseline:
        baseline = json.loads(Path(args.baseline).read_text())
        assert_no_regression(baseline, summary)
        summary["regression_gate"] = {"baseline": args.baseline, "passed": True}
    write_summary(summary, args.output); print(json.dumps(summary, indent=2))
if __name__ == "__main__": main()
