"""Evaluating a candidate model against the fallback it would replace.

The comparison is the whole point. "The fusion model reached nDCG@5 0.71" is not
a reason to serve it; "it reached 0.71 where RRF reaches 0.66 on a split neither
was fitted on" is. So every metric is computed twice, on the same cached passes,
and the approval gate reads the delta rather than the absolute.

Both arms come from one pipeline run per query: the fallback's real output was
recorded during the feature pass, so the baseline cannot drift from what the
fallback would actually have done, and no second pass is needed.
"""

from __future__ import annotations

import json
import math
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

EVAL_SCHEMA_VERSION = "1"

# The metric approval is decided on. Family nDCG for fusion: a ranking model's
# job is to order families, and nDCG uses the graded relevance v2 provides rather
# than collapsing it to binary.
FUSION_PRIMARY_METRIC = "family_ndcg@5"
# Macro per-class recall for the decision head, not accuracy: the classes are
# ~75/11/14, so a model that always says RETURN_RESULTS scores 0.75 accuracy
# while never abstaining, which is the one behaviour the head exists to provide.
DECISION_PRIMARY_METRIC = "macro_per_class_recall"


def _ndcg(ranked: list[str], grades: dict[str, int], k: int) -> float:
    gains = [grades.get(item, 0) for item in ranked[:k]]
    dcg = sum(g / math.log2(i + 2) for i, g in enumerate(gains))
    ideal = sorted(grades.values(), reverse=True)[:k]
    best = sum(g / math.log2(i + 2) for i, g in enumerate(ideal))
    return dcg / best if best else 0.0


def _mrr(ranked: list[str], positives: set[str]) -> float:
    for position, item in enumerate(ranked, start=1):
        if item in positives:
            return 1.0 / position
    return 0.0


def _recall_at(ranked: list[str], positives: set[str], k: int) -> float:
    if not positives:
        return 0.0
    return len(set(ranked[:k]) & positives) / len(positives)


def _family_order(instance_ids: list[str], family_of) -> list[str]:
    """Families in first-appearance order of their instances."""
    seen, out = set(), []
    for instance_id in instance_ids:
        family = family_of(instance_id)
        if family not in seen:
            seen.add(family)
            out.append(family)
    return out


def _family_grades(grades: dict[str, int], family_of) -> dict[str, int]:
    """A family is as relevant as its best instance."""
    out: dict[str, int] = {}
    for instance_id, grade in grades.items():
        family = family_of(instance_id)
        out[family] = max(out.get(family, 0), grade)
    return out


def _ranking_metrics(orderings, passes, family_of) -> dict[str, object]:
    ndcg, mrr, recall, instance_hits, instance_total = [], [], [], 0, 0

    for query_pass, instance_order in zip(passes, orderings, strict=True):
        if not query_pass.grades:
            continue
        family_grades = _family_grades(query_pass.grades, family_of)
        family_order = _family_order(instance_order, family_of)
        positives = {f for f, g in family_grades.items() if g > 0}

        ndcg.append(_ndcg(family_order, family_grades, 5))
        mrr.append(_mrr(family_order, positives))
        recall.append(_recall_at(family_order, positives, 10))

        # Instance selection is only measurable where the labels distinguish
        # siblings, which before v2 was nowhere.
        if query_pass.archetype == "sibling_discriminating":
            best = {r for r, g in query_pass.grades.items() if g == 2}
            chosen = next(
                (i for i in instance_order if family_of(i) in positives), None
            )
            if chosen is not None:
                instance_total += 1
                instance_hits += int(chosen in best)

    def mean(values):
        return round(float(np.mean(values)), 4) if values else 0.0

    return {
        "family_ndcg@5": mean(ndcg),
        "family_mrr": mean(mrr),
        "family_recall@10": mean(recall),
        "instance_recall@1_within_family": (
            round(instance_hits / instance_total, 4) if instance_total else None
        ),
        "sibling_queries_measured": instance_total,
        "queries": len(ndcg),
    }


def evaluate_fusion(model, passes, *, family_of) -> dict:
    """The learned fuser against the RRF ordering it would replace."""
    import torch

    neural_orders: list[list[str]] = []
    with torch.inference_mode():
        for query_pass in passes:
            if not len(query_pass.instance_ids):
                neural_orders.append([])
                continue
            scores = model(
                torch.from_numpy(np.asarray(query_pass.fusion_features))
            ).numpy()
            order = np.argsort(-scores, kind="stable")
            neural_orders.append([query_pass.instance_ids[i] for i in order])

    candidate = _ranking_metrics(neural_orders, passes, family_of)
    fallback = _ranking_metrics(
        [p.rrf_ranking for p in passes], passes, family_of
    )
    fallback["fallback_name"] = "rrf_with_cross_encoder_priority"
    return {"candidate": candidate, "fallback": fallback}


def _decision_metrics(predicted: list[str], actual: list[str]) -> dict:
    classes = ("RETURN_RESULTS", "ASK_CLARIFICATION", "NO_CONFIDENT_MATCH")
    per_class = {}
    for name in classes:
        support = sum(1 for a in actual if a == name)
        hits = sum(1 for p, a in zip(predicted, actual, strict=True)
                   if a == name and p == name)
        per_class[name] = round(hits / support, 4) if support else None

    measured = [v for v in per_class.values() if v is not None]
    return {
        "per_class_recall": per_class,
        # Macro, so the two minority classes count as much as the majority one.
        "macro_per_class_recall": round(float(np.mean(measured)), 4) if measured else 0.0,
        "accuracy": round(
            sum(1 for p, a in zip(predicted, actual, strict=True) if p == a)
            / len(actual), 4
        ) if actual else 0.0,
        "confusion_matrix": {
            expected: {
                got: sum(
                    1 for p, a in zip(predicted, actual, strict=True)
                    if a == expected and p == got
                )
                for got in classes
            }
            for expected in classes
        },
        "predicted_distribution": dict(sorted(Counter(predicted).items())),
        "abstain_rate": round(
            sum(1 for p in predicted if p == "NO_CONFIDENT_MATCH") / len(predicted), 4
        ) if predicted else 0.0,
    }


def evaluate_decision(model, temperature: float, passes, labels: dict[str, str]) -> dict:
    """The learned head against the deterministic policy it would replace."""
    import torch

    from .nets import DecisionHead as _Head

    actual = [labels[p.scenario_id] for p in passes]

    features = np.vstack([p.decision_features for p in passes]).astype(np.float32)
    with torch.inference_mode():
        logits = model(torch.from_numpy(features))
        probabilities = torch.softmax(logits / temperature, dim=-1).numpy()
    predicted = [_Head.CLASSES[int(i)] for i in probabilities.argmax(axis=1)]

    candidate = _decision_metrics(predicted, actual)
    candidate["ece"] = round(
        float(_ece(probabilities.max(axis=1), np.array(
            [p == a for p, a in zip(predicted, actual, strict=True)], dtype=float
        ))), 4
    )

    fallback = _decision_metrics([p.fallback_decision for p in passes], actual)
    fallback["fallback_name"] = "deterministic_three_way_policy"
    return {"candidate": candidate, "fallback": fallback}


def _ece(confidence: np.ndarray, correct: np.ndarray, bins: int = 10) -> float:
    edges = np.linspace(0.0, 1.0, bins + 1)
    total = 0.0
    for low, high in zip(edges[:-1], edges[1:], strict=True):
        mask = (confidence > low) & (confidence <= high)
        if not mask.any():
            continue
        total += mask.mean() * abs(correct[mask].mean() - confidence[mask].mean())
    return total


def write_eval_report(
    kind: str,
    artifact_path: Path,
    metrics: dict,
    *,
    weights_sha256: str,
    feature_hash: str,
    dataset_root: Path,
    dataset_manifest: dict,
    split: str,
    query_count: int,
    slices: dict | None = None,
    config_path: str = "",
    bundle_version: str = "",
    out: Path,
) -> Path:
    """The report `reportfinder-train approve` reads.

    Bound to the model by weights digest rather than by path, so a report cannot
    be pointed at a different artifact -- accidentally or otherwise.
    """
    primary = (
        FUSION_PRIMARY_METRIC if kind == "fusion" else DECISION_PRIMARY_METRIC
    )
    candidate_value = metrics["candidate"].get(primary) or 0.0
    fallback_value = metrics["fallback"].get(primary) or 0.0

    report = {
        "schema_version": EVAL_SCHEMA_VERSION,
        "kind": kind,
        "artifact": {
            "path": str(artifact_path),
            "weights_sha256": weights_sha256,
            "feature_hash": feature_hash,
        },
        "dataset": {
            "root": str(dataset_root),
            "dataset_version": dataset_manifest.get("version", "unknown"),
            "label_source": dataset_manifest.get("label_source", "unknown"),
            "human_validated": bool(dataset_manifest.get("human_validated", False)),
        },
        "split": split,
        "query_count": query_count,
        "primary_metric": {
            "name": primary,
            "candidate": round(float(candidate_value), 4),
            "fallback": round(float(fallback_value), 4),
            "delta": round(float(candidate_value) - float(fallback_value), 4),
        },
        "metrics": metrics,
        "slices": slices or {},
        "config": {"path": config_path, "bundle_version": bundle_version},
        "label_basis": (
            f"{dataset_manifest.get('label_source', 'unknown')}; measures agreement "
            "with the label generator, not production relevance"
        ),
        "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
    }
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return out


def slice_metrics(passes, evaluator) -> dict:
    """Per-archetype breakdown. An aggregate can hide a slice going backwards."""
    out = {}
    for archetype in sorted({p.archetype for p in passes}):
        subset = [p for p in passes if p.archetype == archetype]
        if len(subset) >= 10:
            out[archetype] = evaluator(subset)
    return out
