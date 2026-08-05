"""Decision-head features, computed identically at training and serving time.

The whole point of putting this in one function is that it is impossible to train
on something serving cannot produce. Every value below comes from objects the
pipeline already has in hand mid-request: the ranked families, the recall-risk
result, the union and the reranker output.

Explicitly excluded, and the reason they are dangerous: gold report ids, relevance
grades, gold facets, anything about what the user did next, and any annotation that
exists only in the training set. Each would make the head look excellent offline
and behave randomly in production.
"""

from __future__ import annotations

import hashlib
import math

import numpy as np

DECISION_FEATURE_SCHEMA_VERSION = "1"

DECISION_FEATURE_NAMES = (
    "top_family_score",
    "score_margin_top_two",
    "cross_encoder_top",
    "cross_encoder_margin",
    "cross_encoder_mean_top5",
    "score_entropy_top10",
    "family_count",
    "candidate_count",
    "authorized_candidate_count_log",
    "generator_agreement_top",
    "source_exclusive_ratio_top",
    "risk_low",
    "risk_medium",
    "risk_high",
    "risk_reason_count",
    "query_token_count",
    "stated_facet_count",
    "query_variant_count",
    "distinguishing_facet_present",
    "top_family_instance_count",
)


def decision_feature_hash() -> str:
    payload = "|".join([DECISION_FEATURE_SCHEMA_VERSION, *DECISION_FEATURE_NAMES])
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


DECISION_FEATURE_HASH = decision_feature_hash()


def _entropy(values: list[float]) -> float:
    positive = [v for v in values if v > 0][:10]
    if len(positive) < 2:
        return 0.0
    total = sum(positive)
    probabilities = [v / total for v in positive]
    return -sum(p * math.log(p) for p in probabilities) / math.log(len(probabilities))


def _distinguishing_facet_present(families) -> float:
    """Whether the leading families actually differ on something askable.

    A clarification is only useful if the candidates disagree; without this the
    head could learn to ask questions the candidate set cannot answer.
    """
    leading = families[:5]
    if len(leading) < 2:
        return 0.0
    for attribute in ("data_source", "category", "report_type"):
        values = {
            getattr(f.selected.record, attribute, None) or
            getattr(f, attribute, "") for f in leading
        }
        if len(values - {None, ""}) >= 2:
            return 1.0
    return 0.0


def decision_features(*, families, plan, risk, union, rerank, fusion=None) -> np.ndarray:
    """Serving-time features for the three-way decision."""
    if not families:
        return np.zeros(len(DECISION_FEATURE_NAMES), dtype=np.float32)

    scores = [f.score for f in families]
    top = families[0]
    second = families[1] if len(families) > 1 else None

    ce_values = [
        f.selected.cross_encoder_score for f in families
        if f.selected.cross_encoder_score is not None
    ]
    ce_top = ce_values[0] if ce_values else 0.0
    ce_margin = (ce_values[0] - ce_values[1]) if len(ce_values) > 1 else 0.0
    ce_mean5 = float(np.mean(ce_values[:5])) if ce_values else 0.0

    head = union.ordered()[:20] if union is not None else []
    generators_run = max(1, len(getattr(union, "generators_run", ()) or ()))
    agreement = (
        sum(r.generator_count for r in head) / (len(head) * generators_run)
        if head else 0.0
    )
    exclusive = (
        sum(1 for r in head if r.source_exclusive) / len(head) if head else 0.0
    )

    band = risk.risk_band if risk is not None else "LOW"
    values = [
        float(top.score),
        float(top.score - second.score) if second else 1.0,
        float(ce_top),
        float(ce_margin),
        float(ce_mean5),
        _entropy(scores),
        float(len(families)),
        float(len(union) if union is not None else 0),
        math.log1p(float(risk.signals.get("authorized_instance_count", 0)))
        if risk is not None else 0.0,
        float(agreement),
        float(exclusive),
        1.0 if band == "LOW" else 0.0,
        1.0 if band == "MEDIUM" else 0.0,
        1.0 if band == "HIGH" else 0.0,
        float(len(risk.reasons)) if risk is not None else 0.0,
        float(len(plan.raw_query.split())) if plan is not None else 0.0,
        float(len(plan.stated_facets())) if plan is not None else 0.0,
        float(len(plan.variants)) if plan is not None else 0.0,
        _distinguishing_facet_present(families),
        float(len(top.instances)),
    ]
    return np.asarray(values, dtype=np.float32)
