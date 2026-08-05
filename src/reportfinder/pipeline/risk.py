"""Recall risk: how likely it is the right report was missed.

This is deliberately *not* answer confidence, and conflating the two is the mistake
it exists to prevent. A vague query is uncertain, but the correct response to
uncertainty at retrieval time is to look **wider**, not to abstain. Abstention is
decided later, on the evidence that comes back.

So a HIGH risk band produces a larger shortlist and, when available, an extra
generator. It never narrows anything.

The signals are cheap and computed from what already exists: query shape, how much
the generators agreed, how many candidates only one source found, and how flat the
fused score distribution is.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Literal

from .prepare import FacetState

RiskBand = Literal["LOW", "MEDIUM", "HIGH"]


@dataclass(frozen=True)
class RecallRiskResult:
    """The retrieval-time widening decision."""

    risk_band: RiskBand
    reasons: list[str] = field(default_factory=list)
    rerank_depth: int = 120
    run_late_interaction: bool = False
    signals: dict[str, object] = field(default_factory=dict)

    def telemetry(self) -> dict[str, object]:
        return {
            "risk_band": self.risk_band,
            "reasons": list(self.reasons),
            "rerank_depth": self.rerank_depth,
            "run_late_interaction": self.run_late_interaction,
            "signals": dict(self.signals),
        }


def _score_entropy(scores: list[float]) -> float:
    """Normalized entropy of the fused scores at the head of the ranking.

    Computed over the head rather than the whole union on purpose: RRF gives a
    large union a long tail of near-equal small scores, so whole-union entropy
    approaches 1 for *every* query and stops discriminating. What matters is
    whether retrieval separated the candidates that are actually in contention.
    """
    positive = [s for s in scores if s > 0][:_ENTROPY_HEAD]
    if len(positive) < 2:
        return 0.0
    total = sum(positive)
    probabilities = [s / total for s in positive]
    entropy = -sum(p * math.log(p) for p in probabilities if p > 0)
    return entropy / math.log(len(probabilities))


# How many candidates count as "in contention" for the entropy and margin signals.
_ENTROPY_HEAD = 10


def assess_recall_risk(
    plan,
    union,
    cfg,
    *,
    authorized_count: int,
    late_interaction_available: bool = False,
) -> RecallRiskResult:
    """Decide how wide to search."""
    risk = cfg.risk
    reasons: list[str] = []
    signals: dict[str, object] = {}

    tokens = plan.raw_query.split()
    signals["query_tokens"] = len(tokens)
    if len(tokens) <= risk.short_query_tokens:
        reasons.append("very_short_query")

    stated = plan.stated_facets()
    signals["stated_facet_count"] = len(stated)
    schema_facets = [
        name for name in ("measure", "dimension", "data_source", "population")
        if not isinstance(plan.facets.get(name), FacetState)
    ]
    coverage = len(schema_facets) / 4
    signals["schema_facet_coverage"] = round(coverage, 3)
    if coverage < risk.min_schema_facet_coverage:
        reasons.append("weak_schema_facet_coverage")

    if any(
        plan.facets.get(name) is FacetState.MULTIPLE
        for name in ("time_orientation", "granularity")
    ):
        reasons.append("conflicting_facets")

    records = union.ordered()
    signals["union_size"] = len(records)
    signals["authorized_instance_count"] = authorized_count

    if not records:
        # Nothing was retrieved at all: the widest possible search is the only
        # response that can still recover.
        reasons.append("empty_union")
        return RecallRiskResult(
            risk_band="HIGH",
            reasons=reasons,
            rerank_depth=cfg.shortlist.depth_for("HIGH"),
            run_late_interaction=(
                late_interaction_available
                and cfg.retrieval.late_interaction.trigger_on_high_risk
            ),
            signals=signals,
        )

    generators_run = max(1, len(union.generators_run))
    top = records[: min(20, len(records))]
    overlap = sum(r.generator_count for r in top) / (len(top) * generators_run)
    signals["generator_overlap"] = round(overlap, 3)
    if overlap < risk.min_generator_overlap:
        reasons.append("low_generator_overlap")

    exclusive = sum(1 for r in top if r.source_exclusive) / len(top)
    signals["source_exclusive_ratio"] = round(exclusive, 3)
    if exclusive > risk.high_source_exclusive_ratio:
        # Most of the head was found by exactly one generator: the sources disagree
        # about what is relevant, so any single-source cut would lose a lot.
        reasons.append("high_source_exclusivity")

    families = {r.family_id for r in records}
    signals["candidate_family_count"] = len(families)
    if len(families) >= risk.many_families:
        reasons.append("many_candidate_families")

    entropy = _score_entropy([r.fused_score for r in records])
    signals["score_entropy"] = round(entropy, 3)
    if entropy > risk.high_entropy:
        reasons.append("flat_score_distribution")

    if len(records) >= 2 and records[0].fused_score > 0:
        # Relative, not absolute. RRF scores live on a scale set by the number of
        # generators and the constant -- a rank-1 hit contributes only 1/(k+1) --
        # so an absolute threshold would flag essentially every query.
        margin = (
            records[0].fused_score - records[1].fused_score
        ) / records[0].fused_score
        signals["top_margin_relative"] = round(margin, 4)
        if margin < risk.low_margin:
            reasons.append("low_top_margin")

    if len(union.generators_failed) > 0:
        # A generator that failed is a hole in coverage, so widen to compensate.
        reasons.append("generator_failure")
        signals["failed_generators"] = sorted(union.generators_failed)

    count = len(reasons)
    if count >= risk.high_signal_threshold:
        band: RiskBand = "HIGH"
    elif count >= risk.medium_signal_threshold:
        band = "MEDIUM"
    else:
        band = "LOW"

    run_late = (
        band == "HIGH"
        and late_interaction_available
        and cfg.retrieval.late_interaction.trigger_on_high_risk
    )
    return RecallRiskResult(
        risk_band=band,
        reasons=reasons,
        rerank_depth=cfg.shortlist.depth_for(band),
        run_late_interaction=run_late,
        signals=signals,
    )
