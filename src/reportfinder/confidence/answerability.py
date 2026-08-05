"""Three-way, explicitly uncalibrated answerability decision."""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum

from .support import SupportAccounting


class AnswerabilityOutcome(str, Enum):
    ANSWERABLE = "ANSWERABLE"
    NEEDS_CLARIFICATION = "NEEDS_CLARIFICATION"
    NO_SATISFACTORY_REPORT = "NO_SATISFACTORY_REPORT"
    HIGH_CONFIDENCE = "ANSWERABLE"
    MULTIPLE_PLAUSIBLE_RESULTS = "NEEDS_CLARIFICATION"
    INSUFFICIENT_MATCH = "NO_SATISFACTORY_REPORT"


@dataclass(frozen=True)
class AnswerabilityDecision:
    outcome: AnswerabilityOutcome
    score: float
    reasons: tuple[str, ...] = ()
    calibrated: bool = False


def _clamp(value: float) -> float:
    return min(1.0, max(0.0, value))


def intent_ambiguity(intent, categories, raw_scores) -> tuple[float, tuple[str, ...]]:
    """Return an inspectable intent-specificity ambiguity score.

    This deliberately avoids entropy over a bounded candidate set. It combines
    query specificity, category dispersion among the five best candidates, and
    separation on the ranker's raw scale.
    """
    concepts = [
        concept for concept in getattr(getattr(intent, "expansion", None), "concepts", ())
        if concept.confidence >= .80
    ]
    generic_fields = {"worker", "manager", "organization", "company", "pay"}
    specific = any(
        concept.kind == "field" and concept.canonical.casefold() not in generic_fields
        for concept in concepts
    ) or bool(getattr(intent, "measures", ()))
    concept_sparsity = 1.0 - min(1.0, len(concepts) / 3.0)
    distinct_categories = len({
        str(category).strip().casefold() for category in categories[:5]
        if str(category).strip()
    })
    category_dispersion = _clamp((distinct_categories - 1) / 4.0)
    values = list(raw_scores)
    if len(values) >= 2:
        raw_gap = max(0.0, float(values[0]) - float(values[1]))
        gap_ambiguity = 1.0 - _clamp(raw_gap / .25)
    else:
        gap_ambiguity = 0.0
    score = _clamp(
        .45 * float(not specific)
        + .20 * concept_sparsity
        + .20 * category_dispersion
        + .15 * gap_ambiguity
    )
    reasons = list(getattr(intent, "ambiguity_reasons", ()))
    if not specific:
        reasons.append("The request names only a broad topic, not a specific report field or measure.")
    if distinct_categories >= 3:
        reasons.append(f"The leading matches span {distinct_categories} report categories.")
    if gap_ambiguity >= .75 and len(values) >= 2:
        reasons.append("The two strongest candidates are close on the raw ranking scale.")
    return score, tuple(dict.fromkeys(reasons))


def decide_answerability(
    *, top_score: float, margin: float, field_coverage: float,
    retriever_agreement: float, parser_confidence: float,
    support: SupportAccounting | None = None, breadth: float = 0.0,
    high_threshold: float = .72, insufficient_threshold: float = .35,
    min_support_ratio: float = .50, hard_support_floor: float = .20,
    unsupported: list[str] | None = None,
    breadth_threshold: float = .75, min_margin: float = .06,
    min_high_support: float = .70, min_field_coverage: float = .80,
    intercept: float = -1.60, top_score_weight: float = 1.70,
    margin_weight: float = 1.60, margin_scale: float = .15,
    field_coverage_weight: float = 1.30,
    retriever_agreement_weight: float = .80,
    parser_confidence_weight: float = .50,
    support_weight: float = 1.10, ambiguity_weight: float = 1.20,
    ambiguity_reasons: tuple[str, ...] = (),
) -> AnswerabilityDecision:
    if support is None:
        legacy = tuple(unsupported or ())
        support = SupportAccounting(
            (), 0.0 if legacy else 1.0, (), legacy, len(legacy),
        )
    reasons: list[str] = []
    if support.unknown_requested_fields:
        fields = ", ".join(support.unknown_requested_fields)
        return AnswerabilityDecision(
            AnswerabilityOutcome.NO_SATISFACTORY_REPORT, 0.0,
            (f"Requested field(s) do not exist in the report catalog: {fields}.",),
        )
    if support.required_field_count and not support.required_combination_exists:
        return AnswerabilityDecision(
            AnswerabilityOutcome.NO_SATISFACTORY_REPORT, 0.0,
            ("No single report in the catalog carries all requested fields.",),
        )
    if support.ratio <= hard_support_floor and support.terms:
        terms = support.unsupported_terms[:5]
        return AnswerabilityDecision(
            AnswerabilityOutcome.NO_SATISFACTORY_REPORT, 0.0,
            ("No corpus support for: " + ", ".join(terms),),
        )
    if support.ratio < min_support_ratio and top_score < .30:
        reason = (
            f"Only {support.ratio:.0%} of your query is grounded in the catalog"
            + (f" (unsupported: {', '.join(support.unsupported_terms[:5])})"
               if support.unsupported_terms else "")
            + " and the best match is weak."
        )
        return AnswerabilityDecision(
            AnswerabilityOutcome.NO_SATISFACTORY_REPORT, 0.0, (reason,),
        )
    if support.unmet_required_fields:
        fields = ", ".join(support.unmet_required_fields)
        if len(support.unmet_required_fields) == support.required_field_count:
            return AnswerabilityDecision(
                AnswerabilityOutcome.NO_SATISFACTORY_REPORT, 0.0,
                (f"The best match is missing required field(s): {fields}.",),
            )
        return AnswerabilityDecision(
            AnswerabilityOutcome.NEEDS_CLARIFICATION, .5,
            (f"No single report carries all requested fields; the best match is missing: {fields}.",),
        )
    z = (
        intercept + top_score_weight * _clamp(top_score)
        + margin_weight * min(1.0, max(0.0, margin) / margin_scale)
        + field_coverage_weight * _clamp(field_coverage)
        + retriever_agreement_weight * _clamp(retriever_agreement)
        + parser_confidence_weight * _clamp(parser_confidence)
        + support_weight * _clamp(support.ratio)
        - ambiguity_weight * _clamp(breadth)
    )
    score = 1.0 / (1.0 + math.exp(-z))
    if (
        score >= high_threshold and margin >= min_margin
        and breadth <= breadth_threshold and support.ratio >= min_high_support
        and field_coverage >= min_field_coverage
    ):
        outcome = AnswerabilityOutcome.ANSWERABLE
    elif score < insufficient_threshold and support.ratio < min_support_ratio:
        outcome = AnswerabilityOutcome.NO_SATISFACTORY_REPORT
    else:
        outcome = AnswerabilityOutcome.NEEDS_CLARIFICATION
    if outcome is AnswerabilityOutcome.ANSWERABLE:
        reasons.append("The leading report has specific, well-supported evidence and separates from alternatives.")
    elif outcome is AnswerabilityOutcome.NO_SATISFACTORY_REPORT:
        reasons.append("Catalog grounding and ranking evidence are too weak for a satisfactory report.")
    else:
        reasons.extend(ambiguity_reasons)
        if margin < min_margin:
            reasons.append("The leading candidates are too close to choose one reliably.")
        if breadth > breadth_threshold:
            reasons.append("The request is broad enough to support several plausible interpretations.")
        if field_coverage < min_field_coverage:
            reasons.append("The leading report does not cover enough of the requested field evidence.")
        if not reasons:
            reasons.append("Evidence is not strong enough to select one report without clarification.")
    return AnswerabilityDecision(outcome, score, tuple(reasons), calibrated=False)
