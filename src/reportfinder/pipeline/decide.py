"""The three-way decision: return, clarify, or admit no confident match.

Layered deliberately:

1. **Hard gates** (preserved verbatim from `confidence/answerability.py`). These
   fire before any scoring and encode facts, not confidence: a requested field that
   does not exist in the catalog, or a field combination no single report carries,
   cannot be answered no matter how good the retrieval looked. They are the only
   source of the specific failure messages users get today.
2. **A learned head**, when a trained *and calibrated* artifact exists. None ships.
3. **A deterministic fallback policy**, which is what actually runs.

The head replaces only step 2 -- the hand-written logistic. Replacing the gates
would lose the ability to say *why* something is unanswerable.

Nothing here calls a score a probability. A cross-encoder logit, an RRF sum and a
raw head output are all uncalibrated, and presenting any of them as a confidence
would be a stronger claim than the evidence supports.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

from ..confidence.answerability import AnswerabilityOutcome
from ..config import DEFAULT_CLARIFY_MARGIN, DEFAULT_WEAK_RERANK_LOGIT


class Decision(str, Enum):
    RETURN_RESULTS = "RETURN_RESULTS"
    ASK_CLARIFICATION = "ASK_CLARIFICATION"
    NO_CONFIDENT_MATCH = "NO_CONFIDENT_MATCH"

    @property
    def legacy_outcome(self) -> AnswerabilityOutcome:
        """The existing three-value vocabulary, for API compatibility."""
        return {
            Decision.RETURN_RESULTS: AnswerabilityOutcome.ANSWERABLE,
            Decision.ASK_CLARIFICATION: AnswerabilityOutcome.NEEDS_CLARIFICATION,
            Decision.NO_CONFIDENT_MATCH: AnswerabilityOutcome.NO_SATISFACTORY_REPORT,
        }[self]


@dataclass(frozen=True)
class Clarification:
    """One question, derived from how the leading candidates actually differ."""

    question: str
    options: tuple[str, ...]
    facet: str

    def as_dict(self) -> dict[str, object]:
        return {
            "question": self.question,
            "options": list(self.options),
            "distinguishing_facet": self.facet,
        }


@dataclass
class DecisionResult:
    decision: Decision
    reasons: list[str] = field(default_factory=list)
    clarification: Clarification | None = None
    method: str = "deterministic_three_way_policy"
    fallback_active: bool = True
    calibrated: bool = False
    # Deliberately not named "confidence" or "probability".
    score: float | None = None

    def telemetry(self) -> dict[str, object]:
        return {
            "decision": self.decision.value,
            "decision_method": self.method,
            "decision_fallback_active": self.fallback_active,
            "decision_calibrated": self.calibrated,
            "decision_reasons": list(self.reasons),
            "clarification_asked": self.clarification is not None,
        }


# Facets that genuinely separate reports in this catalog. A clarification is only
# worth asking when the leading candidates actually differ on one of these.
# The gates' reason strings. A decision carrying one of these was decided by a
# catalog fact, not by a threshold, so a threshold sweep must not try to change it.
HARD_GATE_REASONS = (
    "no authorized candidates matched the request",
    "requested field(s) do not exist in the catalog",
    "no single report carries all requested fields",
)


def was_hard_gated(result) -> bool:
    """Whether a hard gate, rather than a threshold, decided this."""
    reasons = getattr(result, "reasons", ()) or ()
    return any(
        reason.startswith(gate) for reason in reasons for gate in HARD_GATE_REASONS
    )


DISTINGUISHING_FACETS = (
    ("data_source", "Which data source should the report run against?"),
    ("category", "Which area should the report cover?"),
    ("report_type", "What kind of report do you need?"),
)


def _candidate_difference(families, corpus, limit: int = 5):
    """Find a facet on which the leading families genuinely disagree.

    Derived from the candidates rather than from a fixed question list: asking
    "current or historical?" when every leading candidate is current wastes the
    user's turn and cannot change the answer.
    """
    leading = families[:limit]
    if len(leading) < 2:
        return None

    for facet, question in DISTINGUISHING_FACETS:
        values: list[str] = []
        for family in leading:
            instance = corpus.instance(family.selected.instance_id)
            value = getattr(instance, facet, "") or ""
            if value and value not in values:
                values.append(value)
        if len(values) >= 2:
            return Clarification(
                question=question, options=tuple(values[:4]), facet=facet
            )
    return None


def decide(
    families,
    *,
    corpus,
    plan,
    risk,
    union,
    rerank,
    fusion,
    support=None,
    head=None,
    cfg=None,
) -> DecisionResult:
    """Choose between returning, clarifying, and abstaining."""
    reasons: list[str] = []

    # -- 1. hard gates -----------------------------------------------------
    # Reasons the gates emit, so an offline replay can tell a gated decision from
    # a thresholded one. Matched on text because the gates are the authority and
    # duplicating their conditions elsewhere would let the two drift.
    if not families:
        return DecisionResult(
            decision=Decision.NO_CONFIDENT_MATCH,
            reasons=["no authorized candidates matched the request"],
        )

    if support is not None:
        if getattr(support, "unknown_requested_fields", ()):
            unknown = ", ".join(sorted(support.unknown_requested_fields)[:5])
            return DecisionResult(
                decision=Decision.NO_CONFIDENT_MATCH,
                reasons=[f"requested field(s) do not exist in the catalog: {unknown}"],
            )
        if not getattr(support, "required_combination_exists", True):
            return DecisionResult(
                decision=Decision.NO_CONFIDENT_MATCH,
                reasons=["no single report carries all requested fields"],
            )

    # -- 2. learned head ---------------------------------------------------
    if head is not None:
        result = head.decide(
            families=families, plan=plan, risk=risk, union=union,
            rerank=rerank, fusion=fusion,
        )
        if result.decision is Decision.ASK_CLARIFICATION and result.clarification is None:
            grounded = _candidate_difference(families, corpus)
            if grounded is None:
                # Nothing concrete to ask about: returning results beats asking a
                # question the candidate set cannot answer.
                result.decision = Decision.RETURN_RESULTS
                result.reasons.append("no distinguishing facet to clarify")
            else:
                result.clarification = grounded
        return result

    # -- 3. deterministic fallback ----------------------------------------
    top = families[0]
    runner_up = families[1] if len(families) > 1 else None

    # The margin is taken from the cross-encoder, not from the fusion score.
    # Fusion scores are ordering scores whose spacing is an artefact of how many
    # candidates were shortlisted -- with 200 candidates every adjacent gap is
    # 1/200, so a threshold on them would fire on every query regardless of how
    # clear-cut the answer was. The reranker's scores are the only ones that carry
    # a magnitude meaning "how good a match is this, really".
    ce_top = top.selected.cross_encoder_score
    ce_second = runner_up.selected.cross_encoder_score if runner_up else None

    if ce_top is not None and ce_second is not None:
        margin = ce_top - ce_second
        margin_kind = "cross_encoder_logit_gap"
    elif runner_up is not None and top.score > 0:
        margin = (top.score - runner_up.score) / top.score
        margin_kind = "relative_fusion_score"
    else:
        margin = float("inf")
        margin_kind = "single_candidate"

    # Abstention is checked before clarification: if nothing is a good match,
    # asking the user to choose between bad options wastes their turn and cannot
    # produce a right answer.
    # Defaults here must equal RiskConfig's, or a caller that passes cfg=None
    # silently gets a different policy from the one the config documents.
    weak_threshold = (
        getattr(cfg.risk, "weak_rerank_logit", DEFAULT_WEAK_RERANK_LOGIT)
        if cfg else DEFAULT_WEAK_RERANK_LOGIT
    )
    if ce_top is not None and ce_top < weak_threshold:
        return DecisionResult(
            decision=Decision.NO_CONFIDENT_MATCH,
            reasons=[
                "the reranker judged no candidate a good match for this request "
                f"(best joint score {ce_top:.2f})"
            ],
        )

    clarify_gap = (
        getattr(cfg.risk, "clarify_margin", DEFAULT_CLARIFY_MARGIN)
        if cfg else DEFAULT_CLARIFY_MARGIN
    )
    if risk.risk_band == "HIGH" and margin < clarify_gap:
        clarification = _candidate_difference(families, corpus)
        if clarification is not None:
            return DecisionResult(
                decision=Decision.ASK_CLARIFICATION,
                reasons=[
                    "high recall risk with several comparably ranked families",
                    f"leading families differ on {clarification.facet}",
                ],
                clarification=clarification,
            )
        reasons.append("high recall risk but no distinguishing facet to ask about")

    reasons.append(
        f"{len(families)} candidate families; leading margin {margin:.2f} "
        f"({margin_kind})"
    )
    return DecisionResult(decision=Decision.RETURN_RESULTS, reasons=reasons)


def replay_fallback_policy(
    *,
    ce_top: float | None,
    ce_second: float | None,
    fusion_top: float,
    fusion_second: float,
    risk_band: str,
    has_distinguishing_facet: bool,
    weak_rerank_logit: float = DEFAULT_WEAK_RERANK_LOGIT,
    clarify_margin: float = DEFAULT_CLARIFY_MARGIN,
    has_families: bool = True,
    hard_gate_decision: str | None = None,
) -> str:
    """The fallback branch of `decide()` as a pure function of its inputs.

    Exists so the two thresholds can be swept offline. Both are post-hoc
    functions of a handful of recorded numbers, so a thousand-point grid should
    be arithmetic over one cached pipeline pass -- not a thousand pipeline runs
    at seconds each. Those two numbers were set by inspecting six probe queries
    and are the least-evidenced values in the system, so making them cheap to
    re-derive is the point.

    Correctness is not assumed: a test drives this and `decide()` over the same
    grid of inputs and requires identical decisions. If the policy changes and
    this is not updated, that test fails -- rather than the sweep quietly
    optimising a policy nobody serves.
    """
    # The hard gates fire *before* the threshold branch and are unaffected by
    # either threshold: a requested field that does not exist, or a combination no
    # single report carries, is a catalog fact. Replaying without them made the
    # sweep score a policy the pipeline does not run -- it disagreed with the
    # measured fallback by 0.12 macro recall, concentrated entirely on the
    # impossible-combination queries the gates exist to catch.
    if hard_gate_decision is not None:
        return hard_gate_decision
    if not has_families:
        return Decision.NO_CONFIDENT_MATCH.value

    if ce_top is not None and ce_second is not None:
        margin = ce_top - ce_second
    elif fusion_second and fusion_top > 0:
        margin = (fusion_top - fusion_second) / fusion_top
    else:
        margin = float("inf")

    # Abstention before clarification, matching `decide()`: if nothing is a good
    # match, asking the user to choose between bad options cannot help.
    if ce_top is not None and ce_top < weak_rerank_logit:
        return Decision.NO_CONFIDENT_MATCH.value
    if risk_band == "HIGH" and margin < clarify_margin and has_distinguishing_facet:
        return Decision.ASK_CLARIFICATION.value
    return Decision.RETURN_RESULTS.value


class DecisionHead:
    """Wraps a trained, calibrated three-class model.

    Constructing this without calibration is refused rather than allowed-with-a-
    warning: an uncalibrated three-class output presented as a decision is exactly
    the failure mode the fallback exists to avoid.
    """

    def __init__(self, model, temperature: float | None) -> None:
        if temperature is None:
            raise ValueError(
                "decision head requires a calibration temperature; refusing to "
                "serve uncalibrated three-class output"
            )
        self.model = model
        self.temperature = temperature

    def decide(self, *, families, plan, risk, union, rerank, fusion) -> DecisionResult:
        import torch

        from ..training.features import decision_features

        features = decision_features(
            families=families, plan=plan, risk=risk, union=union,
            rerank=rerank, fusion=fusion,
        )
        with torch.inference_mode():
            logits = self.model(torch.from_numpy(features).reshape(1, -1))
            # Temperature scaling fitted on a held-out calibration split; only then
            # is this a probability rather than a squashed score.
            probabilities = torch.softmax(logits / self.temperature, dim=-1)[0]

        index = int(torch.argmax(probabilities))
        decision = list(Decision)[index]
        return DecisionResult(
            decision=decision,
            reasons=["trained decision head"],
            method="neural_mlp_32_16_3",
            fallback_active=False,
            calibrated=True,
            score=float(probabilities[index]),
        )


def build_decision_head(cfg):
    """Load the trained head, or return None so the fallback policy runs.

    `calibration_path` is *read*, not merely required to be non-null. It must be
    the evaluation report for this exact model -- matched on the weights digest,
    which survives the metadata rewrite that approval performs. A path that is
    only checked for non-nullness is a checkbox, and a checkbox someone ticks to
    make an error message go away is worse than no check at all.
    """
    if cfg.decision.artifact_path is None:
        return None

    from ..training.nets import load_decision_model

    model, metadata = load_decision_model(cfg.decision.artifact_path)
    temperature = metadata.get("temperature")
    if cfg.decision.calibration_path is None or temperature is None:
        raise ValueError(
            "a decision artifact is configured but no calibration was found; "
            "set decision.calibration_path or clear decision.artifact_path"
        )

    report_path = Path(cfg.decision.calibration_path)
    if not report_path.exists():
        raise ValueError(
            f"decision.calibration_path does not exist: {report_path}. It must "
            "point at the evaluation report produced for this artifact."
        )
    try:
        report = json.loads(report_path.read_text())
    except json.JSONDecodeError as error:
        raise ValueError(
            f"decision.calibration_path is not readable JSON ({report_path}): {error}"
        ) from error

    described = (report.get("artifact") or {}).get("weights_sha256")
    actual = metadata.get("weights_sha256")
    if described != actual:
        raise ValueError(
            f"decision.calibration_path ({report_path}) describes weights "
            f"{described!r}, but the loaded artifact is {actual!r}. Refusing to "
            "serve: the calibration evidence is for a different model."
        )
    return DecisionHead(model, temperature)
