"""Regressions for defects found by inspection rather than by failing tests.

Each of these was live: a guard that could never fire, a config default that
disagreed with the code reading it, a manifest field populated from the wrong
variable. None of them broke a test, which is exactly why they need one.

Run: uv run pytest tests/test_defect_regressions.py -v
"""

from __future__ import annotations

import json

import pytest

from reportfinder.config import (
    DEFAULT_CLARIFY_MARGIN,
    DEFAULT_GENERATOR_WORKERS,
    DEFAULT_WEAK_RERANK_LOGIT,
    RetrievalConfig,
    RiskConfig,
)
from reportfinder.pipeline.shortlist import EXCLUSIVE_QUOTA_SOURCES

# --- config defaults cannot drift from the code that reads them --------------


def test_decision_thresholds_have_one_source_of_truth():
    """`decide()` read these with getattr defaults of its own.

    They disagreed: the fallback used -5.0 while the config said -9.0, so a
    caller passing cfg=None silently ran a stricter policy than the one the
    config documents.
    """
    risk = RiskConfig()
    assert risk.weak_rerank_logit == DEFAULT_WEAK_RERANK_LOGIT
    assert risk.clarify_margin == DEFAULT_CLARIFY_MARGIN


def test_generator_worker_default_has_one_source_of_truth():
    """Same divergence: the orchestrator's getattr default was 1, config's 6."""
    assert RetrievalConfig().max_generator_workers == DEFAULT_GENERATOR_WORKERS


def test_the_dead_risk_knob_is_gone():
    """`low_specificity_idf` was configurable and consumed nowhere -- a knob that
    silently does nothing is worse than no knob."""
    assert not hasattr(RiskConfig(), "low_specificity_idf")


# --- every generator gets a reserved shortlist floor -------------------------


def test_every_quota_source_names_a_real_generator_or_variant():
    for quota, (kind, key) in EXCLUSIVE_QUOTA_SOURCES.items():
        assert kind in {"generator", "variant"}, quota
        assert key, quota


def test_late_interaction_has_a_reserved_floor():
    """It had none. Enabling it would have put late-interaction-only candidates
    into open competition for `fused` -- the exact loss the quota system exists
    to prevent, and invisible until someone measured recall."""
    assert EXCLUSIVE_QUOTA_SOURCES["late_interaction_exclusive"] == (
        "generator", "late_interaction",
    )


# --- lexicon `kind` is validated at load ------------------------------------


def test_an_unknown_emission_kind_is_rejected_at_load(tmp_path):
    """It used to be `str(...)` and trusted, so a misspelled kind produced a
    concept no vocabulary lookup could resolve -- a silent dead rule."""
    from reportfinder.query.expansion.rules import load_lexicon

    (tmp_path / "bad.yaml").write_text(
        "schema: 1\nrules:\n"
        "  - id: broken\n    phrases: ['head count']\n"
        "    emits:\n      - {kind: fields, canonical: Headcount}\n"
    )

    class _Vocab:
        def canonical(self, kind, name):
            return name

    with pytest.raises(ValueError, match="unknown kind"):
        load_lexicon(_Vocab(), directory=tmp_path)


# --- the sealed split guard --------------------------------------------------


def test_split_guard_still_blocks_development_operations():
    """The load-time call used an operation name outside DEVELOPMENT_OPERATIONS,
    so it could never raise. The protection has to live where the operation
    happens; this pins that it works there."""
    from reportfinder.relevance.splits import SplitGuard, SplitLeakageError

    guard = SplitGuard({"train": {"Q1"}, "validation": {"Q2"}, "test": {"Q3"}})
    with pytest.raises(SplitLeakageError):
        guard.assert_allowed(["Q3"], "ranker_training")
    # ...and permits the operation the sealed split exists for.
    guard.assert_allowed(["Q3"], "final_evaluation")


def test_split_guard_rejects_overlapping_splits():
    from reportfinder.relevance.splits import SplitGuard, SplitLeakageError

    with pytest.raises(SplitLeakageError):
        SplitGuard({"train": {"Q1"}, "test": {"Q1"}})


def test_split_guard_rejects_calibration_overlap():
    """Calibration was the one split the overlap check skipped. A temperature
    fitted on rows the model trained on is not a calibration."""
    from reportfinder.relevance.splits import SplitGuard, SplitLeakageError

    with pytest.raises(SplitLeakageError, match="train/calibration"):
        SplitGuard({"train": {"Q1"}, "calibration": {"Q1"}})
    with pytest.raises(SplitLeakageError, match="test/calibration"):
        SplitGuard({"test": {"Q1"}, "calibration": {"Q1"}})


def test_split_guard_refuses_an_unknown_operation_name():
    """A misspelled operation used to disable the guard silently -- the exact
    defect this section exists to pin, one level up."""
    from reportfinder.relevance.splits import SplitGuard

    guard = SplitGuard({"train": {"Q1"}, "test": {"Q3"}})
    with pytest.raises(ValueError, match="unknown split-guard operation"):
        guard.assert_allowed(["Q3"], "ranker_trainingg")


# --- artifact weight digests -------------------------------------------------


def test_weights_digest_ignores_metadata(tmp_path):
    """Approval rewrites metadata. A file digest would change with it and break
    every reference; the digest must describe the weights alone."""
    import torch

    from reportfinder.training.nets import FusionMLP, save_model, weights_sha256

    model = FusionMLP(input_dim=4)
    before = weights_sha256(model.state_dict())

    save_model(model, tmp_path / "m.pt", {"feature_hash": "x", "approved": False})
    payload = torch.load(tmp_path / "m.pt", map_location="cpu", weights_only=False)
    assert payload["metadata"]["weights_sha256"] == before

    # Restamp exactly as `approve` would, and the digest must be unchanged.
    payload["metadata"]["approved"] = True
    torch.save(payload, tmp_path / "m.pt")
    reloaded = torch.load(tmp_path / "m.pt", map_location="cpu", weights_only=False)
    assert weights_sha256(reloaded["state_dict"]) == before


def test_weights_digest_changes_when_weights_change():
    import torch

    from reportfinder.training.nets import FusionMLP, weights_sha256

    model = FusionMLP(input_dim=4)
    before = weights_sha256(model.state_dict())
    with torch.no_grad():
        next(iter(model.parameters())).add_(1.0)
    assert weights_sha256(model.state_dict()) != before


# --- calibration_path is read, not merely non-null ---------------------------


def _decision_artifact(tmp_path, *, temperature=1.5):
    from reportfinder.training.features import DECISION_FEATURE_HASH, DECISION_FEATURE_NAMES
    from reportfinder.training.nets import DecisionHead, save_model

    model = DecisionHead(input_dim=len(DECISION_FEATURE_NAMES))
    path = tmp_path / "decision.pt"
    save_model(model, path, {
        "feature_hash": DECISION_FEATURE_HASH,
        "approved": True,
        "approval": {"basis": "test"},
        "temperature": temperature,
    })
    return path


def _cfg(tmp_path, artifact, calibration):
    from reportfinder.config import from_mapping

    return from_mapping({
        "decision": {
            "artifact_path": str(artifact),
            "calibration_path": str(calibration) if calibration else None,
        },
    })


def test_calibration_path_must_describe_the_served_artifact(tmp_path):
    """The knob was required to be non-null and then never opened -- a checkbox.
    A checkbox someone ticks to silence an error is worse than no check."""
    import torch

    from reportfinder.pipeline.decide import build_decision_head

    artifact = _decision_artifact(tmp_path)
    digest = torch.load(artifact, map_location="cpu", weights_only=False)[
        "metadata"]["weights_sha256"]

    matching = tmp_path / "eval.json"
    matching.write_text(json.dumps({"artifact": {"weights_sha256": digest}}))
    assert build_decision_head(_cfg(tmp_path, artifact, matching)) is not None

    wrong = tmp_path / "other.json"
    wrong.write_text(json.dumps({"artifact": {"weights_sha256": "deadbeef"}}))
    with pytest.raises(ValueError, match="different model"):
        build_decision_head(_cfg(tmp_path, artifact, wrong))


def test_a_missing_calibration_report_is_refused(tmp_path):
    from reportfinder.pipeline.decide import build_decision_head

    artifact = _decision_artifact(tmp_path)
    with pytest.raises(ValueError, match="does not exist"):
        build_decision_head(_cfg(tmp_path, artifact, tmp_path / "absent.json"))
