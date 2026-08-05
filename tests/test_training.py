"""Training: model shapes, losses, artifact safety, and leakage guards.

The assertions that matter most here are the refusals. An artifact trained on a
different feature space, or an unapproved one, or an uncalibrated decision head,
must not be servable -- each would produce confident output computed from the
wrong thing.

Run: uv run pytest tests/test_training.py -v
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path

import numpy as np
import pytest
import torch

from reportfinder.pipeline.fuse import FUSION_FEATURE_HASH, FUSION_FEATURE_NAMES
from reportfinder.relevance.splits import SplitGuard, SplitLeakageError
from reportfinder.training import (
    DecisionHead,
    FusionMLP,
    balanced_class_weights,
    listnet_loss,
    load_decision_model,
    load_fusion_model,
    multi_positive_infonce,
    save_model,
)
from reportfinder.training.datasets import (
    _load_splits,
    cluster_near_duplicates,
    co_positive_mask,
)
from reportfinder.training.features import DECISION_FEATURE_HASH, DECISION_FEATURE_NAMES
from reportfinder.training.oof import (
    assign_folds,
    expected_calibration_error,
    fit_temperature,
)
from reportfinder.training.train_decision import (
    BUNDLE_LABEL_TO_CLASS,
    LABEL_DERIVED_COLUMNS,
    per_class_recall,
)

RELEVANCE_ROOT = Path("data/relevance")
requires_relevance = pytest.mark.skipif(
    not (RELEVANCE_ROOT / "processed" / "answerability_features.parquet").exists(),
    reason="relevance bundle not present in data/",
)


# --- architectures -----------------------------------------------------------


def test_fusion_mlp_has_the_specified_shape():
    model = FusionMLP(input_dim=len(FUSION_FEATURE_NAMES))
    linears = [m for m in model.net if isinstance(m, torch.nn.Linear)]
    assert [(layer.in_features, layer.out_features) for layer in linears] == [
        (len(FUSION_FEATURE_NAMES), 64), (64, 32), (32, 1),
    ]
    out = model(torch.zeros(7, len(FUSION_FEATURE_NAMES)))
    assert out.shape == (7,)


def test_decision_head_has_the_specified_shape():
    model = DecisionHead(input_dim=len(DECISION_FEATURE_NAMES))
    linears = [m for m in model.net if isinstance(m, torch.nn.Linear)]
    assert [(layer.in_features, layer.out_features) for layer in linears] == [
        (len(DECISION_FEATURE_NAMES), 32), (32, 16), (16, 3),
    ]
    assert model(torch.zeros(4, len(DECISION_FEATURE_NAMES))).shape == (4, 3)


def test_decision_classes_are_the_api_decisions():
    assert DecisionHead.CLASSES == (
        "RETURN_RESULTS", "ASK_CLARIFICATION", "NO_CONFIDENT_MATCH",
    )


# --- losses ------------------------------------------------------------------


def test_listnet_rewards_ranking_the_graded_positive_first():
    grades = torch.tensor([[0.0, 3.0, 0.0]])
    good = listnet_loss(torch.tensor([[0.0, 5.0, 0.0]]), grades)
    bad = listnet_loss(torch.tensor([[5.0, 0.0, 0.0]]), grades)
    assert good < bad


def test_listnet_ignores_padding():
    grades = torch.tensor([[3.0, 0.0, 0.0]])
    mask = torch.tensor([[True, True, False]])
    # Padding carries a wild score; masking must make it irrelevant.
    a = listnet_loss(torch.tensor([[5.0, 0.0, 99.0]]), grades, mask)
    b = listnet_loss(torch.tensor([[5.0, 0.0, -99.0]]), grades, mask)
    assert torch.allclose(a, b)


def test_multi_positive_infonce_accepts_several_correct_answers():
    """A vague query legitimately has several right reports."""
    q = torch.nn.functional.normalize(torch.tensor([[1.0, 0.0]]), dim=-1)
    d = torch.nn.functional.normalize(
        torch.tensor([[1.0, 0.0], [0.9, 0.1], [0.0, 1.0]]), dim=-1
    )
    positives = torch.tensor([[True, True, False]])
    loss = multi_positive_infonce(q, d, positives)
    assert torch.isfinite(loss) and loss.item() > 0


def test_a_query_with_no_positive_is_an_error_not_a_silent_skip():
    q = torch.tensor([[1.0, 0.0]])
    d = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    with pytest.raises(ValueError, match="no positive document"):
        multi_positive_infonce(q, d, torch.tensor([[False, False]]))


def test_class_weights_favour_the_rare_classes():
    labels = torch.tensor([0] * 80 + [1] * 10 + [2] * 10)
    weights = balanced_class_weights(labels)
    assert weights[0] < weights[1] and weights[0] < weights[2]


# --- leakage guards ----------------------------------------------------------


def test_co_positives_and_unjudged_are_never_negatives():
    """Both would teach the model that correct answers are wrong."""
    grades = {"R1": 2, "R2": 1, "R3": 0}
    family_of = {"R1": "f1", "R2": "f2", "R3": "f3", "R4": "f1", "R5": "f9"}.get
    mask = co_positive_mask(grades, ["R1", "R2", "R3", "R4", "R5"], family_of)
    assert mask[0] and mask[1], "graded positives are not negatives"
    assert not mask[2], "an explicit grade-0 report is a valid negative"
    assert mask[3], "another instance of a positive's family is not a negative"
    assert mask[4], "an unjudged report is not a negative"


def test_near_duplicate_queries_share_a_cluster():
    clusters = cluster_near_duplicates([
        "headcount by supervisory organization",
        "headcount by supervisory organization ",
        "payroll earnings detail",
    ])
    assert clusters[0] == clusters[1]
    assert clusters[2] != clusters[0]


def test_folds_keep_near_duplicates_together():
    """A paraphrase leaking across folds inflates every metric."""
    ids = [f"Q{i:05d}" for i in range(10)]
    clusters = [0, 0, 1, 1, 2, 2, 3, 3, 4, 4]
    assignment = assign_folds(ids, clusters, n_splits=5)
    by_cluster: dict[int, set[int]] = {}
    for cluster, fold in zip(clusters, assignment.folds.tolist(), strict=True):
        by_cluster.setdefault(cluster, set()).add(fold)
    assert all(len(folds) == 1 for folds in by_cluster.values())


def test_label_derived_columns_are_excluded_from_training():
    assert "top_candidate_is_seed_positive" in LABEL_DERIVED_COLUMNS
    assert "seed_positive_present_in_pool" in LABEL_DERIVED_COLUMNS
    assert "is_no_answer_family" in LABEL_DERIVED_COLUMNS


def test_bundle_label_mapping_is_not_the_identity():
    """The bundle encodes 2=Answerable; this codebase uses 0=RETURN_RESULTS.

    Getting this backwards trains a head that abstains on answerable queries, and
    nothing in the loss curve would reveal it.
    """
    assert BUNDLE_LABEL_TO_CLASS[2] == 0  # Answerable -> RETURN_RESULTS
    assert BUNDLE_LABEL_TO_CLASS[0] == 2  # No answer  -> NO_CONFIDENT_MATCH


# --- calibration -------------------------------------------------------------


def test_temperature_scaling_reduces_calibration_error():
    rng = np.random.default_rng(0)
    labels = rng.integers(0, 3, size=500)
    # Deliberately over-confident logits, the usual state before scaling.
    logits = np.zeros((500, 3), dtype=np.float32)
    logits[np.arange(500), labels] = 6.0
    logits += rng.normal(0, 3.0, size=logits.shape).astype(np.float32)

    def ece_at(t):
        scaled = torch.softmax(torch.from_numpy(logits) / t, dim=-1).numpy()
        return expected_calibration_error(scaled, labels)

    temperature = fit_temperature(logits, labels)
    assert temperature > 0
    assert ece_at(temperature) <= ece_at(1.0) + 1e-6


def test_per_class_recall_exposes_a_head_that_never_abstains():
    """Accuracy would read 80% here; recall shows the abstention class is dead."""
    labels = np.array([0] * 80 + [2] * 20)
    always_return = np.zeros(100, dtype=int)
    recall = per_class_recall(always_return, labels)
    assert recall["RETURN_RESULTS"] == 1.0
    assert recall["NO_CONFIDENT_MATCH"] == 0.0


# --- artifact safety ---------------------------------------------------------


def _metadata(**overrides):
    """A servable artifact's metadata.

    `approval.basis` is part of it now: the loader refuses `approved: true` with
    nothing behind it, because that flag used to be settable by hand-editing the
    `.pt` and nothing checked the edit was earned. It is granted by
    `reportfinder-train approve`, which records what it was earned on.
    """
    base = {
        "feature_hash": FUSION_FEATURE_HASH,
        "approved": True,
        "approval": {"basis": "synthetic_v2_evaluation"},
        "training_label_source": "human",
    }
    base.update(overrides)
    return base


def test_an_approved_matching_fusion_artifact_loads(tmp_path):
    model = FusionMLP(input_dim=len(FUSION_FEATURE_NAMES))
    save_model(model, tmp_path / "fusion.pt", _metadata())
    loaded, metadata = load_fusion_model(tmp_path / "fusion.pt")
    assert loaded.input_dim == len(FUSION_FEATURE_NAMES)
    assert metadata["feature_hash"] == FUSION_FEATURE_HASH


def test_a_feature_space_mismatch_is_refused(tmp_path):
    """Otherwise the model scores using the wrong columns and looks fine."""
    model = FusionMLP(input_dim=len(FUSION_FEATURE_NAMES))
    save_model(model, tmp_path / "fusion.pt", _metadata(feature_hash="deadbeef"))
    with pytest.raises(ValueError, match="wrong columns"):
        load_fusion_model(tmp_path / "fusion.pt")


def test_an_unapproved_artifact_is_refused(tmp_path):
    """This is what keeps synthetic-label training out of production."""
    model = FusionMLP(input_dim=len(FUSION_FEATURE_NAMES))
    save_model(model, tmp_path / "fusion.pt",
               _metadata(approved=False, training_label_source="synthetic_seed"))
    with pytest.raises(ValueError, match="not marked approved"):
        load_fusion_model(tmp_path / "fusion.pt")


def test_decision_artifacts_are_checked_against_their_own_feature_space(tmp_path):
    model = DecisionHead(input_dim=len(DECISION_FEATURE_NAMES))
    save_model(model, tmp_path / "d.pt",
               {"feature_hash": DECISION_FEATURE_HASH, "approved": True,
                "approval": {"basis": "synthetic_v2_evaluation"},
                "temperature": 1.2})
    loaded, metadata = load_decision_model(tmp_path / "d.pt")
    assert metadata["temperature"] == 1.2
    assert loaded.input_dim == len(DECISION_FEATURE_NAMES)


def test_a_decision_head_without_calibration_cannot_be_constructed():
    """An uncalibrated three-class output presented as a decision is the exact
    failure the deterministic fallback exists to avoid."""
    with pytest.raises(ValueError, match="refusing to serve uncalibrated"):
        DecisionHead(input_dim=4), __import__(
            "reportfinder.pipeline.decide", fromlist=["DecisionHead"]
        ).DecisionHead(DecisionHead(input_dim=4), None)


def test_saved_artifacts_carry_a_readable_sidecar(tmp_path):
    model = FusionMLP(input_dim=len(FUSION_FEATURE_NAMES))
    save_model(model, tmp_path / "fusion.pt", _metadata())
    assert (tmp_path / "fusion.json").exists()


# --- against the real bundle -------------------------------------------------


@requires_relevance
def test_decision_labels_load_with_the_corrected_mapping():
    """The bundle's 2 = Answerable is the reverse of this codebase's class order.

    (This used to also assert on the *features* the same file shipped. Those are
    gone: the head was training on 14 parquet columns while stamped with the
    20-name serving hash, so it could never have run at request time. Features
    now come from real pipeline runs and the labels are all this file provides.)
    """
    from reportfinder.training.train_decision import load_answerability_labels

    labels = load_answerability_labels(RELEVANCE_ROOT)
    assert len(labels) == 10000
    assert set(labels.values()) == {
        "RETURN_RESULTS", "ASK_CLARIFICATION", "NO_CONFIDENT_MATCH",
    }
    counts = Counter(labels.values())
    assert counts["RETURN_RESULTS"] == 8350
    assert counts["NO_CONFIDENT_MATCH"] == 700


# --- sealed-split guard ------------------------------------------------------


@pytest.mark.skipif(
    not (RELEVANCE_ROOT / "splits" / "test_queries.txt").exists(),
    reason="split files not present in data/",
)
def test_loading_the_sealed_test_split_runs_the_leakage_guard():
    """The guard must *run*, not merely be called.

    It previously received the relevance root instead of the loaded splits, so it
    raised AttributeError precisely when it was supposed to fire -- and the
    train/validation/test overlap check inside its constructor never executed.
    """
    from reportfinder.training.datasets import _load_splits, load_labelled_queries

    splits = _load_splits(RELEVANCE_ROOT)
    assert splits["test"], "the test split must be non-empty for this to mean anything"
    # The constructor is what checks for overlap; it must accept what we pass it.
    SplitGuard(splits)

    queries = load_labelled_queries(RELEVANCE_ROOT, split="test")
    assert queries


@pytest.mark.skipif(
    not (RELEVANCE_ROOT / "splits" / "test_queries.txt").exists(),
    reason="split files not present in data/",
)
def test_the_guard_rejects_sealed_ids_in_a_development_operation():
    """Proves the guard has teeth, rather than passing because it never fires."""
    splits = _load_splits(RELEVANCE_ROOT)
    sealed = sorted(splits["test"])[:3]

    with pytest.raises(SplitLeakageError):
        SplitGuard(splits).assert_allowed(sealed, "ranker_training")
