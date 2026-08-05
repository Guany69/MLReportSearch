"""The corrected training flow, and the approval gate.

Each test here pins a defect that shipped and that no existing test caught,
because every one of them produced plausible-looking numbers:

* the feature generator iterated `outcome.families`, which is empty on
  NO_CONFIDENT_MATCH -- so ~40% of queries contributed nothing, and the ones that
  vanished were the hard ones;
* the decision head trained on the sealed test split, with no guard;
* it trained on 14 parquet columns while stamping the 20-name serving hash, so
  the loader's check passed and serving would have raised;
* `approved: true` was a hand-editable flag with nothing behind it.

Run: uv run pytest tests/test_training_v2.py -v
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from reportfinder.auth import DEVELOPMENT_PRINCIPAL, AllowAllResolver, SearchRequest
from reportfinder.config import from_mapping
from reportfinder.corpus import build_corpus_model
from reportfinder.pipeline import Decision, RrfFuser, SearchPipeline
from reportfinder.query.intent import IntentParser
from reportfinder.retrieval.field_index import FieldIndex
from reportfinder.training.approve import ApprovalRefused, approve
from reportfinder.training.features import DECISION_FEATURE_NAMES
from reportfinder.training.passes import cache_key, load_passes, run_feature_pass, save_passes

from .fakes import ScriptedGenerator, ScriptedPairScorer
from .vague_outcome_fixture import frame


@pytest.fixture(scope="module")
def fixture_frame():
    return frame()


@pytest.fixture(scope="module")
def corpus(fixture_frame):
    return build_corpus_model(
        fixture_frame, ingest_mode="legacy_single_file", source_file="fixture"
    )


@pytest.fixture(scope="module")
def cfg():
    return from_mapping({
        "ingest_mode": "legacy_single_file",
        "corpus_granularity": "report_row",
        "retrieval_mode": "generators",
        "top_k": 5,
        "shortlist": {
            "standard_rerank_depth": 8, "high_risk_rerank_depth": 10,
            "quotas": {
                "fused": 4, "bm25_exclusive": 1, "splade_exclusive": 1,
                "schema_exclusive": 1, "purpose_exclusive": 1,
                "prototype_exclusive": 1, "alternate_query_exclusive": 1,
            },
        },
    })


def _generator(*instance_ids):
    class _Any(ScriptedGenerator):
        def generate(self, plan, universe, k):
            self.results = {plan.raw_query: [(i, 1.0) for i in instance_ids]}
            return super().generate(plan, universe, k)

    return _Any("bm25f", {})


def _pipeline(corpus, fixture_frame, cfg, *, scores=None, default=0.0):
    return SearchPipeline(
        cfg=cfg, corpus=corpus,
        generators=[_generator("R0105", "R0106", "R0107")],
        resolver=AllowAllResolver(allow_dev_default=True),
        intent_parser=IntentParser(fixture_frame, cfg),
        reranker=ScriptedPairScorer(scores or {}, default=default, corpus=corpus),
        fuser=RrfFuser(constant=cfg.retrieval.rrf_constant),
        field_index=FieldIndex(fixture_frame),
    )


class _Labelled:
    def __init__(self, scenario_id, query, grades):
        self.scenario_id, self.query, self.grades = scenario_id, query, grades


# --- the abstained-query hole ------------------------------------------------


def test_traced_run_keeps_families_when_the_decision_abstains(
    corpus, fixture_frame, cfg
):
    """The defect that silently dropped ~40% of training data.

    `outcome.families` is empty on NO_CONFIDENT_MATCH -- correctly, because a
    user must not be shown cards the system just said it has no confidence in.
    A feature generator reading only the outcome therefore produced nothing for
    exactly the queries a decision head most needs.
    """
    pipeline = _pipeline(corpus, fixture_frame, cfg, default=-40.0)
    state = pipeline.run_traced(SearchRequest("payroll earnings", DEVELOPMENT_PRINCIPAL))

    assert state.outcome.decision is Decision.NO_CONFIDENT_MATCH
    assert state.outcome.families == [], "the user is shown nothing"
    assert state.families, "but training still sees what the ranker produced"


def test_a_feature_pass_covers_abstained_queries(corpus, fixture_frame, cfg):
    pipeline = _pipeline(corpus, fixture_frame, cfg, default=-40.0)
    passes = run_feature_pass(
        pipeline,
        [_Labelled("Q1", "payroll earnings", {"R0105": 2})],
        principal=DEVELOPMENT_PRINCIPAL, archetypes={"Q1": "test"},
    )

    assert len(passes) == 1
    assert passes[0].fallback_decision == "NO_CONFIDENT_MATCH"
    assert len(passes[0].instance_ids) > 0, "an abstained query still yields features"


# --- one run, both feature sets ----------------------------------------------


def test_one_pass_yields_both_feature_sets_at_serving_width(
    corpus, fixture_frame, cfg
):
    pipeline = _pipeline(corpus, fixture_frame, cfg, scores={"R0105": 6.0})
    query_pass = run_feature_pass(
        pipeline, [_Labelled("Q1", "payroll earnings", {"R0105": 2})],
        principal=DEVELOPMENT_PRINCIPAL, archetypes={},
    )[0]

    # 20 is what `decide()` computes at request time. The old trainer fitted on
    # 14 parquet columns and stamped the 20-name hash anyway, so the loader
    # passed and serving would have raised a shape error.
    assert query_pass.decision_features.shape == (len(DECISION_FEATURE_NAMES),)
    assert query_pass.fusion_features.shape[0] == len(query_pass.instance_ids)
    assert query_pass.fusion_features.shape[1] == 40


def test_the_pass_uses_the_real_plan_not_a_stand_in(corpus, fixture_frame, cfg):
    """The old generator passed a `_PlanView` whose `stated_facets()` returned
    `{}`, so `query_specificity` was always 0 in training and real at serving --
    train/serve skew invisible in every metric."""
    from reportfinder.pipeline.fuse import FUSION_FEATURE_NAMES

    pipeline = _pipeline(corpus, fixture_frame, cfg, scores={"R0105": 6.0})
    query_pass = run_feature_pass(
        pipeline,
        [_Labelled("Q1", "payroll earnings by cost center", {"R0105": 2})],
        principal=DEVELOPMENT_PRINCIPAL, archetypes={},
    )[0]

    index = FUSION_FEATURE_NAMES.index("query_specificity")
    assert query_pass.fusion_features[:, index].max() > 0.0


def test_the_pass_refuses_a_pipeline_that_already_has_a_trained_fuser(
    corpus, fixture_frame, cfg
):
    """Features from a pipeline containing a trained fuser describe that fuser,
    so the next one learns to agree with the last. That is a feedback loop."""
    pipeline = _pipeline(corpus, fixture_frame, cfg)
    pipeline.fuser = object()

    with pytest.raises(ValueError, match="RRF fallback"):
        run_feature_pass(
            pipeline, [_Labelled("Q1", "payroll", {})],
            principal=DEVELOPMENT_PRINCIPAL, archetypes={},
        )


# --- the cache is keyed on what changes the answer ---------------------------


def test_cached_passes_round_trip(tmp_path, corpus, fixture_frame, cfg):
    pipeline = _pipeline(corpus, fixture_frame, cfg, scores={"R0105": 6.0})
    passes = run_feature_pass(
        pipeline, [_Labelled("Q1", "payroll earnings", {"R0105": 2})],
        principal=DEVELOPMENT_PRINCIPAL, archetypes={"Q1": "test"},
    )
    key = cache_key(bundle_version="b1", dataset_version="2.0.0", split="train")
    save_passes(passes, tmp_path, key=key)

    reloaded = load_passes(tmp_path, key=key)
    assert reloaded is not None
    assert reloaded[0].scenario_id == passes[0].scenario_id
    assert reloaded[0].grades == passes[0].grades
    assert np.allclose(reloaded[0].decision_features, passes[0].decision_features)
    assert np.allclose(reloaded[0].fusion_features, passes[0].fusion_features)


def test_a_cache_from_another_build_is_ignored(tmp_path, corpus, fixture_frame, cfg):
    """Stale features are worse than none: they are the right shape and the
    wrong numbers, so nothing downstream would notice."""
    pipeline = _pipeline(corpus, fixture_frame, cfg)
    passes = run_feature_pass(
        pipeline, [_Labelled("Q1", "payroll", {"R0105": 2})],
        principal=DEVELOPMENT_PRINCIPAL, archetypes={},
    )
    save_passes(passes, tmp_path, key=cache_key(
        bundle_version="old", dataset_version="2.0.0", split="train"))

    assert load_passes(tmp_path, key=cache_key(
        bundle_version="new", dataset_version="2.0.0", split="train")) is None


def test_the_cache_key_covers_both_feature_schemas():
    """A feature *rename* leaves the width unchanged, so only the hash catches it."""
    key = cache_key(bundle_version="b", dataset_version="d", split="train")
    from reportfinder.pipeline.fuse import FUSION_FEATURE_HASH
    from reportfinder.training.features import DECISION_FEATURE_HASH

    assert FUSION_FEATURE_HASH in key
    assert DECISION_FEATURE_HASH in key


# --- the replayed policy must equal the real one -----------------------------


@pytest.mark.parametrize("ce_top", [None, -12.0, -9.5, -8.0, 0.0, 6.0])
@pytest.mark.parametrize("ce_second", [None, -11.0, -8.5, 5.0])
@pytest.mark.parametrize("risk_band", ["LOW", "HIGH"])
def test_replayed_policy_matches_the_real_decision(
    corpus, fixture_frame, cfg, ce_top, ce_second, risk_band
):
    """The sweep optimises the replay; if it drifts from `decide()`, the sweep
    tunes a policy nobody serves."""
    from dataclasses import dataclass, field

    from reportfinder.pipeline.aggregate import FamilyResult, ScoredInstance
    from reportfinder.pipeline.decide import decide, replay_fallback_policy

    @dataclass
    class _Risk:
        risk_band: str
        reasons: list = field(default_factory=list)
        rerank_depth: int = 120
        run_late_interaction: bool = False

    def _family(family_id, instance_id, score, ce):
        record = type("R", (), {
            "instance_id": instance_id, "family_id": family_id,
            "ranks": {}, "scores": {}, "masked": set(),
            "source_exclusive": False, "found_by": set(),
            "generator_count": 0, "query_variants": set(),
        })()
        chosen = ScoredInstance(
            instance_id=instance_id, family_id=family_id, score=score, rank=1,
            admitted_via="fused", cross_encoder_score=ce, record=record,
        )
        return FamilyResult(
            family_id=family_id, canonical_title=family_id, score=score, rank=1,
            selected=chosen, instances=[chosen],
        )

    families = [_family("payroll earnings detail", "R0105", 1.0, ce_top)]
    if ce_second is not None:
        families.append(
            _family("learning course completion", "R0107", 0.5, ce_second)
        )

    real = decide(
        families, corpus=corpus, plan=None, risk=_Risk(risk_band), union=None,
        rerank=None, fusion=None, support=None, head=None, cfg=cfg,
    )
    from reportfinder.pipeline.decide import _candidate_difference

    replayed = replay_fallback_policy(
        ce_top=ce_top, ce_second=ce_second,
        fusion_top=1.0, fusion_second=0.5 if ce_second is not None else 0.0,
        risk_band=risk_band,
        has_distinguishing_facet=_candidate_difference(families, corpus) is not None,
        weak_rerank_logit=cfg.risk.weak_rerank_logit,
        clarify_margin=cfg.risk.clarify_margin,
    )
    assert replayed == real.decision.value


# --- the approval gate has teeth ---------------------------------------------


def _artifact(tmp_path, *, name="fusion.pt", architecture="pytorch_mlp_64_32_1"):
    from reportfinder.pipeline.fuse import FUSION_FEATURE_HASH
    from reportfinder.training.nets import FusionMLP, save_model

    model = FusionMLP(input_dim=40)
    path = tmp_path / name
    save_model(model, path, {
        "architecture": architecture,
        "feature_hash": FUSION_FEATURE_HASH,
        "approved": False,
        "human_validated": False,
        "training_label_source": "synthetic_v2_generative",
    })
    return path


def _report(tmp_path, artifact, *, delta=0.05, split="validation", digest=None):
    import torch

    from reportfinder.pipeline.fuse import FUSION_FEATURE_HASH
    from reportfinder.training.nets import weights_sha256

    payload = torch.load(artifact, map_location="cpu", weights_only=False)
    path = tmp_path / "eval.json"
    path.write_text(json.dumps({
        "schema_version": "1", "kind": "fusion",
        "artifact": {
            "path": str(artifact),
            "weights_sha256": digest or weights_sha256(payload["state_dict"]),
            "feature_hash": FUSION_FEATURE_HASH,
        },
        "dataset": {"label_source": "synthetic_v2_generative"},
        "split": split, "query_count": 265,
        "primary_metric": {
            "name": "family_ndcg@5", "candidate": 0.71,
            "fallback": round(0.71 - delta, 4), "delta": delta,
        },
    }))
    return path


def test_approve_refuses_a_report_about_a_different_model(tmp_path):
    artifact = _artifact(tmp_path)
    report = _report(tmp_path, artifact, digest="deadbeef")

    with pytest.raises(ApprovalRefused, match="different model"):
        approve(artifact, report)


def test_approve_refuses_when_the_model_only_ties_its_fallback(tmp_path):
    """The fallback is simpler, already serving, and has no artifact to keep in
    sync. A tie is not a reason to take on a model."""
    artifact = _artifact(tmp_path)

    with pytest.raises(ApprovalRefused, match="delta"):
        approve(artifact, _report(tmp_path, artifact, delta=0.0))


def test_approve_refuses_a_report_from_the_fitting_split(tmp_path):
    artifact = _artifact(tmp_path)

    with pytest.raises(ApprovalRefused, match="validation"):
        approve(artifact, _report(tmp_path, artifact, split="train"))


def test_approve_stamps_a_basis_and_keeps_human_validated_false(tmp_path):
    artifact = _artifact(tmp_path)
    metadata = approve(artifact, _report(tmp_path, artifact, delta=0.05))

    assert metadata["approved"] is True
    assert metadata["approval"]["basis"] == "synthetic_v2_evaluation"
    # Beating a fallback on synthetic labels does not make them human ones.
    assert metadata["human_validated"] is False
    assert metadata["approval"]["primary_metric"]["delta"] == 0.05


def test_a_refused_approval_leaves_the_artifact_untouched(tmp_path):
    import torch

    artifact = _artifact(tmp_path)
    before = torch.load(artifact, map_location="cpu", weights_only=False)["metadata"]

    with pytest.raises(ApprovalRefused):
        approve(artifact, _report(tmp_path, artifact, delta=0.0))

    after = torch.load(artifact, map_location="cpu", weights_only=False)["metadata"]
    assert after["approved"] is False
    assert after == before


def test_the_loader_refuses_a_hand_edited_approval(tmp_path):
    """`approved: true` with nothing behind it is what the gate used to be."""
    import torch

    from reportfinder.training.nets import load_fusion_model

    artifact = _artifact(tmp_path)
    payload = torch.load(artifact, map_location="cpu", weights_only=False)
    payload["metadata"]["approved"] = True          # ...and no approval block
    torch.save(payload, artifact)

    with pytest.raises(ValueError, match="approval basis"):
        load_fusion_model(artifact)


def test_an_approved_artifact_loads(tmp_path):
    from reportfinder.training.nets import load_fusion_model

    artifact = _artifact(tmp_path)
    approve(artifact, _report(tmp_path, artifact, delta=0.05))

    model, metadata = load_fusion_model(artifact)
    assert model.input_dim == 40
    assert metadata["approval"]["basis"] == "synthetic_v2_evaluation"


def test_approve_matches_kind_by_architecture_not_substring(tmp_path):
    """`pytorch_mlp_32_16_3` contains neither "fusion" nor "decision".

    A substring test therefore classified every artifact as a fusion model, and
    refused every legitimate decision approval with a message about the wrong
    thing. Only running the real flow surfaced it.
    """
    from reportfinder.training.features import (
        DECISION_FEATURE_HASH,
        DECISION_FEATURE_NAMES,
    )
    from reportfinder.training.nets import (
        DECISION_ARCHITECTURE,
        DecisionHead,
        save_model,
        weights_sha256,
    )

    model = DecisionHead(input_dim=len(DECISION_FEATURE_NAMES))
    artifact = tmp_path / "decision.pt"
    save_model(model, artifact, {
        "architecture": DECISION_ARCHITECTURE,
        "feature_hash": DECISION_FEATURE_HASH,
        "approved": False, "human_validated": False, "temperature": 1.1,
    })

    import torch

    digest = weights_sha256(
        torch.load(artifact, map_location="cpu", weights_only=False)["state_dict"]
    )
    report = tmp_path / "decision_eval.json"
    report.write_text(json.dumps({
        "schema_version": "1", "kind": "decision",
        "artifact": {"weights_sha256": digest, "feature_hash": DECISION_FEATURE_HASH},
        "dataset": {"label_source": "synthetic_v2_generative"},
        "split": "validation", "query_count": 265,
        "primary_metric": {
            "name": "macro_per_class_recall", "candidate": 0.80,
            "fallback": 0.71, "delta": 0.09,
        },
        "metrics": {"candidate": {"accuracy": 0.8}, "fallback": {"accuracy": 0.7}},
    }))

    metadata = approve(artifact, report)
    assert metadata["approved"] is True


def test_approve_refuses_a_model_that_regresses_a_secondary_metric(tmp_path):
    """The case the real fusion run hit: +0.0031 on the headline while
    instance selection went backwards. A positive primary delta is not, alone,
    evidence that a model is better."""
    from reportfinder.pipeline.fuse import FUSION_FEATURE_HASH
    from reportfinder.training.nets import FusionMLP, save_model, weights_sha256

    model = FusionMLP(input_dim=40)
    artifact = tmp_path / "fusion.pt"
    save_model(model, artifact, {
        "architecture": "pytorch_mlp_64_32_1",
        "feature_hash": FUSION_FEATURE_HASH,
        "approved": False, "human_validated": False,
    })

    import torch

    digest = weights_sha256(
        torch.load(artifact, map_location="cpu", weights_only=False)["state_dict"]
    )
    report = tmp_path / "eval.json"
    report.write_text(json.dumps({
        "schema_version": "1", "kind": "fusion",
        "artifact": {"weights_sha256": digest, "feature_hash": FUSION_FEATURE_HASH},
        "dataset": {"label_source": "synthetic_v2_generative"},
        "split": "validation", "query_count": 265,
        "primary_metric": {
            "name": "family_ndcg@5", "candidate": 0.1095,
            "fallback": 0.1064, "delta": 0.0031,
        },
        "metrics": {
            "candidate": {"family_ndcg@5": 0.1095,
                          "instance_recall@1_within_family": 0.8684},
            "fallback": {"family_ndcg@5": 0.1064,
                         "instance_recall@1_within_family": 0.8947},
        },
    }))

    with pytest.raises(ApprovalRefused, match="instance_recall@1_within_family"):
        approve(artifact, report)

    # ...unless the trade is asked for explicitly.
    assert approve(artifact, report, allow_regressions=True)["approved"] is True


# --- the replay must equal the real pipeline, on real data -------------------


CACHE_ROOT = Path("artifacts/feature_cache")


def _cached_validation_passes():
    from reportfinder.training.passes import cache_key, load_passes

    key = cache_key(
        bundle_version="b-d196fdc6dba2f2ba-021cf6db",
        dataset_version="2.0.0", split="validation",
    )
    return load_passes(CACHE_ROOT / key, key=key)


@pytest.mark.skipif(
    _cached_validation_passes() is None,
    reason="no cached feature pass; run `reportfinder-train fusion` first",
)
def test_the_replayed_policy_reproduces_every_real_decision():
    """The check that caught the sweep scoring a policy nobody runs.

    `decide()` applies hard gates -- a requested field that does not exist, a
    combination no report carries -- *before* the threshold branch, and no
    threshold moves them. Replaying without them disagreed with the pipeline on
    0.12 macro recall, concentrated entirely on the impossible-combination
    queries the gates exist to catch, and would have tuned the thresholds against
    a policy that does not exist.

    Comparing against the decisions the pipeline actually made is the only check
    that could have found it; a synthetic grid cannot, because the gates depend
    on catalog facts a fixture does not have.
    """
    from reportfinder.config import DEFAULT_CLARIFY_MARGIN, DEFAULT_WEAK_RERANK_LOGIT
    from reportfinder.pipeline.decide import replay_fallback_policy

    passes = _cached_validation_passes()
    assert passes, "cache present but empty"

    mismatches = []
    for query_pass in passes:
        replayed = replay_fallback_policy(
            ce_top=query_pass.ce_top, ce_second=query_pass.ce_second,
            fusion_top=query_pass.fusion_top, fusion_second=query_pass.fusion_second,
            risk_band=query_pass.risk_band,
            has_distinguishing_facet=query_pass.has_distinguishing_facet,
            weak_rerank_logit=DEFAULT_WEAK_RERANK_LOGIT,
            clarify_margin=DEFAULT_CLARIFY_MARGIN,
            has_families=bool(query_pass.families_ranked),
            hard_gate_decision=query_pass.hard_gate_decision,
        )
        if replayed != query_pass.fallback_decision:
            mismatches.append(
                (query_pass.scenario_id, query_pass.archetype,
                 replayed, query_pass.fallback_decision)
            )

    assert not mismatches, (
        f"{len(mismatches)}/{len(passes)} replayed decisions disagree with what "
        f"the pipeline actually decided, e.g. {mismatches[:3]}"
    )


@pytest.mark.skipif(
    _cached_validation_passes() is None,
    reason="no cached feature pass; run `reportfinder-train fusion` first",
)
def test_hard_gated_queries_are_recorded_as_such():
    passes = _cached_validation_passes()
    gated = [p for p in passes if p.hard_gate_decision is not None]

    assert gated, "the estate has impossible-combination queries; some must gate"
    assert all(p.hard_gate_decision == "NO_CONFIDENT_MATCH" for p in gated)
    assert all(p.fallback_decision == p.hard_gate_decision for p in gated)
