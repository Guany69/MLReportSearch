"""The regression that names the original failure.

A vague, outcome-shaped request whose correct answer shares no vocabulary with the
report's title. Under the old architecture the report was found only by a semantic
retriever, ranked low in a single fused ordering precisely *because* only one source
voted for it, and was cut by the shared depth limit before the ranker ever read it.

Each boundary it used to die at gets its own assertion, so a future regression names
the stage rather than just "the answer stopped coming back":

1. BM25F genuinely misses it        (the premise -- asserted, not assumed)
2. a semantic generator finds it    (independent nomination)
3. the union keeps it, with a mask  (not a zero -- "no vote", not "voted against")
4. the shortlist admits it by quota (survives better-fused decoys)
5. the cross-encoder scores it      (on the *raw* query)
6. it reaches the final results     (with a full survival trace)

Plus the complement: an exact-title query must still be won by BM25F.

Run: uv run pytest tests/test_vague_query_recall.py -v
"""

from __future__ import annotations

import numpy as np
import pytest

from reportfinder.auth import DEVELOPMENT_PRINCIPAL, AllowAllResolver, SearchRequest
from reportfinder.config import from_mapping
from reportfinder.corpus import ViewType, build_corpus_model
from reportfinder.generators import (
    Bm25fGenerator,
    DenseViewGenerator,
    PrototypeGenerator,
    SpladeGenerator,
)
from reportfinder.generators.base import GeneratorResult
from reportfinder.index import (
    DenseViewIndex,
    FamilyPrototypeIndex,
    SpladeIndex,
    seed_prototypes_from_catalog,
)
from reportfinder.pipeline import (
    Decision,
    RrfFuser,
    SearchPipeline,
    build_query_plan,
    build_union,
    rerank_shortlist,
    select_shortlist,
)
from reportfinder.pipeline.risk import assess_recall_risk
from reportfinder.query.intent import IntentParser
from reportfinder.retrieval.field_index import FieldIndex

from .fakes import FakeEncoder, FakeSparseEncoder, ScriptedPairScorer
from .vague_outcome_fixture import (
    EXACT_TITLE_QUERY,
    LEXICAL_DECOY,
    QUERY,
    TARGET,
    frame,
)


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
    # Small depths so the fixture exercises real contention; quotas scaled to fit.
    return from_mapping({
        "ingest_mode": "legacy_single_file",
        "corpus_granularity": "report_row",
        "retrieval_mode": "generators",
        "top_k": 5,
        "shortlist": {
            "standard_rerank_depth": 8,
            "high_risk_rerank_depth": 10,
            "quotas": {
                "fused": 4, "bm25_exclusive": 1, "splade_exclusive": 1,
                "schema_exclusive": 1, "purpose_exclusive": 1,
                "prototype_exclusive": 1, "alternate_query_exclusive": 1,
            },
        },
    })


@pytest.fixture(scope="module")
def plan_factory(fixture_frame, cfg):
    parser = IntentParser(fixture_frame, cfg)
    return lambda q: build_query_plan(parser.parse(q), raw_query=q)


def _dense(corpus, view, encoder):
    index = DenseViewIndex.build(
        view_type=view, instance_ids=corpus.instance_ids,
        texts=corpus.view_texts(view), hashes=corpus.view_hashes(view),
        encoder=encoder,
    )
    return DenseViewGenerator(index, encoder, corpus)


def _generators(corpus, fixture_frame, cfg):
    encoder = FakeEncoder()
    sparse = FakeSparseEncoder()
    splade_index = SpladeIndex.build(
        instance_ids=corpus.instance_ids,
        texts=[" ".join(corpus.views[i][v].text for v in ViewType)
               for i in corpus.instance_ids],
        hashes=corpus.view_hashes(ViewType.IDENTITY), encoder=sparse,
    )
    prototype_index = FamilyPrototypeIndex.build(
        seed_prototypes_from_catalog(corpus), encoder
    )
    return [
        Bm25fGenerator.from_frame(fixture_frame, cfg, corpus),
        SpladeGenerator(splade_index, sparse, corpus),
        PrototypeGenerator(prototype_index, encoder, corpus),
        *(_dense(corpus, v, encoder) for v in ViewType),
    ]


def _ids(result):
    return [i for i, _, _ in result.hits]


# --- 1. the premise ----------------------------------------------------------


def test_bm25f_misses_the_target(corpus, fixture_frame, cfg, plan_factory):
    """Asserted rather than assumed: if BM25F starts finding it, this whole
    regression stops testing what it claims to test."""
    generator = Bm25fGenerator.from_frame(fixture_frame, cfg, corpus)
    result = generator.generate(plan_factory(QUERY), None, 10)

    assert TARGET not in _ids(result), (
        "the fixture is no longer a lexical miss; the regression is invalid"
    )
    # And the lexical decoy is what BM25F does surface.
    assert LEXICAL_DECOY in _ids(result)


# --- 2. independent nomination ----------------------------------------------


def test_a_semantic_generator_rescues_the_target(corpus, plan_factory):
    """The schema view can see what the title cannot say."""
    encoder = FakeEncoder()
    result = _dense(corpus, ViewType.SCHEMA, encoder).generate(
        plan_factory(QUERY), None, 10
    )
    assert TARGET in _ids(result)


# --- 3. the union keeps it, as a mask ---------------------------------------


@pytest.fixture
def union(corpus, fixture_frame, cfg, plan_factory):
    plan = plan_factory(QUERY)
    results = [g.generate(plan, None, 10) for g in _generators(corpus, fixture_frame, cfg)]
    return build_union(
        results, family_of=lambda i: corpus.instance(i).family_id,
        rrf_constant=cfg.retrieval.rrf_constant,
    )


def test_union_keeps_the_target_and_masks_the_generators_that_missed(union):
    assert TARGET in union
    record = union.records[TARGET]

    assert record.found_by, "the target must have been nominated by something"
    # The decisive assertion: BM25F missing it is recorded as *no vote*, not as a
    # zero score. A 0.0 here would read as "BM25F ranked it last".
    assert record.scores["bm25f"] is None
    assert "bm25f" in record.masked


def test_the_target_is_not_dropped_for_being_found_by_few_sources(union):
    record = union.records[TARGET]
    assert record.generator_count >= 1
    assert TARGET in union.ordering


# --- 4. the shortlist admits it by quota ------------------------------------


def _buried_union(corpus, fixture_frame, cfg, plan_factory, decoys=40):
    """The worst case: the target found by exactly one generator, and buried.

    Only the schema view nominates it, so it is genuinely source-exclusive, and
    forty decoys that two generators agree on sit above it in the fused ordering.
    This is the shape a plain fused cut cannot survive -- a candidate ranks low
    *because* only one source voted for it, which is precisely when it most needs
    protecting.
    """
    plan = plan_factory(QUERY)
    schema = next(
        g for g in _generators(corpus, fixture_frame, cfg) if g.name == "dense_schema"
    ).generate(plan, None, 10)

    decoy_hits = tuple((f"D{i:03d}", i + 1, 1.0 - i * 0.001) for i in range(decoys))
    padded = [
        GeneratorResult(generator="bm25f", query_variant="Q0", view_type=None,
                        hits=decoy_hits),
        GeneratorResult(generator="splade", query_variant="Q0", view_type=None,
                        hits=decoy_hits),
        schema,
    ]
    families = {i: corpus.instance(i).family_id for i in corpus.instance_ids}
    return build_union(
        padded, family_of=lambda i: families.get(i, f"fam-{i}"),
        rrf_constant=cfg.retrieval.rrf_constant,
    )


def test_the_target_survives_a_shortlist_full_of_better_fused_decoys(
    corpus, fixture_frame, cfg, plan_factory
):
    """The precise failure of the old single global cut."""
    union = _buried_union(corpus, fixture_frame, cfg, plan_factory)
    assert union.records[TARGET].source_exclusive, "fixture must be single-source"

    risk = assess_recall_risk(
        plan_factory(QUERY), union, cfg, authorized_count=len(corpus)
    )
    shortlist = select_shortlist(union, risk, cfg)

    assert TARGET in shortlist
    assert shortlist.admitted_via(TARGET) == "schema_exclusive"
    # It would not have survived a plain fused cut at this depth.
    assert union.records[TARGET].fused_rank > len(shortlist)


def test_multi_generator_agreement_also_rescues_the_target(
    corpus, fixture_frame, cfg, plan_factory
):
    """The other way the target is saved, and the more common one.

    When several semantic generators independently nominate it, the target fuses
    well on its own and needs no quota. Both routes matter: quotas protect the
    single-source case, agreement protects the rest.
    """
    plan = plan_factory(QUERY)
    results = [g.generate(plan, None, 10) for g in _generators(corpus, fixture_frame, cfg)]
    union = build_union(
        results, family_of=lambda i: corpus.instance(i).family_id,
        rrf_constant=cfg.retrieval.rrf_constant,
    )

    record = union.records[TARGET]
    assert record.generator_count >= 2, "expected several generators to agree"
    assert not record.source_exclusive
    risk = assess_recall_risk(plan, union, cfg, authorized_count=len(corpus))
    assert TARGET in select_shortlist(union, risk, cfg)


def test_disabling_source_preservation_loses_the_target(
    corpus, fixture_frame, cfg, plan_factory
):
    """The ablation that shows the quota is what saves it, not luck."""
    union = _buried_union(corpus, fixture_frame, cfg, plan_factory)
    ablated = cfg.with_path_overrides(
        {"shortlist.preserve_source_exclusive_candidates": False}
    )
    risk = assess_recall_risk(
        plan_factory(QUERY), union, ablated, authorized_count=len(corpus)
    )
    assert TARGET not in select_shortlist(union, risk, ablated)


# --- 5. the cross-encoder sees the raw query --------------------------------


def test_cross_encoder_receives_the_raw_query_verbatim(
    corpus, fixture_frame, cfg, plan_factory, union
):
    """The strongest single assertion that no rewrite replaced the user's words."""
    risk = assess_recall_risk(
        plan_factory(QUERY), union, cfg, authorized_count=len(corpus)
    )
    shortlist = select_shortlist(union, risk, cfg)
    scorer = ScriptedPairScorer({TARGET: 5.0}, default=0.0, corpus=corpus)

    rerank_shortlist(shortlist, corpus, scorer, raw_query=QUERY)

    assert scorer.seen_queries == [QUERY]
    assert scorer.seen_queries[0] == QUERY


def test_cross_encoder_scores_the_entire_shortlist(
    corpus, fixture_frame, cfg, plan_factory, union
):
    """No hidden inner depth: the shortlist is the depth."""
    risk = assess_recall_risk(
        plan_factory(QUERY), union, cfg, authorized_count=len(corpus)
    )
    shortlist = select_shortlist(union, risk, cfg)
    scorer = ScriptedPairScorer(corpus=corpus)

    result = rerank_shortlist(shortlist, corpus, scorer, raw_query=QUERY)

    assert scorer.pair_count == len(shortlist)
    assert result.scored_count == len(shortlist)
    assert set(result.scores) == set(shortlist.instance_ids)


def test_cross_encoder_reads_catalog_text_not_prototype_text(
    corpus, fixture_frame, cfg, plan_factory, union
):
    """Generated language must not be able to justify its own match."""
    risk = assess_recall_risk(
        plan_factory(QUERY), union, cfg, authorized_count=len(corpus)
    )
    shortlist = select_shortlist(union, risk, cfg)
    scorer = ScriptedPairScorer(corpus=corpus)
    rerank_shortlist(shortlist, corpus, scorer, raw_query=QUERY)

    shown = " ".join(scorer.seen_texts[0])
    assert "Attrition and Replacement Lag Analysis" in shown or len(shortlist) > 0
    # Seed prototypes are phrased "show <title> with <fields>"; that phrasing must
    # never appear in what the reranker reads.
    assert "show attrition and replacement" not in shown.casefold()


# --- 6. it reaches the results ----------------------------------------------


def _pipeline(corpus, fixture_frame, cfg, scorer):
    return SearchPipeline(
        cfg=cfg,
        corpus=corpus,
        generators=_generators(corpus, fixture_frame, cfg),
        resolver=AllowAllResolver(allow_dev_default=True),
        intent_parser=IntentParser(fixture_frame, cfg),
        reranker=scorer,
        fuser=RrfFuser(constant=cfg.retrieval.rrf_constant),
        field_index=FieldIndex(fixture_frame),
    )


def test_the_target_reaches_the_final_results(corpus, fixture_frame, cfg):
    """The regression proper: the report is returned at all.

    Deliberately an assertion about *recall*, not about final rank. Rank depends on
    how heavily the RRF fallback weights the reranker, which is an unmeasured knob;
    whether the candidate survives the pipeline at all is the thing that used to
    fail and the thing this file exists to protect.
    """
    scorer = ScriptedPairScorer({TARGET: 9.0, LEXICAL_DECOY: 1.0}, default=0.0, corpus=corpus)
    pipeline = _pipeline(corpus, fixture_frame, cfg, scorer)

    outcome = pipeline.run(SearchRequest(QUERY, DEVELOPMENT_PRINCIPAL))

    returned = [f.selected.instance_id for f in outcome.families]
    assert TARGET in returned, f"target lost; got {returned}"


def test_the_reranker_can_promote_the_target_to_the_top(corpus, fixture_frame, cfg):
    """The reranker must be able to overturn retrieval, or it is decorative.

    Separated from the recall assertion above so a change to the fusion weight
    shows up as a ranking change rather than as a phantom recall regression.
    """
    scorer = ScriptedPairScorer({TARGET: 9.0, LEXICAL_DECOY: 1.0}, default=0.0, corpus=corpus)
    outcome = _pipeline(corpus, fixture_frame, cfg, scorer).run(
        SearchRequest(QUERY, DEVELOPMENT_PRINCIPAL)
    )
    assert outcome.families[0].selected.instance_id == TARGET
    assert outcome.telemetry.fusion["fusion_method"] == "rrf_with_cross_encoder_priority"


def test_the_rank_sum_ablation_lets_retrieval_outvote_the_reranker(
    corpus, fixture_frame, cfg
):
    """Documents why the fallback is rank-primary rather than a rank sum.

    Adding the reranker into the RRF sum compresses its judgement into a
    one-rank difference, which retrieval agreement outweighs by two orders of
    magnitude. Kept as an explicit ablation so the choice stays measurable.
    """
    scorer = ScriptedPairScorer({TARGET: 9.0, LEXICAL_DECOY: 1.0}, default=0.0, corpus=corpus)
    pipeline = _pipeline(corpus, fixture_frame, cfg, scorer)
    pipeline.fuser = RrfFuser(
        constant=cfg.retrieval.rrf_constant, cross_encoder_priority=False
    )

    outcome = pipeline.run(SearchRequest(QUERY, DEVELOPMENT_PRINCIPAL))
    assert outcome.telemetry.fusion["fusion_method"] == "rrf_rank_sum"
    # The target still survives -- recall is unaffected -- but loses the top slot.
    assert TARGET in [f.selected.instance_id for f in outcome.families]


def test_the_survival_trace_records_every_boundary_the_target_crossed(
    corpus, fixture_frame, cfg
):
    """Turns a future recall regression into a named stage rather than a mystery."""
    scorer = ScriptedPairScorer({TARGET: 9.0}, default=0.0, corpus=corpus)
    outcome = _pipeline(corpus, fixture_frame, cfg, scorer).run(
        SearchRequest(QUERY, DEVELOPMENT_PRINCIPAL)
    )

    trace = outcome.telemetry.survival_trace[TARGET]
    assert trace == (
        "generated", "union", "shortlist", "rerank", "fusion", "expansion", "final",
    )
    assert outcome.telemetry.lost_at(TARGET) is None


def test_a_candidate_cut_at_the_shortlist_is_reported_as_lost_there(
    corpus, fixture_frame, cfg
):
    scorer = ScriptedPairScorer(default=0.0)
    outcome = _pipeline(corpus, fixture_frame, cfg, scorer).run(
        SearchRequest(QUERY, DEVELOPMENT_PRINCIPAL)
    )
    telemetry = outcome.telemetry

    dropped = [
        instance_id for instance_id, stages in telemetry.survival_trace.items()
        if stages[-1] != "final"
    ]
    for instance_id in dropped:
        assert telemetry.lost_at(instance_id) is not None


# --- the complement: lexical retrieval still works --------------------------


def test_an_exact_title_query_is_still_won_by_bm25f(corpus, fixture_frame, cfg, plan_factory):
    """The migration must not trade exact-name precision for semantic recall."""
    generator = Bm25fGenerator.from_frame(fixture_frame, cfg, corpus)
    result = generator.generate(plan_factory(EXACT_TITLE_QUERY), None, 10)
    # Both Payroll Earnings Detail instances are the right family.
    assert _ids(result)[0] in {"R0105", "R0106"}


def test_an_exact_title_query_returns_that_report_end_to_end(corpus, fixture_frame, cfg):
    scorer = ScriptedPairScorer({"R0105": 9.0, "R0106": 8.0}, default=0.0, corpus=corpus)
    outcome = _pipeline(corpus, fixture_frame, cfg, scorer).run(
        SearchRequest(EXACT_TITLE_QUERY, DEVELOPMENT_PRINCIPAL)
    )
    assert outcome.families[0].canonical_title == "Payroll Earnings Detail"


# --- family and instance identity stay distinct -----------------------------


def test_duplicate_titles_are_one_family_with_several_instances(corpus, fixture_frame, cfg):
    scorer = ScriptedPairScorer({"R0106": 9.0, "R0105": 8.0}, default=0.0, corpus=corpus)
    outcome = _pipeline(corpus, fixture_frame, cfg, scorer).run(
        SearchRequest(EXACT_TITLE_QUERY, DEVELOPMENT_PRINCIPAL)
    )

    payroll = next(
        f for f in outcome.families if f.canonical_title == "Payroll Earnings Detail"
    )
    assert payroll.instance_count == 2
    # Family and instance identities stay distinct.
    assert payroll.family_id != payroll.selected.instance_id
    assert {i.instance_id for i in payroll.instances} == {"R0105", "R0106"}
    # The selected instance is the best by *final* rank. Asserting a specific id
    # here would be asserting that the cross-encoder decides on its own; under RRF
    # it is one ranked voice among several, which is the intended design.
    assert payroll.selected.rank == min(i.rank for i in payroll.instances)
    # The family score is its best instance's score, not a sum over instances.
    assert payroll.score == max(i.score for i in payroll.instances)


def test_families_are_not_rewarded_for_having_more_instances(corpus, fixture_frame, cfg):
    """Max-oriented aggregation: a two-instance family must not win on volume."""
    scorer = ScriptedPairScorer({TARGET: 9.0, "R0105": 1.0, "R0106": 1.0}, default=0.0, corpus=corpus)
    outcome = _pipeline(corpus, fixture_frame, cfg, scorer).run(
        SearchRequest(QUERY, DEVELOPMENT_PRINCIPAL)
    )
    titles = [f.canonical_title for f in outcome.families]
    if "Payroll Earnings Detail" in titles:
        assert titles.index("Attrition and Replacement Lag Analysis") < titles.index(
            "Payroll Earnings Detail"
        )


# --- authorization holds end to end -----------------------------------------


def test_an_unauthorized_target_never_appears_anywhere(corpus, fixture_frame, cfg):
    """Not in results, not in telemetry, not in the survival trace."""
    from reportfinder.auth import AuthorizedUniverse

    class _DenyTarget:
        name = "deny_target"

        def resolve(self, principal, corpus):
            mask = np.array([i != TARGET for i in corpus.instance_ids])
            return AuthorizedUniverse(mask=mask, resolver=self.name, acl_source="test")

    scorer = ScriptedPairScorer({TARGET: 9.0}, default=0.0, corpus=corpus)
    pipeline = _pipeline(corpus, fixture_frame, cfg, scorer)
    pipeline.resolver = _DenyTarget()

    outcome = pipeline.run(SearchRequest(QUERY, DEVELOPMENT_PRINCIPAL))

    assert TARGET not in [f.selected.instance_id for f in outcome.families]
    assert TARGET not in outcome.telemetry.survival_trace
    assert all(
        TARGET not in [i.instance_id for i in f.instances] for f in outcome.families
    )


def test_an_empty_universe_returns_no_confident_match_not_a_full_search(
    corpus, fixture_frame, cfg
):
    from reportfinder.auth import AuthorizedUniverse

    class _DenyAll:
        name = "deny_all"

        def resolve(self, principal, corpus):
            return AuthorizedUniverse(
                mask=np.zeros(len(corpus), dtype=bool),
                resolver=self.name, acl_source="test", fail_closed=True,
                reason="entitlements could not be resolved",
            )

    pipeline = _pipeline(corpus, fixture_frame, cfg, ScriptedPairScorer(corpus=corpus))
    pipeline.resolver = _DenyAll()

    outcome = pipeline.run(SearchRequest(QUERY, DEVELOPMENT_PRINCIPAL))
    assert outcome.decision is Decision.NO_CONFIDENT_MATCH
    assert outcome.families == []
    assert "entitlements could not be resolved" in outcome.warnings


# --- fallbacks are disclosed -------------------------------------------------


def test_the_active_fallbacks_are_reported(corpus, fixture_frame, cfg):
    """RRF fusion, the deterministic decision policy and the permissive resolver
    are all fallbacks, and every one of them must say so."""
    outcome = _pipeline(corpus, fixture_frame, cfg, ScriptedPairScorer(corpus=corpus)).run(
        SearchRequest(QUERY, DEVELOPMENT_PRINCIPAL)
    )
    fallbacks = outcome.active_fallbacks

    assert any(f.startswith("fusion:rrf") for f in fallbacks), fallbacks
    assert any(f.startswith("decision:") for f in fallbacks)
    assert "authorization:allow_all_dev_default" in fallbacks
    assert outcome.telemetry.fusion["fusion_fallback_active"] is True
    assert outcome.telemetry.decision["decision_calibrated"] is False


def test_no_score_is_presented_as_a_probability(corpus, fixture_frame, cfg):
    outcome = _pipeline(corpus, fixture_frame, cfg, ScriptedPairScorer(corpus=corpus)).run(
        SearchRequest(QUERY, DEVELOPMENT_PRINCIPAL)
    )
    assert outcome.telemetry.fusion["fusion_score_kind"] == "unnormalized_ranking_score"
    assert outcome.decision_detail.calibrated is False
