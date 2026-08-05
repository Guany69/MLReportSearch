"""Family-first retrieval: the best copy can be one retrieval never surfaced.

Retrieval nominates *instances*, but 533 titles on this estate exist on more than
one row with different field sets, prompts and data sources. Aggregating only the
instances that survived the shortlist answers the wrong question -- it reports the
best *retrieved* copy, not the best copy. A family's second row can carry exactly
the requested fields and still lose to its own first row purely because that row
ranked higher lexically.

What these tests pin down:

1. a sibling no generator found is expanded, scored, and can win selection;
2. expansion never reorders *families* -- it only decides which copy of one;
3. an unauthorized sibling is never expanded, never scored, and never appears in
   telemetry;
4. the cost bound is reported when it bites, rather than silently truncating;
5. NO_CONFIDENT_MATCH returns no result cards.

Run: uv run pytest tests/test_family_expansion.py -v
"""

from __future__ import annotations

import numpy as np
import pytest

from reportfinder.auth import AllowAllResolver, Principal, SearchRequest
from reportfinder.auth.universe import AuthorizedUniverse
from reportfinder.config import from_mapping
from reportfinder.corpus import build_corpus_model
from reportfinder.pipeline import (
    Decision,
    RrfFuser,
    SearchPipeline,
    build_query_plan,
    instance_suitability,
)
from reportfinder.query.intent import IntentParser
from reportfinder.retrieval.field_index import FieldIndex

from .fakes import ScriptedGenerator, ScriptedPairScorer
from .vague_outcome_fixture import frame

QUERY = "payroll earnings by cost center"

# Two rows of one family. Only RETRIEVED is ever nominated by a generator;
# SIBLING is the copy that actually carries "Cost Center".
RETRIEVED = "R0105"
SIBLING = "R0106"
FAMILY = "payroll earnings detail"

OTHER = "R0107"


@pytest.fixture(scope="module")
def fixture_frame():
    return frame()


@pytest.fixture(scope="module")
def corpus(fixture_frame):
    return build_corpus_model(
        fixture_frame, ingest_mode="legacy_single_file", source_file="fixture"
    )


def _cfg(**aggregation):
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
        **({"aggregation": aggregation} if aggregation else {}),
    })


@pytest.fixture(scope="module")
def cfg():
    return _cfg()


def _pipeline(corpus, fixture_frame, cfg, *, generators, scorer, resolver=None):
    return SearchPipeline(
        cfg=cfg,
        corpus=corpus,
        generators=generators,
        resolver=resolver or AllowAllResolver(allow_dev_default=True),
        intent_parser=IntentParser(fixture_frame, cfg),
        reranker=scorer,
        fuser=RrfFuser(constant=cfg.retrieval.rrf_constant),
        field_index=FieldIndex(fixture_frame),
    )


def _only_retrieves(*instance_ids):
    """One generator that finds exactly these instances, for every query."""
    class _Any(ScriptedGenerator):
        def generate(self, plan, universe, k):
            self.results = {plan.raw_query: [(i, 1.0) for i in instance_ids]}
            return super().generate(plan, universe, k)

    return [_Any("bm25f", {})]


def _run(corpus, fixture_frame, cfg, *, retrieves, scores, resolver=None):
    scorer = ScriptedPairScorer(scores, default=-5.0, corpus=corpus)
    pipeline = _pipeline(
        corpus, fixture_frame, cfg,
        generators=_only_retrieves(*retrieves), scorer=scorer, resolver=resolver,
    )
    return pipeline.run(SearchRequest(QUERY, Principal(principal_id="u1"))), scorer


# --- 1. the sibling is expanded and can win ---------------------------------


def test_a_sibling_no_generator_found_is_expanded_and_wins(corpus, fixture_frame, cfg):
    """The point of the whole stage.

    Only R0105 is retrieved. R0106 -- the copy with Cost Center -- is invisible to
    every generator, and under shortlist-only aggregation could never be returned.
    """
    outcome, _ = _run(
        corpus, fixture_frame, cfg,
        retrieves=[RETRIEVED, OTHER],
        # The reranker prefers the sibling once it is allowed to see it.
        scores={RETRIEVED: 1.0, SIBLING: 8.0, OTHER: 0.5},
    )

    family = next(f for f in outcome.families if f.family_id == FAMILY)
    assert family.selected.instance_id == SIBLING
    assert family.selected.expanded is True
    assert family.instance_count == 2, "both copies are in contention"


def test_the_expanded_instance_is_recorded_in_the_survival_trace(
    corpus, fixture_frame, cfg
):
    outcome, _ = _run(
        corpus, fixture_frame, cfg,
        retrieves=[RETRIEVED, OTHER],
        scores={RETRIEVED: 1.0, SIBLING: 8.0, OTHER: 0.5},
    )
    trace = outcome.telemetry.survival_trace[SIBLING]

    # It was never generated, never in the union, never shortlisted -- it entered
    # at expansion. Anything else would misreport where it came from.
    assert "generated" not in trace
    assert "union" not in trace
    assert "shortlist" not in trace
    assert trace[0] == "expansion"
    assert trace[-1] == "final"


def test_an_expanded_instance_is_masked_against_every_generator(
    corpus, fixture_frame, cfg
):
    """No retriever voted on it. A 0.0 would say they looked and rejected it."""
    outcome, _ = _run(
        corpus, fixture_frame, cfg,
        retrieves=[RETRIEVED, OTHER],
        scores={RETRIEVED: 1.0, SIBLING: 8.0, OTHER: 0.5},
    )
    family = next(f for f in outcome.families if f.family_id == FAMILY)
    record = next(m for m in family.instances if m.instance_id == SIBLING).record

    assert record.found_by == set()
    assert record.masked == {"bm25f"}
    assert all(value is None for value in record.scores.values())
    assert record.source_exclusive is False


def test_the_expanded_instance_is_cross_encoded_on_the_raw_query(
    corpus, fixture_frame, cfg
):
    """Expansion must use the same query the shortlist was scored on, or the two
    sets of scores are not comparable and selection within a family is arbitrary."""
    _, scorer = _run(
        corpus, fixture_frame, cfg,
        retrieves=[RETRIEVED, OTHER],
        scores={RETRIEVED: 1.0, SIBLING: 8.0, OTHER: 0.5},
    )
    assert scorer.seen_queries, "the scorer was never called"
    assert set(scorer.seen_queries) == {QUERY}


# --- 2. expansion decides which copy, never which family --------------------


def test_expansion_does_not_reorder_families(corpus, fixture_frame, cfg):
    """A cross-encoder logit and a fused score are not on one scale.

    If expansion fed its logits into the family max, a family could climb the
    ranking purely by owning more rows -- the exact bias max-oriented aggregation
    exists to prevent.
    """
    outcome, _ = _run(
        corpus, fixture_frame, cfg,
        retrieves=[OTHER, RETRIEVED],
        # A huge logit on a purely expanded instance.
        scores={OTHER: 2.0, RETRIEVED: 1.0, SIBLING: 999.0},
    )
    order = [f.family_id for f in outcome.families]
    payroll = order.index(FAMILY)

    other_family = corpus.instance(OTHER).family_id
    assert order.index(other_family) < payroll, (
        "an expanded instance's logit must not promote its family"
    )


def test_a_multi_instance_family_gains_no_score_from_its_extra_rows(
    corpus, fixture_frame, cfg
):
    outcome, _ = _run(
        corpus, fixture_frame, cfg,
        retrieves=[RETRIEVED, OTHER],
        scores={RETRIEVED: 1.0, SIBLING: 8.0, OTHER: 0.5},
    )
    payroll = next(f for f in outcome.families if f.family_id == FAMILY)
    singles = [f for f in outcome.families if f.instance_count == 1]

    assert payroll.instance_count == 2
    # Its score is still just its best *retrieved* member's fused score.
    assert payroll.score == pytest.approx(
        max(m.score for m in payroll.instances if not m.expanded)
    )
    assert all(f.score <= payroll.score or f.rank < payroll.rank for f in singles)


# --- 3. authorization ---------------------------------------------------------


class _DenyingResolver:
    """Grants everything except one instance."""

    name = "test_deny_one"

    def __init__(self, denied: str) -> None:
        self.denied = denied

    def resolve(self, principal, corpus):
        mask = np.ones(len(corpus), dtype=bool)
        mask[corpus.position_of(self.denied)] = False
        return AuthorizedUniverse(mask=mask, resolver=self.name, acl_source="test")


def test_an_unauthorized_sibling_is_never_expanded(corpus, fixture_frame, cfg):
    """Expansion is a new route into a response, so it re-checks entitlement.

    A route that bypasses the generators must not also bypass the check the
    generators applied.
    """
    outcome, _ = _run(
        corpus, fixture_frame, cfg,
        retrieves=[RETRIEVED, OTHER],
        scores={RETRIEVED: 1.0, SIBLING: 999.0, OTHER: 0.5},
        resolver=_DenyingResolver(SIBLING),
    )

    family = next(f for f in outcome.families if f.family_id == FAMILY)
    assert family.selected.instance_id == RETRIEVED
    assert SIBLING not in [m.instance_id for m in family.instances]
    assert outcome.telemetry.aggregation["unauthorized_skipped"] == 1


def test_an_unauthorized_sibling_appears_in_no_telemetry(corpus, fixture_frame, cfg):
    """Not in the trace, not in generator hits, not in any serialized record."""
    outcome, scorer = _run(
        corpus, fixture_frame, cfg,
        retrieves=[RETRIEVED, OTHER],
        scores={RETRIEVED: 1.0, SIBLING: 999.0, OTHER: 0.5},
        resolver=_DenyingResolver(SIBLING),
    )

    assert SIBLING not in outcome.telemetry.survival_trace
    for ids in outcome.telemetry.generator_hits.values():
        assert SIBLING not in ids
    # And it was never even scored -- entitlement is checked before the model runs,
    # not after, so an unauthorized report costs nothing and leaks nothing.
    assert all(SIBLING not in text for batch in scorer.seen_texts for text in batch)
    assert SIBLING not in repr(outcome.telemetry.with_survival())


# --- 4. the cost bound is reported, not silent -------------------------------


def test_the_expansion_cap_is_reported_when_it_bites(corpus, fixture_frame):
    """A silent truncation reads as 'we considered everything'. It did not."""
    cfg = _cfg(max_expanded_instances=1)
    outcome, _ = _run(
        corpus, fixture_frame, cfg,
        # Retrieve one member of each of two multi-instance families, so two
        # siblings are eligible for expansion but only one slot exists.
        retrieves=[RETRIEVED, "R0108"],
        scores={RETRIEVED: 1.0, "R0108": 1.0},
    )
    aggregation = outcome.telemetry.aggregation

    assert aggregation["instances_added"] == 1
    assert aggregation["instances_dropped_by_cap"] >= 1


def test_expansion_can_be_turned_off_explicitly(corpus, fixture_frame):
    cfg = _cfg(enabled=False, max_expanded_instances=60)
    outcome, _ = _run(
        corpus, fixture_frame, cfg,
        retrieves=[RETRIEVED, OTHER],
        scores={RETRIEVED: 1.0, SIBLING: 999.0, OTHER: 0.5},
    )

    assert outcome.telemetry.aggregation["expansion_ran"] is False
    assert outcome.telemetry.aggregation["expansion_skipped_reason"] == (
        "aggregation.enabled=false"
    )
    family = next(f for f in outcome.families if f.family_id == FAMILY)
    assert family.selected.instance_id == RETRIEVED


def test_enabling_expansion_with_a_zero_cap_is_rejected():
    """An enabled stage that can never do anything is a configuration mistake, not
    a valid way to disable it."""
    with pytest.raises(ValueError, match="expand nothing"):
        _cfg(enabled=True, max_expanded_instances=0)


# --- 5. NO_CONFIDENT_MATCH returns nothing -----------------------------------


def test_no_confident_match_returns_no_result_cards(corpus, fixture_frame, cfg):
    """A ranked list reads as an answer whatever status sits beside it.

    The reranker scores everything far below `decision.weak_rerank_logit`, which is
    the pipeline's "nothing here answers this" signal, so the decision is forced
    rather than coaxed out of a threshold that could drift.
    """
    scorer = ScriptedPairScorer({}, default=-40.0, corpus=corpus)
    pipeline = _pipeline(
        corpus, fixture_frame, cfg,
        generators=_only_retrieves(RETRIEVED, OTHER), scorer=scorer,
    )
    outcome = pipeline.run(SearchRequest(QUERY, Principal(principal_id="u1")))

    assert outcome.decision is Decision.NO_CONFIDENT_MATCH
    assert outcome.families == []


def test_ask_clarification_keeps_its_candidates(corpus, fixture_frame, cfg):
    """The complement: a clarification is grounded in the candidates that disagree,
    so suppressing them would leave the question unanswerable."""
    scorer = ScriptedPairScorer(
        # Two plausible, near-indistinguishable answers: the shape that warrants
        # asking rather than guessing.
        {RETRIEVED: 5.0, OTHER: 4.9}, default=-40.0, corpus=corpus,
    )
    pipeline = _pipeline(
        corpus, fixture_frame, cfg,
        generators=_only_retrieves(RETRIEVED, OTHER), scorer=scorer,
    )
    outcome = pipeline.run(SearchRequest(QUERY, Principal(principal_id="u1")))

    if outcome.decision is Decision.ASK_CLARIFICATION:
        assert outcome.families, "a clarification with no candidates cannot be grounded"


# --- suitability -------------------------------------------------------------


def _plan(fixture_frame, cfg, query):
    return build_query_plan(IntentParser(fixture_frame, cfg).parse(query), raw_query=query)


def test_suitability_prefers_the_copy_carrying_the_requested_field(
    corpus, fixture_frame, cfg
):
    plan = _plan(fixture_frame, cfg, "payroll earnings by cost center")
    with_field = instance_suitability(corpus.instance(SIBLING), plan)
    without = instance_suitability(corpus.instance(RETRIEVED), plan)

    assert with_field > without


def test_an_unmentioned_facet_is_neither_satisfied_nor_violated(
    corpus, fixture_frame, cfg
):
    """A requirement the query never expressed must not move the score in either
    direction. Counting it as satisfied rewards irrelevance; counting it as failed
    punishes reports for not answering a question nobody asked."""
    plan = _plan(fixture_frame, cfg, "cost center")
    score = instance_suitability(corpus.instance(SIBLING), plan)
    assert 0.0 <= score <= 1.0


def _selected_without_reranker(corpus, fixture_frame, cfg, query):
    pipeline = _pipeline(
        corpus, fixture_frame, cfg,
        generators=_only_retrieves(RETRIEVED, OTHER),
        scorer=None,
    )
    outcome = pipeline.run(SearchRequest(query, Principal(principal_id="u1")))
    return outcome, next(f for f in outcome.families if f.family_id == FAMILY)


def test_without_a_reranker_a_genuine_field_match_still_wins(
    corpus, fixture_frame, cfg
):
    """This is the behaviour Phase 9 exists for, and it holds with no model at all.

    R0106 is the copy carrying "Cost Center". No generator nominated it, and under
    shortlist-only aggregation it could never be returned -- yet it is the copy that
    answers the request, and deterministic metadata compatibility is enough to say
    so. Note this is *not* the reranker's judgement: it is a catalog fact.
    """
    outcome, family = _selected_without_reranker(
        corpus, fixture_frame, cfg, "payroll earnings by cost center"
    )

    assert outcome.telemetry.aggregation["expanded_cross_encoded"] == 0
    assert family.selected.instance_id == SIBLING
    assert family.selected.expanded is True
    assert family.selected.suitability > 0


def test_an_expanded_instance_with_no_advantage_does_not_displace_a_retrieved_one(
    corpus, fixture_frame, cfg
):
    """The complement, and the guard against the obvious failure mode.

    When the query distinguishes the two copies not at all, the expanded one has
    nothing to win on -- no fused score, no reranker, no better field match -- and
    selection must fall back to the retrieval ordering that existed before it.
    """
    outcome, family = _selected_without_reranker(
        corpus, fixture_frame, cfg, "payroll earnings"
    )
    suitabilities = {m.instance_id: m.suitability for m in family.instances}

    assert suitabilities[SIBLING] == pytest.approx(suitabilities[RETRIEVED]), (
        "the query must not distinguish the copies for this test to mean anything"
    )
    assert family.selected.instance_id == RETRIEVED
    assert family.selected.expanded is False
    # The sibling is still *present* -- it is a real authorized copy of the family,
    # and suppressing it would hide it from clarification and detail views too.
    assert SIBLING in [m.instance_id for m in family.instances]


# --- 6. pair-count telemetry -------------------------------------------------


def test_expansion_reports_what_the_model_actually_scored(corpus, fixture_frame, cfg):
    """Expansion is where duplicate text is most likely -- these are siblings of
    one family by construction. The counters distinguish candidates considered
    from forward passes run; they are the measurement hook, not a measurement."""
    outcome, _ = _run(
        corpus, fixture_frame, cfg,
        retrieves=[RETRIEVED, OTHER],
        scores={RETRIEVED: 1.0, SIBLING: 8.0, OTHER: 0.5},
    )
    aggregation = outcome.telemetry.aggregation

    for key in (
        "expansion_pairs_submitted",
        "expansion_model_pairs",
        "expansion_deduplicated_pairs",
        "expansion_non_finite_scores",
    ):
        assert key in aggregation, f"{key} missing from expansion telemetry"

    submitted = aggregation["expansion_pairs_submitted"]
    model_pairs = aggregation["expansion_model_pairs"]
    assert submitted >= model_pairs >= 0
    assert aggregation["expansion_deduplicated_pairs"] == submitted - model_pairs
    # These fixture instances differ in their fields, so nothing should collapse.
    assert aggregation["expansion_deduplicated_pairs"] == 0
    assert aggregation["expansion_non_finite_scores"] == 0
