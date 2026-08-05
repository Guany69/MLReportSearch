"""Generators nominate independently.

This is where the system invariant is actually pinned: every generator searches the
whole authorized universe on its own terms, keeps its own scale, and can surface a
report that every other generator missed. If any of these fail, some retriever has
quietly become a gate again.

Run: uv run pytest tests/test_generators.py -v
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from reportfinder.auth import AuthorizedUniverse
from reportfinder.config import DEFAULT
from reportfinder.corpus import ViewType, build_corpus_model
from reportfinder.generators import (
    Bm25fGenerator,
    DenseViewGenerator,
    DisabledLateInteractionGenerator,
    LateInteractionGenerator,
    PrototypeGenerator,
    SpladeGenerator,
)
from reportfinder.index import (
    DenseViewIndex,
    FamilyPrototypeIndex,
    LateInteractionIndex,
    SpladeIndex,
    seed_prototypes_from_catalog,
)
from reportfinder.pipeline.prepare import build_query_plan
from reportfinder.query.intent import IntentParser

from .fakes import FakeEncoder, FakeSparseEncoder, FakeTokenEncoder

# R0004 is the target: its title shares no token with the vague query, but its
# fields do the semantic work. R0006 is the lexical decoy.
ROWS = [
    {"title": "Attrition and Replacement Lag Analysis",
     "fields": ["Termination Reason", "Time to Fill", "Headcount"],
     "description": "Tracks why workers leave and how long replacement takes.",
     "category": "Worker Data"},
    {"title": "Headcount by Supervisory Organization",
     "fields": ["Headcount", "Supervisory Organization"],
     "description": "Organization sizes.", "category": "Worker Data"},
    {"title": "Backfill Request Log",
     "fields": ["Requisition Status"],
     "description": "Open requisitions.", "category": "Recruiting"},
    {"title": "Payroll Earnings Detail",
     "fields": ["Net Pay", "Pay Period"],
     "description": "Earnings by period.", "category": "Payroll"},
]


def _frame():
    return pd.DataFrame([
        {
            "report_key": f"R{4 + i:04d}", "title_key": r["title"].casefold(),
            "source_row": 4 + i, "title": r["title"], "description": r["description"],
            "category": r["category"], "data_source": "All Workers",
            "report_type": "Advanced", "prompts": ["Effective Date"],
            "fields": r["fields"], "tags": "", "area_where_used": "",
            "worklet": "Standard", "chart_type": "Bar", "landing_page": "People",
            "worklet_landing_pages": "", "shared": "Yes", "field_meta": [],
        }
        for i, r in enumerate(ROWS)
    ])


@pytest.fixture(scope="module")
def corpus():
    return build_corpus_model(
        _frame(), ingest_mode="legacy_single_file", source_file="f"
    )


@pytest.fixture(scope="module")
def plan_factory():
    parser = IntentParser(_frame(), DEFAULT)
    return lambda q: build_query_plan(parser.parse(q), raw_query=q)


def _dense_gen(corpus, view):
    encoder = FakeEncoder()
    index = DenseViewIndex.build(
        view_type=view, instance_ids=corpus.instance_ids,
        texts=corpus.view_texts(view), hashes=corpus.view_hashes(view),
        encoder=encoder,
    )
    return DenseViewGenerator(index, encoder, corpus)


def _splade_gen(corpus):
    encoder = FakeSparseEncoder()
    texts = [
        " ".join(corpus.views[i][v].text for v in ViewType)
        for i in corpus.instance_ids
    ]
    index = SpladeIndex.build(
        instance_ids=corpus.instance_ids, texts=texts,
        hashes=corpus.view_hashes(ViewType.IDENTITY), encoder=encoder,
    )
    return SpladeGenerator(index, encoder, corpus)


def _prototype_gen(corpus):
    encoder = FakeEncoder()
    index = FamilyPrototypeIndex.build(seed_prototypes_from_catalog(corpus), encoder)
    return PrototypeGenerator(index, encoder, corpus)


def _bm25_gen(corpus):
    return Bm25fGenerator.from_frame(_frame(), DEFAULT, corpus)


def _all_generators(corpus):
    return [
        _bm25_gen(corpus), _splade_gen(corpus), _prototype_gen(corpus),
        *(_dense_gen(corpus, v) for v in ViewType),
    ]


def _ids(result):
    return [instance_id for instance_id, _, _ in result.hits]


# --- the contract ------------------------------------------------------------


def test_every_generator_returns_the_common_contract(corpus, plan_factory):
    plan = plan_factory("headcount by supervisory organization")
    for generator in _all_generators(corpus):
        result = generator.generate(plan, None, 10)
        assert result.generator == generator.name
        assert result.ok
        for instance_id, rank, score in result.hits:
            assert instance_id in corpus.instance_ids
            assert isinstance(rank, int) and rank >= 1
            assert isinstance(score, float)
        # Ranks are dense and 1-based within the generator.
        assert [r for _, r, _ in result.hits] == list(range(1, len(result.hits) + 1))


def test_hits_carry_family_variant_and_view_provenance(corpus, plan_factory):
    plan = plan_factory("headcount")
    for generator in _all_generators(corpus):
        hits = generator.generate(plan, None, 10).to_hits(
            lambda i: corpus.instance(i).family_id
        )
        for hit in hits:
            assert hit.family_id == corpus.instance(hit.report_instance_id).family_id
            assert hit.query_variant
            assert hit.generator == generator.name
            assert hit.view_type == generator.view_type


def test_each_generator_keeps_its_own_scale(corpus, plan_factory):
    """Scores are not normalized to each other; fusion is over ranks."""
    plan = plan_factory("headcount by supervisory organization")
    bm25 = _bm25_gen(corpus).generate(plan, None, 10)
    dense = _dense_gen(corpus, ViewType.IDENTITY).generate(plan, None, 10)
    # A cosine is bounded by 1; a BM25F score is not bounded at all.
    assert all(abs(s) <= 1.0 + 1e-6 for _, _, s in dense.hits)
    assert max(s for _, _, s in bm25.hits) > 1.0


# --- independent nomination --------------------------------------------------


def test_bm25f_misses_the_vague_query_but_semantic_generators_do_not(corpus, plan_factory):
    """The failure this migration exists to fix.

    The query shares no token with the target's title. BM25F cannot see it; the
    schema and purpose views can, because the target's *fields* and *description*
    carry the meaning.
    """
    plan = plan_factory("why are we losing people faster than we can backfill")

    bm25 = _bm25_gen(corpus).generate(plan, None, 10)
    schema = _dense_gen(corpus, ViewType.SCHEMA).generate(plan, None, 10)
    purpose = _dense_gen(corpus, ViewType.PURPOSE).generate(plan, None, 10)

    assert "R0004" not in _ids(bm25)[:1], "BM25F should not rank the target first"
    assert "R0004" in _ids(schema), "schema view must nominate the target"
    assert "R0004" in _ids(purpose), "purpose view must nominate the target"


def test_exact_title_query_is_still_won_by_bm25f(corpus, plan_factory):
    """The complement: lexical retrieval must not be weakened by the migration."""
    plan = plan_factory("Payroll Earnings Detail")
    assert _ids(_bm25_gen(corpus).generate(plan, None, 10))[0] == "R0007"


def test_the_union_reaches_further_than_any_single_generator(corpus, plan_factory):
    """The recall claim, stated directly.

    Phrased as "union beats the best single slate" rather than "every generator
    contributes something unique": on a four-document fixture with k=2 the latter
    is a statement about the fixture, while this holds at any corpus size and is
    what actually determines whether the right report can be reached.
    """
    plan = plan_factory("why are we losing people faster than we can backfill")
    slates = {g.name: set(_ids(g.generate(plan, None, 2))) for g in _all_generators(corpus)}
    union = set().union(*slates.values())

    assert len(union) > max(len(slate) for slate in slates.values()), (
        "no single generator reaches as far as the union: "
        f"{ {k: sorted(v) for k, v in slates.items()} }"
    )


def test_a_generator_searches_the_whole_universe_not_another_slate(corpus, plan_factory):
    """Nothing is required to appear in BM25F to be retrievable."""
    plan = plan_factory("why are we losing people faster than we can backfill")
    bm25_ids = set(_ids(_bm25_gen(corpus).generate(plan, None, 2)))
    schema_ids = set(_ids(_dense_gen(corpus, ViewType.SCHEMA).generate(plan, None, 4)))
    assert schema_ids - bm25_ids, "dense retrieval must reach beyond the lexical slate"


# --- query variants add, never remove ---------------------------------------


def test_the_raw_query_is_always_one_of_the_searched_variants(corpus, plan_factory):
    plan = plan_factory("why are we losing people")
    for generator in _all_generators(corpus):
        result = generator.generate(plan, None, 10)
        if result.hits:
            assert any(
                result.variant_for(i) in {v.key for v in plan.variants}
                for i, _, _ in result.hits
            )


def test_variant_provenance_records_which_lens_found_each_hit(corpus, plan_factory):
    plan = plan_factory("why are we losing people faster than we can backfill")
    result = _dense_gen(corpus, ViewType.SCHEMA).generate(plan, None, 10)
    assert result.variant_for("R0004") in {v.key for v in plan.variants}


def test_more_variants_never_shrink_a_generators_slate(corpus, plan_factory):
    """Adding a lens must be monotonic: it can only add candidates."""
    plan = plan_factory("why are we losing people faster than we can backfill")
    generator = _dense_gen(corpus, ViewType.SCHEMA)

    raw_only = generator.generate(
        build_query_plan(plan.intent, raw_query=plan.raw_query,
                         enable_alternate=False), None, 10
    )
    full = generator.generate(plan, None, 10)
    assert set(_ids(raw_only)) <= set(_ids(full))


# --- authorization -----------------------------------------------------------


@pytest.mark.parametrize("factory", [
    _bm25_gen, _splade_gen, _prototype_gen,
    lambda c: _dense_gen(c, ViewType.IDENTITY),
    lambda c: _dense_gen(c, ViewType.SCHEMA),
])
def test_no_generator_returns_an_unauthorized_instance(corpus, plan_factory, factory):
    plan = plan_factory("headcount worker termination payroll")
    mask = np.array([False, True, False, True])
    universe = AuthorizedUniverse(mask=mask, resolver="t", acl_source="t")

    result = factory(corpus).generate(plan, universe, 10)
    assert set(_ids(result)) <= {"R0005", "R0007"}


@pytest.mark.parametrize("factory", [
    _bm25_gen, _splade_gen, _prototype_gen,
    lambda c: _dense_gen(c, ViewType.IDENTITY),
])
def test_an_empty_universe_yields_no_candidates(corpus, plan_factory, factory):
    plan = plan_factory("headcount")
    universe = AuthorizedUniverse(
        mask=np.zeros(len(corpus), dtype=bool), resolver="t", acl_source="t"
    )
    assert factory(corpus).generate(plan, universe, 10).hits == ()


def test_prototype_generator_does_not_leak_unauthorized_family_members(corpus, plan_factory):
    """A visible family must not carry its hidden instances back in."""
    plan = plan_factory("show attrition and replacement lag analysis")
    mask = np.array([False, True, True, True])
    universe = AuthorizedUniverse(mask=mask, resolver="t", acl_source="t")
    assert "R0004" not in _ids(_prototype_gen(corpus).generate(plan, universe, 10))


# --- evidence ----------------------------------------------------------------


def test_bm25f_records_which_terms_matched_where(corpus, plan_factory):
    plan = plan_factory("headcount supervisory organization")
    result = _bm25_gen(corpus).generate(plan, None, 10)
    evidence = result.match_evidence["R0005"]
    assert "headcount" in evidence["matched_title_terms"]
    assert evidence["matched_field_terms"]


def test_dense_evidence_names_the_view_that_matched(corpus, plan_factory):
    plan = plan_factory("why are we losing people faster than we can backfill")
    result = _dense_gen(corpus, ViewType.SCHEMA).generate(plan, None, 10)
    assert result.match_evidence["R0004"]["view"] == "schema"
    assert "cosine" in result.match_evidence["R0004"]


def test_prototype_evidence_declares_itself_non_authoritative(corpus, plan_factory):
    """Generated language must never be mistakable for catalog evidence."""
    plan = plan_factory("show headcount by supervisory organization")
    result = _prototype_gen(corpus).generate(plan, None, 10)
    for evidence in result.match_evidence.values():
        assert evidence["prototype_is_authoritative_catalog_text"] is False


def test_splade_evidence_labels_its_score_as_sparse(corpus, plan_factory):
    plan = plan_factory("headcount")
    result = _splade_gen(corpus).generate(plan, None, 10)
    for evidence in result.match_evidence.values():
        assert "sparse_score" in evidence
        assert "cosine" not in evidence


# --- late interaction --------------------------------------------------------


def test_late_interaction_generates_when_an_index_exists(corpus, plan_factory):
    encoder = FakeTokenEncoder()
    index = LateInteractionIndex.build(
        instance_ids=corpus.instance_ids,
        texts=corpus.view_texts(ViewType.IDENTITY), encoder=encoder,
    )
    result = LateInteractionGenerator(index, encoder, corpus).generate(
        plan_factory("headcount supervisory organization"), None, 10
    )
    assert result.ok and result.hits


def test_disabled_late_interaction_reports_that_it_did_not_run(corpus, plan_factory):
    """An empty result with no error would look like "ran, found nothing"."""
    generator = DisabledLateInteractionGenerator("no approved artifact")
    result = generator.generate(plan_factory("headcount"), None, 10)
    assert result.hits == ()
    assert not result.ok
    assert "not_run" in result.error
    status = generator.status()
    assert status.enabled is False and status.built is False
    assert status.reason == "no approved artifact"
