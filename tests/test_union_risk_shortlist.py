"""Union, recall risk, and shortlist selection.

These three stages are where the old architecture lost the right answer, so the
assertions here are about *preservation* rather than ranking quality:

* a missing generator is a mask, not a zero;
* a candidate one generator found survives a shortlist full of better-fused decoys;
* a vague query widens the search instead of narrowing it.

Run: uv run pytest tests/test_union_risk_shortlist.py -v
"""

from __future__ import annotations

import pytest

from reportfinder.config import DEFAULT, from_mapping
from reportfinder.generators.base import GeneratorResult
from reportfinder.pipeline.prepare import FacetState, QueryPlan, QueryVariant
from reportfinder.pipeline.risk import assess_recall_risk
from reportfinder.pipeline.shortlist import select_shortlist
from reportfinder.pipeline.union import build_union


def _result(generator, hits, *, view_type=None, variant="Q0", error=None,
            variant_by_instance=None):
    return GeneratorResult(
        generator=generator, query_variant=variant, view_type=view_type,
        hits=tuple((i, r, s) for r, (i, s) in enumerate(hits, start=1)),
        error=error, variant_by_instance=variant_by_instance or {},
    )


def _family_of(instance_id: str) -> str:
    return f"fam-{instance_id}"


def _union(*results, constant=60):
    return build_union(results, family_of=_family_of, rrf_constant=constant)


def _plan(query="why are we losing people", variants=("Q0",), facets=None):
    return QueryPlan(
        raw_query=query,
        variants=tuple(QueryVariant(k, query, "raw", "user", 1.0) for k in variants),
        facets=facets or {},
    )


def _cfg_depth(depth, **quota_overrides):
    """A config with a small shortlist and quotas scaled to fit.

    The default quotas total 70 exclusive slots, which `ShortlistConfig` rightly
    refuses to fit into a 20-slot shortlist. Tests that want a small depth must
    scale the floors too, exactly as a real deployment would.
    """
    quotas = {
        "fused": max(1, depth // 2), "bm25_exclusive": 2, "splade_exclusive": 2,
        "schema_exclusive": 2, "purpose_exclusive": 1, "prototype_exclusive": 1,
        "alternate_query_exclusive": 2,
    }
    quotas.update(quota_overrides)
    # Both depths are pinned: several of these fixtures are vague enough to land in
    # the HIGH band, which would otherwise correctly use the 200-slot depth and
    # make the test about risk banding rather than about quota mechanics.
    return from_mapping({"shortlist": {
        "standard_rerank_depth": depth,
        "high_risk_rerank_depth": depth,
        "quotas": quotas,
    }})


# --- masks vs zeros ----------------------------------------------------------


def test_a_generator_that_missed_records_a_mask_not_a_zero():
    """A 0.0 would read as "this generator scored it lowest"; None reads as no vote."""
    union = _union(
        _result("bm25f", [("R1", 5.0)]),
        _result("dense_schema", [("R2", 0.8)]),
    )
    r1 = union.records["R1"]
    assert r1.scores["bm25f"] == 5.0
    assert r1.scores["dense_schema"] is None
    assert r1.masked == {"dense_schema"}
    assert "dense_schema" not in r1.ranks


def test_a_generator_that_failed_is_not_recorded_as_a_mask():
    """Masking against a generator that never ran would misreport coverage."""
    union = _union(
        _result("bm25f", [("R1", 5.0)]),
        _result("splade", [], error="not_run: backend unavailable"),
    )
    assert union.generators_run == ("bm25f",)
    assert "splade" in union.generators_failed
    assert union.records["R1"].masked == set()


def test_optional_generator_failure_does_not_lose_other_candidates():
    union = _union(
        _result("bm25f", [("R1", 5.0)]),
        _result("splade", [], error="not_run: backend unavailable"),
        _result("dense_schema", [("R2", 0.8)]),
    )
    assert set(union.records) == {"R1", "R2"}


# --- preservation ------------------------------------------------------------


def test_no_candidate_is_dropped_because_another_generator_missed_it():
    union = _union(
        _result("bm25f", [("R1", 9.0), ("R2", 8.0)]),
        _result("dense_schema", [("R3", 0.9)]),
        _result("prototype", [("R4", 0.7)]),
    )
    assert set(union.records) == {"R1", "R2", "R3", "R4"}


def test_every_generator_rank_score_and_variant_survives():
    union = _union(
        _result("bm25f", [("R1", 9.0)]),
        _result("dense_schema", [("R1", 0.42)], view_type="schema", variant="Q4",
                variant_by_instance={"R1": "Q4"}),
    )
    record = union.records["R1"]
    assert record.ranks == {"bm25f": 1, "dense_schema": 1}
    assert record.scores["bm25f"] == 9.0
    assert record.scores["dense_schema"] == pytest.approx(0.42)
    assert record.variants["dense_schema"] == "Q4"
    assert record.views["dense_schema"] == "schema"
    assert record.query_variants == {"Q0", "Q4"}


def test_candidates_are_deduplicated_by_instance_id():
    union = _union(
        _result("bm25f", [("R1", 9.0)]),
        _result("dense_identity", [("R1", 0.8)]),
        _result("dense_schema", [("R1", 0.7)]),
    )
    assert len(union) == 1
    assert union.records["R1"].generator_count == 3


def test_family_identity_survives_the_union():
    union = _union(_result("bm25f", [("R1", 9.0)]))
    assert union.records["R1"].family_id == "fam-R1"


def test_source_exclusive_flags_are_computed():
    union = _union(
        _result("bm25f", [("R1", 9.0), ("R2", 8.0)]),
        _result("dense_schema", [("R2", 0.9), ("R3", 0.8)]),
    )
    assert union.records["R1"].source_exclusive and union.records["R1"].exclusive_to == "bm25f"
    assert not union.records["R2"].source_exclusive
    assert union.records["R3"].exclusive_to == "dense_schema"


# --- fusion ------------------------------------------------------------------


def test_fusion_uses_ranks_not_raw_scores():
    """BM25F scores and cosines have no common scale; only ranks are comparable."""
    big = _union(_result("bm25f", [("R1", 900.0)]), _result("dense_schema", [("R2", 0.9)]))
    small = _union(_result("bm25f", [("R1", 0.9)]), _result("dense_schema", [("R2", 900.0)]))
    assert big.ordering == small.ordering


def test_agreement_between_generators_raises_the_fused_score():
    union = _union(
        _result("bm25f", [("R1", 1.0), ("R2", 0.9)]),
        _result("dense_schema", [("R1", 1.0)]),
    )
    assert union.records["R1"].fused_score > union.records["R2"].fused_score
    assert union.ordering[0] == "R1"


def test_fused_ordering_is_deterministic_under_ties():
    first = _union(_result("a", [("R2", 1.0)]), _result("b", [("R1", 1.0)]))
    second = _union(_result("b", [("R1", 1.0)]), _result("a", [("R2", 1.0)]))
    assert first.ordering == second.ordering


def test_union_telemetry_counts_sources():
    union = _union(
        _result("bm25f", [("R1", 9.0), ("R2", 8.0)]),
        _result("dense_schema", [("R2", 0.9), ("R3", 0.8)]),
    )
    telemetry = union.telemetry()
    assert telemetry["union_size"] == 3
    assert telemetry["candidates_by_generator"] == {"bm25f": 2, "dense_schema": 2}
    assert telemetry["source_exclusive_by_generator"] == {"bm25f": 1, "dense_schema": 1}


# --- recall risk -------------------------------------------------------------


def _risk(union, plan, cfg=None, **kwargs):
    return assess_recall_risk(
        plan, union, cfg or DEFAULT, authorized_count=kwargs.pop("authorized", 4000),
        **kwargs,
    )


def test_a_vague_query_widens_the_search_rather_than_narrowing_it():
    """The central rule: uncertainty at retrieval time means look wider."""
    union = _union(
        _result("bm25f", [("R1", 1.0)]),
        _result("dense_schema", [("R2", 0.9)]),
        _result("prototype", [("R3", 0.8)]),
    )
    risk = _risk(union, _plan("why", facets={"measure": FacetState.NONE}))
    assert risk.risk_band == "HIGH"
    assert risk.rerank_depth == DEFAULT.shortlist.high_risk_rerank_depth
    assert risk.rerank_depth > DEFAULT.shortlist.standard_rerank_depth


def test_a_specific_well_agreed_query_stays_low_risk():
    """Every generator agrees on one winner, with a tail only some of them found.

    That shape -- not "all generators return the same ordering" -- is what a
    confident retrieval looks like under RRF. When every generator ranks the same
    documents 1, 2, 3, adjacent RRF scores differ by only 1/(k+2), so there is
    genuinely nothing separating them and MEDIUM is the correct reading.
    """
    union = _union(
        _result("bm25f", [("R1", 9.0), ("R2", 4.0), ("R3", 1.0)]),
        _result("dense_identity", [("R1", 0.9), ("R2", 0.5), ("R4", 0.3)]),
        _result("dense_schema", [("R1", 0.9), ("R3", 0.5), ("R5", 0.3)]),
        _result("splade", [("R1", 8.0), ("R2", 3.0), ("R3", 1.0)]),
    )
    plan = _plan(
        "headcount by supervisory organization for all workers",
        facets={"measure": ["Headcount"], "dimension": ["Organization"],
                "data_source": ["All Workers"], "population": ["Worker Data"]},
    )
    risk = _risk(union, plan)
    assert risk.risk_band == "LOW", risk.reasons
    assert risk.rerank_depth == DEFAULT.shortlist.standard_rerank_depth


def test_short_query_is_flagged():
    union = _union(_result("bm25f", [("R1", 1.0)]))
    assert "very_short_query" in _risk(union, _plan("pay")).reasons


def test_high_source_exclusivity_is_flagged():
    union = _union(
        _result("bm25f", [("R1", 1.0)]),
        _result("dense_schema", [("R2", 1.0)]),
        _result("prototype", [("R3", 1.0)]),
    )
    assert "high_source_exclusivity" in _risk(union, _plan()).reasons


def test_conflicting_facets_are_flagged():
    union = _union(_result("bm25f", [("R1", 1.0)]))
    plan = _plan(facets={"time_orientation": FacetState.MULTIPLE})
    assert "conflicting_facets" in _risk(union, plan).reasons


def test_a_generator_failure_widens_the_search():
    """A missing source is a hole in coverage, so compensate rather than shrink."""
    union = _union(
        _result("bm25f", [("R1", 1.0)]),
        _result("splade", [], error="not_run: unavailable"),
    )
    assert "generator_failure" in _risk(union, _plan()).reasons


def test_an_empty_union_is_maximum_risk():
    risk = _risk(_union(), _plan())
    assert risk.risk_band == "HIGH"
    assert "empty_union" in risk.reasons


def test_late_interaction_only_triggers_on_high_risk_and_when_available():
    union = _union()
    assert _risk(union, _plan(), late_interaction_available=False).run_late_interaction is False
    assert _risk(union, _plan(), late_interaction_available=True).run_late_interaction is True

    low = _union(*[_result(f"g{i}", [("R1", 1.0), ("R2", .5), ("R3", .2)]) for i in range(4)])
    plan = _plan("headcount by supervisory organization for all workers",
                 facets={"measure": ["H"], "dimension": ["D"],
                         "data_source": ["S"], "population": ["P"]})
    assert _risk(low, plan, late_interaction_available=True).run_late_interaction is False


# --- shortlist ---------------------------------------------------------------


def _big_union(exclusive_generator="dense_schema", decoys=60):
    """One schema-exclusive target buried under many better-fused decoys."""
    shared = [(f"D{i}", 1.0 - i * 0.001) for i in range(decoys)]
    return _union(
        _result("bm25f", shared),
        _result("splade", shared),
        _result(exclusive_generator, [*shared[:1], ("TARGET", 0.9)]),
    )


def test_a_source_exclusive_candidate_survives_a_shortlist_of_better_decoys():
    """The exact failure the architecture removes."""
    union = _big_union()
    cfg = _cfg_depth(20)
    risk = _risk(union, _plan(), cfg)
    shortlist = select_shortlist(union, risk, cfg)

    assert "TARGET" in shortlist
    assert shortlist.admitted_via("TARGET") == "schema_exclusive"
    # It would never have made a plain fused cut at this depth.
    assert union.records["TARGET"].fused_rank > 20


def test_quotas_are_floors_so_a_quiet_generator_does_not_shrink_the_shortlist():
    union = _big_union()
    cfg = _cfg_depth(30)
    shortlist = select_shortlist(union, _risk(union, _plan(), cfg), cfg)

    assert len(shortlist) == min(30, len(union))
    # Unused reservations flowed to the fused fill rather than being lost.
    assert shortlist.redistributed
    assert shortlist.quota_usage["fused"] > 0


def test_each_candidate_is_counted_once():
    union = _big_union()
    cfg = _cfg_depth(40)
    shortlist = select_shortlist(union, _risk(union, _plan(), cfg), cfg)
    ids = shortlist.instance_ids
    assert len(ids) == len(set(ids))


def test_shortlist_never_exceeds_the_risk_selected_depth():
    """Across both risk bands, and at depths other than the shipped defaults.

    (Previously this looped over two identical configs, so it only ever checked
    the standard band at the default depth.)
    """
    union = _big_union(decoys=300)
    for query, depths in (
        # 80 is the smallest standard depth the shipped quotas fit inside.
        ("headcount by organization", (80, 90)),
        ("why", (80, 90)),
        ("headcount by organization", (120, 200)),
        ("why", (120, 200)),
    ):
        standard, high = depths
        cfg = from_mapping({"shortlist": {
            "standard_rerank_depth": standard, "high_risk_rerank_depth": high,
        }})
        risk = _risk(union, _plan(query), cfg)
        shortlist = select_shortlist(union, risk, cfg)
        assert len(shortlist) <= risk.rerank_depth, (query, depths, risk.risk_band)


def test_high_risk_uses_the_expanded_depth():
    union = _big_union(decoys=300)
    cfg = DEFAULT
    high = _risk(union, _plan("why"), cfg)
    assert high.risk_band == "HIGH"
    assert len(select_shortlist(union, high, cfg)) == cfg.shortlist.high_risk_rerank_depth


def test_alternate_query_exclusive_candidates_get_reserved_room():
    """Otherwise adding an interpretation could not change what the ranker sees."""
    shared = [(f"D{i}", 1.0 - i * 0.001) for i in range(60)]
    union = _union(
        _result("bm25f", shared),
        _result("dense_identity", [*shared[:1], ("ALT", 0.5)],
                variant="Q5", variant_by_instance={"ALT": "Q5", shared[0][0]: "Q5"}),
    )
    cfg = _cfg_depth(15)
    shortlist = select_shortlist(union, _risk(union, _plan(), cfg), cfg)
    assert shortlist.admitted_via("ALT") == "alternate_query_exclusive"


def test_disabling_preservation_reproduces_the_old_single_cut():
    """Kept as an explicit ablation, so the improvement can be measured."""
    union = _big_union()
    cfg = _cfg_depth(20).with_path_overrides(
        {"shortlist.preserve_source_exclusive_candidates": False}
    )
    shortlist = select_shortlist(union, _risk(union, _plan(), cfg), cfg)
    assert "TARGET" not in shortlist
    assert all(e.admitted_via == "fused" for e in shortlist)


def test_shortlist_records_how_each_candidate_was_admitted():
    union = _big_union()
    cfg = _cfg_depth(25)
    shortlist = select_shortlist(union, _risk(union, _plan(), cfg), cfg)
    for entry in shortlist:
        assert entry.admitted_via in {"fused", *DEFAULT.shortlist.quota_map}
    assert shortlist.telemetry()["source_exclusive_admitted"] >= 1


def test_a_union_smaller_than_the_depth_is_returned_whole():
    union = _union(_result("bm25f", [("R1", 1.0), ("R2", 0.5)]))
    cfg = DEFAULT
    shortlist = select_shortlist(union, _risk(union, _plan(), cfg), cfg)
    assert len(shortlist) == 2


def test_an_empty_union_yields_an_empty_shortlist():
    union = _union()
    shortlist = select_shortlist(union, _risk(union, _plan(), DEFAULT), DEFAULT)
    assert len(shortlist) == 0
