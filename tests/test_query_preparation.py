"""Query preparation: the raw query survives, and facets are never guessed.

Two failures this guards against, both silent:

* a rewrite replacing the user's words, so the one representation guaranteed
  correct stops being searched;
* an unstated facet being filled in with a guess, which then acts as a filter and
  removes the right answer.

Run: uv run pytest tests/test_query_preparation.py -v
"""

from __future__ import annotations

import pandas as pd
import pytest

from reportfinder.config import DEFAULT
from reportfinder.pipeline.prepare import FacetState, build_query_plan
from reportfinder.query.intent import IntentParser


def _frame():
    rows = [
        {"title": "Headcount by Supervisory Organization",
         "fields": ["Headcount", "Supervisory Organization", "Worker"],
         "prompts": ["Effective Date"], "category": "Worker Data",
         "data_source": "All Workers", "report_type": "Advanced",
         "tags": "", "description": "", "area_where_used": "", "field_meta": []},
        {"title": "Termination Detail",
         "fields": ["Termination Reason", "Worker", "Hire Date"],
         "prompts": ["Effective Date"], "category": "Worker Data",
         "data_source": "All Workers", "report_type": "Advanced",
         "tags": "", "description": "", "area_where_used": "", "field_meta": []},
        {"title": "Payroll Earnings",
         "fields": ["Net Pay", "Pay Period", "Worker"],
         "prompts": [], "category": "Payroll", "data_source": "Payroll Results",
         "report_type": "Advanced", "tags": "", "description": "",
         "area_where_used": "", "field_meta": []},
    ]
    return pd.DataFrame(rows)


@pytest.fixture(scope="module")
def parser():
    return IntentParser(_frame(), DEFAULT)


def _plan(parser, query, **kwargs):
    return build_query_plan(parser.parse(query), raw_query=query, **kwargs)


# --- the raw query is inviolable --------------------------------------------


@pytest.mark.parametrize("query", [
    "headcount by supervisory organization",
    "why are we losing people",
    "workers NOT in the US, hired 2024-01-15",
    "TA partner report v2 (draft)",
    "  spaced   out   query  ",
])
def test_q0_is_always_present_and_verbatim(parser, query):
    plan = _plan(parser, query)
    raw = plan.variant("Q0")
    assert raw is not None
    assert raw.text == query
    assert raw.is_raw and raw.weight == 1.0
    assert plan.raw_query == query


def test_variants_add_rather_than_replace(parser):
    plan = _plan(parser, "headcount by boss")
    keys = [v.key for v in plan.variants]
    assert keys[0] == "Q0"
    assert len(keys) == len(set(keys))
    # Whatever else was derived, the raw query is still one of them.
    assert any(v.text == "headcount by boss" for v in plan.variants)


def test_variants_for_always_includes_the_raw_query(parser):
    """A generator may prefer another lens but must still search what was typed."""
    plan = _plan(parser, "why are we losing people")
    selected = plan.variants_for(("Q3",))
    assert any(v.is_raw for v in selected)


def test_codes_dates_and_punctuation_survive_into_q0(parser):
    query = "report v2 for 2024-01-15 (cost center CC-1234)"
    plan = _plan(parser, query)
    assert plan.variant("Q0").text == query


def test_negation_is_recorded_not_dropped(parser):
    """A dropped negation silently inverts the request."""
    assert _plan(parser, "workers not in the US").facets["negation"] is True
    assert _plan(parser, "workers in the US").facets["negation"] is False


# --- facets are explicit -----------------------------------------------------


def test_unstated_facets_are_unknown_rather_than_guessed(parser):
    plan = _plan(parser, "headcount")
    assert plan.facets["time_orientation"] is FacetState.UNKNOWN
    assert plan.facets["report_type"] is FacetState.UNKNOWN
    assert plan.facets["interface_requirement"] is FacetState.UNKNOWN


def test_stated_orientation_is_captured(parser):
    assert _plan(parser, "current headcount").facets["time_orientation"] == "current"
    assert _plan(parser, "historical headcount trend").facets["time_orientation"] == "historical"


def test_conflicting_orientation_is_multiple_not_a_coin_flip(parser):
    """Picking one would silently discard half the request."""
    plan = _plan(parser, "current and historical headcount")
    assert plan.facets["time_orientation"] is FacetState.MULTIPLE


def test_conflicting_granularity_is_multiple(parser):
    plan = _plan(parser, "total headcount and a list of each worker")
    assert plan.facets["granularity"] is FacetState.MULTIPLE


def test_granularity_is_captured_when_stated(parser):
    assert _plan(parser, "list of each worker").facets["granularity"] == "detail"
    assert _plan(parser, "total headcount").facets["granularity"] == "aggregate"


def test_absent_concepts_are_none_not_empty_guesses(parser):
    plan = _plan(parser, "zzzz")
    assert plan.facets["subject"] is FacetState.NONE
    assert plan.facets["population"] is FacetState.NONE


def test_stated_facets_excludes_the_unresolved_ones(parser):
    plan = _plan(parser, "current headcount")
    stated = plan.stated_facets()
    assert "time_orientation" in stated
    assert "report_type" not in stated


# --- the bounded alternate ---------------------------------------------------


def test_no_alternate_is_manufactured_without_ambiguity(parser):
    plan = _plan(parser, "headcount by supervisory organization")
    if plan.alternate is not None:
        # If one exists it must be a genuine reinterpretation, not a copy.
        assert plan.alternate.text != plan.raw_query


def test_alternate_can_be_disabled(parser):
    plan = _plan(parser, "show pay", enable_alternate=False)
    assert plan.alternate is None


def test_alternate_is_marked_as_an_interpretation(parser):
    """Provenance is what stops it being treated as authoritative."""
    plan = _plan(parser, "show pay")
    if plan.alternate is not None:
        assert plan.alternate.provenance == "interpretation"
        assert plan.alternate.weight < plan.variant("Q0").weight
        # Bounded: an extension of the query, not a replacement.
        assert plan.alternate.text.startswith(plan.raw_query)


# --- clarification -----------------------------------------------------------


def test_clarification_adds_context_without_editing_the_query(parser):
    plan = _plan(parser, "headcount", clarification_context=("current, not historical",))
    assert plan.variant("Q0").text == "headcount"
    clarified = plan.variant("QC")
    assert clarified is not None
    assert "current, not historical" in clarified.text
    assert clarified.text.startswith("headcount")


# --- telemetry ---------------------------------------------------------------


def test_telemetry_reports_variant_count_and_facets(parser):
    telemetry = _plan(parser, "current headcount by boss").telemetry()
    assert telemetry["query_variant_count"] >= 1
    assert "Q0" in telemetry["query_variants"]
    assert "time_orientation" in telemetry["stated_facets"]
    assert isinstance(telemetry["has_alternate_interpretation"], bool)
