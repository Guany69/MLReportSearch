"""Search regression against the real estate (`data/Reports.xlsx`).

Checks that the built representation ranks sensibly, that field-level metadata is
actually used in explanations rather than merely carried, that field-less reports
are not inflated, and that results are deterministic.

These build a real index (dense encoder + LSA), so they are the slow tests in the
suite.

Previously `test_phase2_search.py`, skipped in full because it named the absent
Phase 2 workbooks. Nine of its eleven assertions are about ranking and explanation
behaviour that holds for any corpus; those are here. Two were genuinely about
dual-file mechanics -- ambiguous `Where_Used` link disclosure, and the presence of
field-dictionary description text in the document string -- and legacy row-level
ingest cannot produce either (it links fields directly, so `ambiguous_field_count`
is 0 by construction). Those were deleted rather than weakened into assertions
that would pass without testing anything; the dual-file linker keeps its coverage
in `test_phase2_regression.py`, which runs today against synthetic fixtures.

Run: uv run pytest tests/test_real_estate_search.py -v
"""

from __future__ import annotations

import pytest

from reportfinder.config import DEFAULT
from reportfinder.model import ReportFinder, explain_fields
from reportfinder.represent import load_or_build

from .conftest import requires_real_estate

pytestmark = requires_real_estate

# Representative queries spanning the estate's categories.
QUERIES = [
    "Show employees who terminated last quarter by manager.",
    "Find a report with worker location, company, and supervisory organization.",
    "I need compensation amounts and salary ranges by employee.",
    "Show job requisitions, recruiters, candidates, and time to fill.",
    "Find reports related to employee benefits and coverage elections.",
]


@pytest.fixture(scope="module")
def finder(tmp_path_factory):
    """A real index over the estate, cached to a scratch dir so it doesn't
    clobber `.cache`.

    Pinned to `hybrid`: these assert on `probability` ordering and `p1`/`p2`,
    which are the mixture-of-experts posterior the hybrid path produces. The
    generator path deliberately does not emit a posterior -- it emits an
    uncalibrated ranking score and a three-way decision -- so those fields are
    not merely different there, they are meaningless.
    """
    cfg = DEFAULT.with_overrides(
        ingest_mode="legacy_single_file",
        retrieval_mode="hybrid",
        cache_dir=tmp_path_factory.mktemp("real_estate_cache"),
    )
    rep = load_or_build(cfg, rebuild=True, verbose=False)
    return ReportFinder(rep, cfg), cfg, rep


@pytest.mark.parametrize("query", QUERIES)
def test_queries_return_ranked_results(finder, query):
    reportfinder, _, _ = finder
    result = reportfinder.query(query)

    assert result.candidates, "every query must return at least one candidate"
    probs = [c.probability for c in result.candidates]
    assert probs == sorted(probs, reverse=True), "candidates must be rank-ordered"
    assert 0.0 <= result.p1 <= 1.0
    assert result.p1 >= result.p2


def test_ranking_is_deterministic(finder):
    reportfinder, _, _ = finder
    query = "Show employees who terminated last quarter by manager."
    first = reportfinder.query(query)
    second = reportfinder.query(query)
    assert [c.index for c in first.candidates] == [c.index for c in second.candidates]
    assert first.p1 == second.p1


def test_field_metadata_is_used_in_explanations(finder):
    """Field evidence must be surfaced, not just carried."""
    reportfinder, _, _ = finder
    result = reportfinder.query(
        "Find a report with worker location, company, and supervisory organization."
    )
    top = result.candidates[0]
    assert top.field_matches, "top result should cite the fields that matched"
    assert any(m.exact for m in top.field_matches), "expect literal field-name matches"
    assert explain_fields(top), "explanation lines must render"
    assert top.field_coverage is not None


def test_reports_without_matching_fields_are_not_inflated(finder):
    """A report whose fields are irrelevant must not outrank one whose fields match."""
    reportfinder, _, _ = finder
    result = reportfinder.query("gross pay and net pay by pay group", top_k=10)

    top = result.candidates[0]
    # The winner must have at least some field-level evidence; if the top hit had
    # zero matching fields while lower-ranked ones had many, the enrichment would
    # be actively misleading the ranker.
    assert top.field_matches or not any(c.field_matches for c in result.candidates), (
        "top result has no field evidence while lower results do"
    )


def test_field_coverage_is_bounded(finder):
    reportfinder, _, _ = finder
    for query in QUERIES:
        for candidate in reportfinder.query(query, top_k=5).candidates:
            if candidate.field_coverage is not None:
                assert 0.0 <= candidate.field_coverage <= 1.0
                assert candidate.concepts_covered <= candidate.concepts_total


def test_corpus_is_sane(finder):
    _, _, rep = finder
    assert len(rep) > 3000, "expected roughly the full estate after collapse"
    # Field metadata must have survived into the representation. Under legacy
    # ingest this comes from the workbook's own `Fields` column rather than from
    # a dictionary join, but it must be populated either way.
    assert any(len(m) > 0 for m in rep.frame["field_meta"])
