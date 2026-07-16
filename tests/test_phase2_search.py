"""Search regression for Phase 2.

Checks that the enriched representation still ranks sensibly, that Phase 2
metadata is actually used, that field-less reports are not inflated, and that
results are deterministic.

These build a real index (dense encoder + LSA), so they are the slow tests in the
suite. They run against the real workbooks when present.

Run: uv run pytest tests/test_phase2_search.py -v
"""

from __future__ import annotations

import numpy as np
import pytest

from reportfinder.config import DEFAULT
from reportfinder.model import Candidate, ExpertTrace, ReportFinder, explain_fields
from reportfinder.represent import load_or_build

REAL_CATALOG = DEFAULT.catalog_path
REAL_DICTIONARY = DEFAULT.field_dictionary_path

requires_real_data = pytest.mark.skipif(
    not (REAL_CATALOG.exists() and REAL_DICTIONARY.exists()),
    reason="real Phase 2 workbooks not present in data/",
)

# Representative queries from the Phase 2 brief.
QUERIES = [
    "Show employees who terminated last quarter by manager.",
    "Find a report with worker location, company, and supervisory organization.",
    "I need compensation amounts and salary ranges by employee.",
    "Show job requisitions, recruiters, candidates, and time to fill.",
    "Find reports related to employee benefits and coverage elections.",
]


@pytest.fixture(scope="module")
def phase2_finder(tmp_path_factory):
    """A real Phase 2 index. Cached to a scratch dir so it doesn't clobber .cache."""
    cfg = DEFAULT.with_overrides(
        ingest_mode="phase2_dual_file",
        cache_dir=tmp_path_factory.mktemp("phase2_cache"),
    )
    rep = load_or_build(cfg, rebuild=True, verbose=False)
    return ReportFinder(rep, cfg), cfg, rep


@requires_real_data
@pytest.mark.parametrize("query", QUERIES)
def test_queries_return_ranked_results(phase2_finder, query):
    finder, cfg, _ = phase2_finder
    result = finder.query(query)

    assert result.candidates, "every query must return at least one candidate"
    probs = [c.probability for c in result.candidates]
    assert probs == sorted(probs, reverse=True), "candidates must be rank-ordered"
    assert 0.0 <= result.p1 <= 1.0
    assert result.p1 >= result.p2


@requires_real_data
def test_ranking_is_deterministic(phase2_finder):
    finder, _, _ = phase2_finder
    query = "Show employees who terminated last quarter by manager."
    first = finder.query(query)
    second = finder.query(query)
    assert [c.index for c in first.candidates] == [c.index for c in second.candidates]
    assert first.p1 == second.p1


@requires_real_data
def test_phase2_field_metadata_is_used_in_explanations(phase2_finder):
    """Phase 2 must actually surface field-level evidence, not just carry it."""
    finder, _, _ = phase2_finder
    result = finder.query(
        "Find a report with worker location, company, and supervisory organization."
    )
    top = result.candidates[0]
    assert top.field_matches, "top result should cite the fields that matched"
    assert any(m.exact for m in top.field_matches), "expect literal field-name matches"
    assert explain_fields(top), "explanation lines must render"
    assert top.field_coverage is not None


@requires_real_data
def test_explanations_disclose_ambiguous_links(phase2_finder):
    """Any report carrying ambiguous links must say so rather than imply certainty.

    Ranking is not used to reach the ambiguous report: duplicate titles mean the
    intended row may not win its own title query. The candidate is built directly
    from a known-ambiguous row so the assertion is deterministic.
    """
    finder, _, rep = phase2_finder
    ambiguous_rows = np.flatnonzero(rep.frame["has_ambiguous_fields"].to_numpy())
    assert ambiguous_rows.size > 0, "real corpus is known to contain ambiguous links"

    idx = int(ambiguous_rows[0])
    row = rep.frame.iloc[idx]

    # Query with a term drawn from the row's own fields so the field matcher fires.
    ambiguous_field = next(f for f in row["field_meta"] if f.is_ambiguous)
    matches, total, covered = finder._explain_fields(ambiguous_field.field_name, idx)
    assert matches, "expected the field matcher to cite this report's fields"

    candidate = Candidate(
        index=idx,
        probability=0.5,
        row=row,
        trace=ExpertTrace(0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
        field_matches=matches,
        concepts_total=total,
        concepts_covered=covered,
    )
    if not candidate.has_ambiguous_links:
        pytest.skip("matched fields for this row happened to be unambiguous")

    lines = " ".join(explain_fields(candidate))
    assert "could not be uniquely resolved" in lines
    assert "shares its name" in lines


@requires_real_data
def test_reports_without_matching_fields_are_not_inflated(phase2_finder):
    """A report whose fields are irrelevant must not outrank one whose fields match."""
    finder, _, rep = phase2_finder
    result = finder.query("gross pay and net pay by pay group", top_k=10)

    top = result.candidates[0]
    # The winner must have at least some field-level evidence; if the top hit had
    # zero matching fields while lower-ranked ones had many, the enrichment would
    # be actively misleading the ranker.
    assert top.field_matches or not any(c.field_matches for c in result.candidates), (
        "top result has no field evidence while lower results do"
    )


@requires_real_data
def test_field_coverage_is_bounded(phase2_finder):
    finder, _, _ = phase2_finder
    for query in QUERIES:
        for candidate in finder.query(query, top_k=5).candidates:
            if candidate.field_coverage is not None:
                assert 0.0 <= candidate.field_coverage <= 1.0
                assert candidate.concepts_covered <= candidate.concepts_total


@requires_real_data
def test_phase2_corpus_is_sane(phase2_finder):
    _, _, rep = phase2_finder
    assert len(rep) > 3000, "expected roughly the full estate after collapse"
    # Field metadata must have survived into the representation.
    assert any(len(m) > 0 for m in rep.frame["field_meta"])
    # And the docs must contain the Phase 2 description text.
    assert any("associated with the reporting record" in d for d in rep.docs)
