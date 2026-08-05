"""Plain-English queries must reach the right canonical fields.

Previously skipped because the fixture named a dual-file ingest mode whose
workbooks were absent, though nothing asserted here touched it. Runs against the real
estate.
"""

from __future__ import annotations

import pytest

from reportfinder.config import DEFAULT
from reportfinder.model import ReportFinder
from reportfinder.represent import load_or_build

from .conftest import requires_real_estate

pytestmark = requires_real_estate


@pytest.fixture(scope="module")
def finder():
    """The real estate, legacy single-file ingest.

    These assertions are about query *interpretation* -- that "Why did people
    leave?" resolves to the canonical field `Termination Reason`. That is a
    property of the lexicon and the corpus vocabulary, neither of which depends
    on which workbook the vocabulary came from. Every field named below is
    present in `data/Reports.xlsx`.

    Pinned to `hybrid` so `result.candidates` stays a meaningful assertion. Under
    the `generators` default a vague query can legitimately return no cards
    (NO_CONFIDENT_MATCH shows nothing), so asserting on candidates there would be
    asserting on an uncalibrated abstention threshold rather than on
    interpretation. The generator path's decision behaviour is measured by the
    evaluation harness, not pinned by hand here.
    """
    cfg = DEFAULT.with_overrides(
        ingest_mode="legacy_single_file", dense_mode="off", retrieval_mode="hybrid",
    )
    return ReportFinder(load_or_build(cfg, verbose=False), cfg)


@pytest.mark.parametrize(("query", "expected_fields"), [
    ("Why did people leave?", {"Termination Reason"}),
    ("Show voluntary turnover by boss.", {"Turnover Rate", "Manager"}),
    ("How big is each team?", {"Headcount"}),
    ("Show take-home pay and deductions by paycheck period.", {"Net Pay", "Deductions", "Pay Period"}),
    ("Who could replace each leader?", {"Succession Candidate", "Succession Readiness"}),
    ("Where does each employee work?", {"Worker", "Location"}),
    ("Show applicants by recruiting stage.", {"Candidate", "Application Stage"}),
    ("Which training assignments are overdue?", {"Learning Assignment", "Due Date"}),
    ("recent starters with start date", {"Hire Date"}),
    ("prior boss for each worker", {"Previous Manager"}),
    ("grouped by organization headcount", {"Supervisory Organization", "Headcount"}),
])
def test_plain_language_interpretation(finder, query, expected_fields):
    result = finder.query(query)
    parsed = {field.value for field in result.intent.fields}
    assert expected_fields <= parsed
    assert result.candidates


def test_training_does_not_drift_to_tracking(finder):
    result = finder.query("overdue training for each learner")
    assert "tracking" not in result.intent.expanded_query.casefold()
