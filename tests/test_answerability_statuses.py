"""The three public answerability statuses, on the real estate.

Previously skipped for an absent dual-file estate. Nothing here depends on that
ingestion -- the statuses are decided by field existence and score evidence.
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

    Every precondition these cases need holds here: 391 rows carry Worker +
    Location + Company + Supervisory Organization (the ANSWERABLE case), no row
    carries both Succession Candidate and Net Pay (the in-domain-unsatisfiable
    case), and quantum/submarine/telemetry appear nowhere in the workbook (the
    out-of-domain case, which the lexicon is additionally forbidden to ground).

    Pinned to `hybrid`: these cases assert the *legacy answerability heuristic*'s
    three statuses, which is the component they were written for and which still
    runs on that path. The generator architecture reaches the same three statuses
    through a different, deliberately uncalibrated policy -- hand-pinning its
    outputs would freeze thresholds that the decision sweep exists to change. Its
    structural contract is asserted separately below.
    """
    cfg = DEFAULT.with_overrides(
        ingest_mode="legacy_single_file", dense_mode="off", retrieval_mode="hybrid",
    )
    return ReportFinder(load_or_build(cfg, verbose=False), cfg)


@pytest.mark.parametrize(("query", "status"), [
    ("worker location, company and supervisory organization", "ANSWERABLE"),
    ("show pay", "NEEDS_CLARIFICATION"),
    ("employee report", "NEEDS_CLARIFICATION"),
    ("quantum submarine telemetry", "NO_SATISFACTORY_REPORT"),
    ("Succession Candidate with Net Pay", "NO_SATISFACTORY_REPORT"),
])
def test_public_answerability_statuses(finder, query, status):
    result = finder.query(query)
    assert result.status == status
    assert result.answerability.reasons
    assert result.answerability.calibrated is False


def test_unsatisfiable_combo_names_the_failure(finder):
    result = finder.query("Succession Candidate with Net Pay")
    assert "all requested fields" in result.answerability.reasons[0]


# --- the generator path, structurally ---------------------------------------


@pytest.fixture(scope="module")
def generator_finder():
    """The live default runtime."""
    cfg = DEFAULT.with_overrides(ingest_mode="legacy_single_file")
    return ReportFinder(load_or_build(cfg, verbose=False), cfg)


@pytest.mark.parametrize("query", [
    "worker location, company and supervisory organization",
    "show pay",
    "quantum submarine telemetry",
    "Succession Candidate with Net Pay",
])
def test_generator_path_reaches_a_grounded_three_way_status(generator_finder, query):
    """Which status each query gets is *not* pinned here, on purpose.

    The generator path's thresholds were set from six probe queries and are the
    least-evidenced numbers in the system; freezing their current outputs as
    expectations would make the sweep that fixes them look like a regression.
    What must hold regardless of where the thresholds land is the contract: one
    of exactly three statuses, a stated reason, and no claim of calibration.
    """
    result = generator_finder.query(query)

    assert result.status in {
        "ANSWERABLE", "NEEDS_CLARIFICATION", "NO_SATISFACTORY_REPORT",
    }
    assert result.answerability.reasons, "a decision must say why"
    assert result.answerability.calibrated is False
    if result.status == "NO_SATISFACTORY_REPORT":
        assert not result.candidates, "abstention must not show result cards"


def test_out_of_domain_vocabulary_is_never_answerable(generator_finder):
    """The one status assertion that does not depend on a tuned threshold:
    quantum/submarine/telemetry appear nowhere in the estate, and the lexicon is
    forbidden from grounding them, so no evidence can exist."""
    result = generator_finder.query("quantum submarine telemetry")
    assert result.status != "ANSWERABLE"
