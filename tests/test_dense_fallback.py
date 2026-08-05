"""Retrieval must degrade cleanly when no dense model is available.

Both assertions are about the retrieval fallback path and are blind to which
workbook the corpus came from; the fixture merely happened to name
`phase2_dual_file`, which is why they were skipped.
"""

from __future__ import annotations

from reportfinder.config import DEFAULT
from reportfinder.model import ReportFinder
from reportfinder.represent import load_or_build

from .conftest import requires_real_estate

pytestmark = requires_real_estate


def test_legacy_mode_uses_lexical_lsa_when_dense_is_off():
    cfg = DEFAULT.with_overrides(
        ingest_mode="legacy_single_file", dense_mode="off",
        retrieval_mode="legacy_weighted_logit",
    )
    rep = load_or_build(cfg, verbose=False)
    result = ReportFinder(rep, cfg).query("voluntary turnover by boss")
    assert rep.dense_mode != "native"
    assert result.candidates


def test_hybrid_fallback_has_no_duplicate_dense_vote():
    cfg = DEFAULT.with_overrides(
        ingest_mode="legacy_single_file", dense_mode="off", retrieval_mode="hybrid",
    )
    result = ReportFinder(load_or_build(cfg, verbose=False), cfg).query("headcount")
    assert all("dense" not in candidate.retriever_ranks for candidate in result.candidates)
