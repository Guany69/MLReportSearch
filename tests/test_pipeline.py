"""Tests for the parser and the product-of-experts model.

Run:  uv run pytest tests/ -v
The dense-model tests need the encoder weights present locally.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from reportfinder import data as data_mod
from reportfinder.config import DEFAULT
from reportfinder.model import ReportFinder, _entropy_bits, _log_softmax
from reportfinder.represent import build_doc

FIXTURE = Path(__file__).parent / "fixtures" / "fixture_reports.xlsx"


@pytest.fixture(scope="session")
def fixture_path() -> Path:
    if not FIXTURE.exists():
        subprocess.run(
            [sys.executable, str(Path(__file__).parent / "make_fixture.py"), "600"],
            check=True,
        )
    return FIXTURE


@pytest.fixture(scope="session")
def corpus(fixture_path):
    return data_mod.load_corpus(fixture_path)


# --- parsing ----------------------------------------------------------------


def test_header_is_on_third_row(fixture_path):
    """The real header is on row 3; naive parsing picks up the banner instead."""
    naive = pd.read_excel(fixture_path)
    assert "Custom Report" not in naive.columns

    correct = data_mod.load_raw(fixture_path)
    assert "Custom Report" in correct.columns
    assert "Fields" in correct.columns
    assert len(correct.columns) == 22


def test_split_list_cell_handles_junk():
    assert data_mod.split_list_cell("A; B ;C") == ["A", "B", "C"]
    assert data_mod.split_list_cell("A; ; B;") == ["A", "B"]
    assert data_mod.split_list_cell("") == []
    assert data_mod.split_list_cell(None) == []
    assert data_mod.split_list_cell(float("nan")) == []
    assert data_mod.split_list_cell("  Single  ") == ["Single"]


def test_required_columns_are_full(corpus):
    frame = corpus.frame
    assert frame["title"].notna().all()
    assert (frame["title"].str.strip() != "").all()
    assert all(len(f) > 0 for f in frame["fields"])


def test_families_collapse_exact_duplicates(corpus):
    frame = corpus.frame
    assert len(frame) < corpus.raw_row_count, "duplicates should have collapsed"
    assert frame["family_key"].is_unique
    assert frame["family_size"].sum() == corpus.raw_row_count


def test_family_key_is_order_insensitive():
    a = data_mod._family_key("My Report", ["Worker", "Hire Date"])
    b = data_mod._family_key("my report", ["Hire Date", "worker"])
    assert a == b, "same title + same field set = same family regardless of order/case"

    c = data_mod._family_key("My Report", ["Worker", "Termination Date"])
    assert a != c, "a different field set is a different definition"


def test_nulls_are_flagged_not_zero_filled(corpus):
    """A report that never ran is not a report that ran on the epoch."""
    frame = corpus.frame
    assert frame["last_run"].isna().any(), "fixture should contain null dates"
    # Every null date must be flagged, and no null may have become a real date.
    assert (frame["last_run"].isna() == frame["last_run_missing"]).all()
    assert (frame["last_updated"].isna() == frame["last_updated_missing"]).all()
    # No epoch/zero sentinel snuck in.
    non_null = frame["last_run"].dropna()
    assert (non_null > pd.Timestamp("2000-01-01")).all()


def test_description_is_not_used_for_matching(corpus):
    """Description has ~7 distinct values and must not enter the document."""
    assert "description" not in corpus.frame.columns
    row = corpus.frame.iloc[0]
    doc = build_doc(row, DEFAULT)
    for description in ["Custom report created for business reporting needs.",
                        "Standard operational report."]:
        assert description not in doc


# --- representation ---------------------------------------------------------


def test_doc_repeats_strong_zones(corpus):
    row = corpus.frame.iloc[0]
    doc = build_doc(row, DEFAULT)
    assert doc.count(row["title"]) >= DEFAULT.w_title
    assert row["category"] in doc
    assert row["data_source"] in doc


def test_doc_zone_weights_are_honoured(corpus):
    row = corpus.frame.iloc[0]
    cfg = DEFAULT.with_overrides(w_title=1, w_fields=1, w_prompts=1)
    doc = build_doc(row, cfg)
    assert doc.count(row["title"]) == 1


# --- the model's probabilistic claims ---------------------------------------


def test_log_softmax_normalizes():
    scores = np.array([0.9, 0.5, 0.1, -0.3])
    log_p = _log_softmax(scores, 0.05)
    assert np.isclose(np.exp(log_p).sum(), 1.0)


def test_log_softmax_temperature_sharpens():
    scores = np.array([0.9, 0.5, 0.1])
    cold = np.exp(_log_softmax(scores, 0.01))
    warm = np.exp(_log_softmax(scores, 1.0))
    assert cold.max() > warm.max(), "lower temperature => peakier distribution"


def test_log_softmax_rejects_bad_temperature():
    with pytest.raises(ValueError):
        _log_softmax(np.array([1.0, 2.0]), 0.0)


def test_entropy_bounds():
    n = 8
    uniform = np.full(n, np.log(1.0 / n))
    assert np.isclose(_entropy_bits(uniform), np.log2(n)), "uniform => max entropy"

    peaked = np.log(np.array([1.0 - 1e-12] + [1e-12 / 7] * 7))
    assert _entropy_bits(peaked) < 0.01, "near-certain => ~zero entropy"


@pytest.fixture(scope="session")
def finder(fixture_path):
    """The hybrid runtime, pinned.

    This file tests the mixture-of-experts posterior -- `alpha`, `t_dense`,
    `t_lsa`, the field expert -- which are hybrid-path concepts with no counterpart
    in the generator architecture. Since `generators` became the default the mode
    has to be stated rather than inherited; the assertions are unchanged.
    """
    from reportfinder.represent import load_or_build

    cfg = DEFAULT.with_overrides(
        data_path=fixture_path,
        cache_dir=Path("/tmp/rf_test_cache"),
        retrieval_mode="hybrid",
    )
    rep = load_or_build(cfg, rebuild=False, verbose=False)
    return ReportFinder(rep, cfg), cfg


@pytest.fixture(scope="session")
def legacy_finder(finder):
    rf, cfg = finder
    legacy = cfg.with_overrides(retrieval_mode="legacy_weighted_logit")
    return ReportFinder(rf.rep, legacy), legacy


def test_posterior_is_a_valid_distribution(finder):
    """The headline claim: this is a posterior, not a score."""
    rf, _ = finder
    result = rf.query("terminated workers by supervisory organization")

    assert 0.0 <= result.p1 <= 1.0
    assert result.p1 >= result.p2, "top-1 must be >= runner-up"
    assert result.margin >= 0
    assert 0.0 <= result.normalized_entropy <= 1.0
    assert result.entropy_bits <= np.log2(result.n_reports) + 1e-6


def test_candidates_are_ranked_by_probability(finder):
    rf, _ = finder
    result = rf.query("payroll gross pay by pay group")
    probs = [c.probability for c in result.candidates]
    assert probs == sorted(probs, reverse=True)


def test_decision_rule_matches_thresholds(legacy_finder):
    rf, cfg = legacy_finder
    result = rf.query("new hires with start dates")
    expected = (result.p1 >= cfg.tau) and (result.margin >= cfg.delta)
    assert result.confident == expected
    assert len(result.candidates) == (1 if result.confident else min(cfg.top_k, result.n_reports))


def test_alpha_extremes_select_single_experts(legacy_finder):
    """α=1 must be dense-only and α=0 LSA-only -- proving it's a real mixture."""
    rf, cfg = legacy_finder
    query = "employees who transferred between departments"

    dense_only = ReportFinder(rf.rep, cfg.with_overrides(alpha=1.0)).query(query)
    lsa_only = ReportFinder(rf.rep, cfg.with_overrides(alpha=0.0)).query(query)

    # Each pure expert should rank by its own similarity.
    #
    # Asserted as "the returned report has the maximal similarity", not as
    # "its index equals argmax". Exact ties are real on this estate -- duplicate
    # report definitions produce byte-identical text and therefore identical
    # vectors -- and `np.argmax` breaks such a tie by lowest index while the
    # ranker breaks it its own way. Comparing indices tests the tie-break
    # convention, not the claim in the docstring, and flips whenever a cache
    # rebuild changes which pairs happen to tie.
    for candidates, sims in (
        (dense_only.candidates, rf._dense_similarities(query)),
        (lsa_only.candidates, rf._lsa_similarities(query)),
    ):
        top = candidates[0].index
        assert float(sims[top]) == pytest.approx(float(np.max(sims))), (
            f"pure expert returned {top} (sim {sims[top]:.6f}) but the best "
            f"available similarity is {np.max(sims):.6f}"
        )


def test_product_of_experts_can_veto(legacy_finder):
    """A product lets either expert veto -- the key difference from a weighted sum.

    The fused top-1 must not be a candidate that one expert considers hopeless,
    even if the other loves it.
    """
    rf, cfg = legacy_finder
    query = "monthly turnover by organization"
    result = rf.query(query)
    top = result.candidates[0]

    s_dense = rf._dense_similarities(query)
    s_lsa = rf._lsa_similarities(query)
    # The fused winner should sit in the upper tail of BOTH experts, not just one.
    dense_pct = (s_dense < s_dense[top.index]).mean()
    lsa_pct = (s_lsa < s_lsa[top.index]).mean()
    assert dense_pct > 0.5, f"fused winner is below-median for dense expert ({dense_pct:.2f})"
    assert lsa_pct > 0.5, f"fused winner is below-median for LSA expert ({lsa_pct:.2f})"


def test_empty_query_rejected(finder):
    rf, _ = finder
    with pytest.raises(ValueError):
        rf.query("   ")


def test_missing_workbook_gives_actionable_error(tmp_path):
    """The signature check runs before load_raw; it must not mask the message."""
    from reportfinder.represent import load_or_build

    cfg = DEFAULT.with_overrides(data_path=tmp_path / "nope.xlsx")
    with pytest.raises(FileNotFoundError, match="Report workbook not found"):
        load_or_build(cfg, verbose=False)


# --- optional field expert ---------------------------------------------------


def test_field_expert_off_by_default(finder):
    rf, _ = finder
    result = rf.query("workers with termination date and last day of work")
    assert not result.field_expert_used
    assert result.detected_fields == []


def test_field_expert_detects_literal_field_names(finder):
    rf, cfg = finder
    rf_on = ReportFinder(rf.rep, cfg.with_overrides(use_field_expert=True))
    result = rf_on.query("workers with termination date and last day of work")

    assert result.field_expert_used
    assert "termination date" in result.detected_fields
    assert "last day of work" in result.detected_fields


def test_field_expert_is_a_noop_when_nothing_detected(finder):
    """No detected fields => no extra factor => posterior unchanged."""
    rf, cfg = finder
    query = "zzz nothing here matches any field name zzz"

    off = rf.query(query)
    on = ReportFinder(rf.rep, cfg.with_overrides(use_field_expert=True)).query(query)
    assert not on.field_expert_used
    assert on.p1 == pytest.approx(off.p1)


def test_field_expert_keeps_every_report_in_support(finder):
    """Smoothing matters: a string match must not hard-veto a report to zero."""
    rf, cfg = finder
    rf_on = ReportFinder(rf.rep, cfg.with_overrides(use_field_expert=True))
    log_p, detected = rf_on._field_log_prob("report with termination date")

    assert detected, "expected at least one field to be detected"
    assert np.all(np.isfinite(log_p)), "no -inf: every report stays in the support"
    assert np.isclose(np.exp(log_p).sum(), 1.0), "field expert must be a distribution"


def test_repeated_query_is_deterministic(finder):
    rf, _ = finder
    a = rf.query("active headcount by company")
    b = rf.query("active headcount by company")
    assert a.p1 == b.p1
    assert [c.index for c in a.candidates] == [c.index for c in b.candidates]
