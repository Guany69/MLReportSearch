"""Ingest primitives shared by the whole pipeline: normalization and catalog loading.

These were extracted from the dual-file ingest suite when that path was removed.
They are not dual-file tests -- `normalize.py` backs the single-file row
reader and `ReportCatalogLoader` is what reads `Reports.xlsx` at row granularity,
so both are on the live path and were only ever tested from there by accident of
where the file happened to live.

Run: uv run pytest tests/test_ingest_catalog.py -v
"""

from __future__ import annotations

import pandas as pd
import pytest

from reportfinder.ingest.catalog import ReportCatalogLoader
from reportfinder.ingest.normalize import (
    normalize_display,
    normalize_header,
    normalize_match,
    resolve_columns,
    split_multi,
)

from .ingest_fixtures import catalog_row, write_catalog

# --- normalization ----------------------------------------------------------


def test_split_multi_handles_semicolons_and_newlines():
    assert split_multi("A; B;C") == ["A", "B", "C"]
    assert split_multi("A\nB\r\nC") == ["A", "B", "C"]
    assert split_multi("A;\n B ;;C\n") == ["A", "B", "C"]
    assert split_multi("") == []
    assert split_multi(None) == []
    assert split_multi(float("nan")) == []


def test_split_multi_never_splits_on_comma():
    """Commas legitimately occur inside report names."""
    assert split_multi("Headcount by Region, Country and Company") == [
        "Headcount by Region, Country and Company"
    ]


def test_split_multi_dedupes_preserving_order():
    assert split_multi("B; A; B; a") == ["B", "A"]


def test_normalize_match_folds_only_cosmetic_differences():
    assert normalize_match("  Worker   Snapshot ") == "worker snapshot"
    assert normalize_match("Report – Copy") == "report - copy"  # en-dash
    assert normalize_match("Report — Copy") == "report - copy"  # em-dash
    assert normalize_match("REPORT") == normalize_match("report")


@pytest.mark.parametrize(
    "a,b",
    [
        ("Report", "Report v2"),
        ("Report", "Report - Copy"),
        ("Report - Summary", "Report - Current"),
        ("Report - Audit", "Report - Copy"),
        ("Report v2", "Report v3"),
    ],
)
def test_normalize_match_keeps_meaningful_variants_distinct(a, b):
    """Versions/suffixes are load-bearing and must never be normalized away."""
    assert normalize_match(a) != normalize_match(b)


def test_normalize_display_preserves_original_text():
    assert normalize_display("  AXIS - Worker  Snapshot ") == "AXIS - Worker  Snapshot"
    assert normalize_display(None) == ""
    assert normalize_display("nan") == ""


def test_normalize_header_folds_case_underscores_and_spacing():
    assert normalize_header("Where_Used") == normalize_header("where used")
    assert normalize_header("  LAST  RUN Date ") == "last run date"
    assert normalize_header("Report Tag(s)") == "report tag s"


def test_resolve_columns_matches_any_listed_alias():
    """Header spellings that don't normalize alike are covered by listing both."""
    aliases = {"where_used": ("Where_Used", "Where Used")}
    assert resolve_columns(["Where Used"], aliases) == {"where_used": "Where Used"}
    assert resolve_columns(["Where_Used"], aliases) == {"where_used": "Where_Used"}


def test_resolve_columns_omits_absent_optional_columns():
    resolved = resolve_columns(
        ["Custom Report"], {"report_name": ("Custom Report",), "category": ("Category",)}
    )
    assert resolved == {"report_name": "Custom Report"}, "absent optional is simply missing"


# --- catalog loading --------------------------------------------------------


def test_catalog_loads_without_fields_column(tmp_path):
    path = write_catalog(
        [catalog_row("Alpha"), catalog_row("Beta")], tmp_path / "cat.xlsx"
    )
    records = ReportCatalogLoader().load(path)
    assert [r.report_name for r in records] == ["Alpha", "Beta"]
    assert all(r.fields_raw is None for r in records), "no Fields column present"


def test_catalog_missing_report_name_column_fails_fast(tmp_path):
    frame = pd.DataFrame([{"Nonsense": "x"}])
    from .ingest_fixtures import _write_with_banner

    path = tmp_path / "bad.xlsx"
    _write_with_banner(frame, path, "banner")
    with pytest.raises(ValueError, match="missing required column"):
        ReportCatalogLoader().load(path)


def test_catalog_skips_blank_rows_and_records_provenance(tmp_path):
    path = write_catalog(
        [catalog_row("Alpha"), catalog_row(""), catalog_row("Beta")],
        tmp_path / "cat.xlsx",
    )
    records = ReportCatalogLoader().load(path)
    assert len(records) == 2
    # Header on row 3 => first data record is physical row 4.
    assert records[0].source.source_row == 4
    assert records[0].source.source_file == "cat.xlsx"


def test_catalog_header_aliases(tmp_path):
    rows = [catalog_row("Alpha")]
    frame = pd.DataFrame(rows)
    frame = frame.rename(columns={"Custom Report": "Report Name"})
    from .ingest_fixtures import _write_with_banner

    path = tmp_path / "alias.xlsx"
    _write_with_banner(frame, path, "banner")
    records = ReportCatalogLoader().load(path)
    assert records[0].report_name == "Alpha"


def test_catalog_recognizes_the_real_run_count_header(tmp_path):
    """A real exported catalog header reads "Number of Times Executed", which was
    not in the alias list -- so the column resolved to nothing and every run count,
    plus the family-rank ordering built on it, was NaN on the real estate."""
    frame = pd.DataFrame([catalog_row("Alpha")]).rename(
        columns={"Number of Times": "Number of Times Executed"}
    )
    from .ingest_fixtures import _write_with_banner

    path = tmp_path / "runs.xlsx"
    _write_with_banner(frame, path, "banner")
    records = ReportCatalogLoader().load(path)
    assert records[0].number_of_times == 1


def test_catalog_whitespace_normalized_but_display_preserved(tmp_path):
    path = write_catalog([catalog_row("  Alpha  Report ")], tmp_path / "cat.xlsx")
    record = ReportCatalogLoader().load(path)[0]
    assert record.report_name == "Alpha  Report"  # display: inner spacing kept
    assert record.report_key == "R0004"  # stable physical catalog-row identity
    assert record.title_key == "alpha report"  # normalized title lookup


def test_catalog_duplicate_names_are_distinct_rows(tmp_path):
    path = write_catalog(
        [catalog_row("Dup"), catalog_row("Dup")], tmp_path / "cat.xlsx"
    )
    records = ReportCatalogLoader().load(path)
    assert len(records) == 2
    assert records[0].report_key != records[1].report_key
    assert records[0].title_key == records[1].title_key
    assert records[0].row_index != records[1].row_index


def test_catalog_missing_optional_columns_is_tolerated(tmp_path):
    frame = pd.DataFrame([{"Custom Report": "Alpha"}])
    from .ingest_fixtures import _write_with_banner

    path = tmp_path / "min.xlsx"
    _write_with_banner(frame, path, "banner")
    record = ReportCatalogLoader().load(path)[0]
    assert record.report_name == "Alpha"
    assert record.category == ""
