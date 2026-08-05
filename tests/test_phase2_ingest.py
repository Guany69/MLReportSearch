"""Tests for Phase 2 ingestion: loading, normalization, linking, enrichment.

Run: uv run pytest tests/test_phase2_ingest.py -v
"""

from __future__ import annotations

import pandas as pd
import pytest

from reportfinder.config import DEFAULT
from reportfinder.ingest import build_corpus
from reportfinder.ingest.catalog import ReportCatalogLoader
from reportfinder.ingest.field_dictionary import FieldDictionaryLoader, make_field_key
from reportfinder.ingest.linker import ReportFieldLinker
from reportfinder.ingest.models import AmbiguityPolicy, AmbiguityStatus, MatchMethod
from reportfinder.ingest.normalize import (
    normalize_display,
    normalize_header,
    normalize_match,
    resolve_columns,
    split_multi,
)

from .phase2_fixtures import (
    catalog_row,
    dictionary_row,
    write_catalog,
    write_dictionary,
)

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
    assert all(r.fields_raw is None for r in records), "Phase 2 catalog has no Fields"


def test_catalog_missing_report_name_column_fails_fast(tmp_path):
    frame = pd.DataFrame([{"Nonsense": "x"}])
    from .phase2_fixtures import _write_with_banner

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
    from .phase2_fixtures import _write_with_banner

    path = tmp_path / "alias.xlsx"
    _write_with_banner(frame, path, "banner")
    records = ReportCatalogLoader().load(path)
    assert records[0].report_name == "Alpha"


def test_catalog_recognizes_the_real_run_count_header(tmp_path):
    """The real Phase 2 catalog header is "Number of Times Executed", which was
    not in the alias list -- so the column resolved to nothing and every run count,
    plus the family-rank ordering built on it, was NaN on the real estate."""
    frame = pd.DataFrame([catalog_row("Alpha")]).rename(
        columns={"Number of Times": "Number of Times Executed"}
    )
    from .phase2_fixtures import _write_with_banner

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
    from .phase2_fixtures import _write_with_banner

    path = tmp_path / "min.xlsx"
    _write_with_banner(frame, path, "banner")
    record = ReportCatalogLoader().load(path)[0]
    assert record.report_name == "Alpha"
    assert record.category == ""


# --- field dictionary loading ----------------------------------------------


def test_dictionary_requires_core_columns(tmp_path):
    frame = pd.DataFrame([{"Business Object": "Worker", "Field Name": "Status"}])
    from .phase2_fixtures import _write_with_banner

    path = tmp_path / "d.xlsx"
    _write_with_banner(frame, path, "banner")
    with pytest.raises(ValueError, match="missing required column"):
        FieldDictionaryLoader().load(path)


def test_dictionary_blank_identity_rows_are_errors_not_fatal(tmp_path):
    path = write_dictionary(
        [
            dictionary_row("Worker", "Status", "Alpha"),
            dictionary_row("", "Orphan", "Alpha"),
            dictionary_row("Worker", "", "Alpha"),
            dictionary_row("Worker", "Hire Date", "Alpha"),
        ],
        tmp_path / "d.xlsx",
    )
    loader = FieldDictionaryLoader()
    records = loader.load(path)
    assert len(records) == 2, "valid rows still load"
    assert len(loader.errors) == 2, "bad rows recorded, not silently dropped"


def test_dictionary_blank_where_used_yields_no_links(tmp_path):
    path = write_dictionary(
        [dictionary_row("Worker", "Status", "")], tmp_path / "d.xlsx"
    )
    record = FieldDictionaryLoader().load(path)[0]
    assert record.where_used_report_names == []


def test_field_identity_separates_same_name_different_object():
    a = make_field_key("Worker", "Status")
    b = make_field_key("Job Requisition", "Status")
    assert a != b, "same display name, different business object => different fields"


def test_dictionary_merges_repeated_records(tmp_path):
    path = write_dictionary(
        [
            dictionary_row("Worker", "Status", "Alpha"),
            dictionary_row("Worker", "Status", "Beta"),
        ],
        tmp_path / "d.xlsx",
    )
    records = FieldDictionaryLoader().load(path)
    assert len(records) == 1
    assert records[0].where_used_report_names == ["Alpha", "Beta"]
    assert len(records[0].sources) == 2, "provenance from both rows retained"


def test_dictionary_flags_conflicting_metadata_without_overwriting(tmp_path):
    path = write_dictionary(
        [
            dictionary_row("Worker", "Status", "Alpha", Description="First meaning."),
            dictionary_row("Worker", "Status", "Beta", Description="Different meaning."),
        ],
        tmp_path / "d.xlsx",
    )
    loader = FieldDictionaryLoader()
    record = loader.load(path)[0]
    assert record.description == "First meaning.", "first value kept, not overwritten"
    assert len(loader.conflicts) == 1
    assert loader.conflicts[0].attribute == "description"


def test_dictionary_fills_blanks_from_later_rows(tmp_path):
    path = write_dictionary(
        [
            dictionary_row("Worker", "Status", "Alpha", Domain=""),
            dictionary_row("Worker", "Status", "Beta", Domain="Worker Data"),
        ],
        tmp_path / "d.xlsx",
    )
    loader = FieldDictionaryLoader()
    record = loader.load(path)[0]
    assert record.domain == "Worker Data"
    assert not loader.conflicts, "filling a blank is not a conflict"


# --- linking ----------------------------------------------------------------


def _link(catalog_rows, dictionary_rows, tmp_path, **linker_kwargs):
    cat = write_catalog(catalog_rows, tmp_path / "c.xlsx")
    dic = write_dictionary(dictionary_rows, tmp_path / "d.xlsx")
    reports = ReportCatalogLoader().load(cat)
    fields = FieldDictionaryLoader().load(dic)
    return reports, ReportFieldLinker(**linker_kwargs).link(reports, fields)


def test_link_exact_name(tmp_path):
    reports, result = _link(
        [catalog_row("Alpha")], [dictionary_row("Worker", "Status", "Alpha")], tmp_path
    )
    assert result.report_field_links[0].match_method is MatchMethod.EXACT_NAME
    assert [f.field_name for f in result.linked_reports[0]] == ["Status"]


def test_link_case_and_whitespace_normalized(tmp_path):
    reports, result = _link(
        [catalog_row("Alpha Report")],
        [dictionary_row("Worker", "Status", "  alpha   report ")],
        tmp_path,
    )
    assert result.report_field_links[0].match_method is MatchMethod.NORMALIZED_NAME
    assert len(result.linked_reports[0]) == 1


def test_link_dash_normalization(tmp_path):
    reports, result = _link(
        [catalog_row("Alpha - Copy")],
        [dictionary_row("Worker", "Status", "Alpha – Copy")],  # en-dash
        tmp_path,
    )
    assert result.linked_reports[0][0].field_name == "Status"


def test_link_report_name_with_comma(tmp_path):
    name = "Headcount by Region, Country and Company"
    reports, result = _link(
        [catalog_row(name)], [dictionary_row("Worker", "Status", name)], tmp_path
    )
    assert result.unmatched_where_used == []
    assert len(result.linked_reports[0]) == 1


def test_link_versioned_reports_stay_distinct(tmp_path):
    reports, result = _link(
        [catalog_row("Alpha"), catalog_row("Alpha v2")],
        [dictionary_row("Worker", "Status", "Alpha v2")],
        tmp_path,
    )
    assert 0 not in result.linked_reports, "plain 'Alpha' must not receive v2's field"
    assert [f.field_name for f in result.linked_reports[1]] == ["Status"]


def test_link_copy_and_summary_stay_distinct(tmp_path):
    reports, result = _link(
        [catalog_row("Alpha - Copy"), catalog_row("Alpha - Summary")],
        [dictionary_row("Worker", "Status", "Alpha - Summary")],
        tmp_path,
    )
    assert 0 not in result.linked_reports
    assert len(result.linked_reports[1]) == 1


def test_link_unmatched_where_used_is_recorded(tmp_path):
    reports, result = _link(
        [catalog_row("Alpha")],
        [dictionary_row("Worker", "Status", "Does Not Exist")],
        tmp_path,
    )
    assert len(result.unmatched_where_used) == 1
    assert result.unmatched_where_used[0].report_name == "Does Not Exist"
    assert not result.linked_reports


def test_typed_custom_report_prefix_is_stripped_and_resolved(tmp_path):
    """`Where_Used` holds typed Workday references, not bare titles. Comparing
    them raw against catalog titles matched zero of 20,951 distinct entries on the
    real estate, because the catalog stores the title without the type."""
    reports, result = _link(
        [catalog_row("Alpha")],
        [dictionary_row("Worker", "Status", "Custom Report - Alpha")],
        tmp_path,
    )
    assert result.report_field_links[0].match_method is MatchMethod.EXACT_NAME
    assert [f.field_name for f in result.linked_reports[0]] == ["Status"]
    assert result.validation_metrics.non_report_where_used_skipped == 0
    assert result.unmatched_where_used == []


def test_recognized_non_report_types_are_skipped_not_unmatched(tmp_path):
    """A calculated field is not a report that went missing. On the real estate
    these outnumber real report references roughly ten to one, so counting them as
    unmatched would bury every entry that genuinely failed to resolve."""
    reports, result = _link(
        [catalog_row("Alpha")],
        [
            dictionary_row("Worker", "Status", "Calculated Field - CF_TF_CSO"),
            dictionary_row("Worker", "Grade", "Condition Rule - Location is US"),
        ],
        tmp_path,
    )
    assert result.unmatched_where_used == []
    assert result.validation_metrics.non_report_where_used_skipped == 2
    assert not result.linked_reports


def test_a_direct_title_match_beats_typed_parsing(tmp_path):
    """A report may legitimately be titled with a dash. Stripping a prefix before
    trying the raw value would destroy such a title; matching first cannot."""
    reports, result = _link(
        [catalog_row("Custom Report - Alpha"), catalog_row("Alpha")],
        [dictionary_row("Worker", "Status", "Custom Report - Alpha")],
        tmp_path,
    )
    assert [f.field_name for f in result.linked_reports[0]] == ["Status"]
    assert 1 not in result.linked_reports, "the bare title must not receive it"


def test_an_unrecognized_prefix_stays_actionable(tmp_path):
    """An unenumerated type falls through to unmatched rather than being guessed
    away as a non-report -- otherwise a new Workday type would silently delete
    links and nothing would say so."""
    reports, result = _link(
        [catalog_row("Alpha")],
        [dictionary_row("Worker", "Status", "Flux Capacitor - Thing")],
        tmp_path,
    )
    assert len(result.unmatched_where_used) == 1
    assert result.unmatched_where_used[0].report_name == "Flux Capacitor - Thing"
    assert result.validation_metrics.non_report_where_used_skipped == 0


def test_a_typed_report_that_does_not_exist_is_unmatched_under_its_raw_value(tmp_path):
    reports, result = _link(
        [catalog_row("Alpha")],
        [dictionary_row("Worker", "Status", "Custom Report - Nonexistent")],
        tmp_path,
    )
    assert len(result.unmatched_where_used) == 1
    assert result.unmatched_where_used[0].report_name == "Custom Report - Nonexistent", (
        "recorded under the original cell value, so the diagnostic is greppable"
    )
    assert result.validation_metrics.non_report_where_used_skipped == 0


def test_typed_parsing_is_deterministic(tmp_path):
    rows = [
        dictionary_row("Worker", "Status", "Custom Report - Alpha"),
        dictionary_row("Worker", "Grade", "Calculated Field - CF_X"),
    ]
    first = _link([catalog_row("Alpha")], rows, tmp_path / "a")[1]
    second = _link([catalog_row("Alpha")], rows, tmp_path / "b")[1]
    assert [f.field_name for f in first.linked_reports[0]] == [
        f.field_name for f in second.linked_reports[0]
    ]
    assert (
        first.validation_metrics.non_report_where_used_skipped
        == second.validation_metrics.non_report_where_used_skipped
    )


def test_link_composite_resolves_duplicate_names(tmp_path):
    """Duplicate title + business object == data source narrows to exactly one row."""
    reports, result = _link(
        [
            catalog_row("Dup", **{"Data Source": "All Workers"}),
            catalog_row("Dup", **{"Data Source": "Payroll Results"}),
        ],
        [dictionary_row("Payroll Results", "Gross Pay", "Dup")],
        tmp_path,
    )
    link = result.report_field_links[0]
    assert link.match_method is MatchMethod.COMPOSITE_BUSINESS_OBJECT
    assert link.ambiguity_status is AmbiguityStatus.RESOLVED
    assert 0 not in result.linked_reports
    assert [f.field_name for f in result.linked_reports[1]] == ["Gross Pay"]


def test_link_ambiguous_when_composite_cannot_decide(tmp_path):
    """Same title AND same data source => genuinely undecidable."""
    reports, result = _link(
        [catalog_row("Dup"), catalog_row("Dup")],
        [dictionary_row("All Workers", "Status", "Dup")],
        tmp_path,
    )
    link = result.report_field_links[0]
    assert link.ambiguity_status is AmbiguityStatus.AMBIGUOUS
    assert link.match_method is MatchMethod.AMBIGUOUS_MULTI
    assert set(link.candidate_row_indices) == {0, 1}, "all candidates recorded"
    assert len(result.ambiguous_matches) == 1


def test_strict_policy_withholds_ambiguous_links(tmp_path):
    reports, result = _link(
        [catalog_row("Dup"), catalog_row("Dup")],
        [dictionary_row("All Workers", "Status", "Dup")],
        tmp_path,
        policy=AmbiguityPolicy.STRICT,
    )
    assert result.linked_reports == {}, "strict attaches nothing when undecidable"
    assert len(result.ambiguous_matches) == 1, "but still reports the ambiguity"


def test_permissive_policy_attaches_to_all_candidates(tmp_path):
    reports, result = _link(
        [catalog_row("Dup"), catalog_row("Dup")],
        [dictionary_row("All Workers", "Status", "Dup")],
        tmp_path,
        policy=AmbiguityPolicy.PERMISSIVE,
    )
    assert len(result.linked_reports) == 2
    assert all(
        f.is_ambiguous for fields in result.linked_reports.values() for f in fields
    ), "attached, but every such field is flagged ambiguous"


def test_duplicate_links_are_removed(tmp_path):
    """The same field naming the same report twice yields one edge."""
    reports, result = _link(
        [catalog_row("Alpha")],
        [dictionary_row("Worker", "Status", "Alpha; Alpha")],
        tmp_path,
    )
    assert len(result.linked_reports[0]) == 1


def test_linking_is_deterministic(tmp_path):
    rows_cat = [catalog_row("Alpha"), catalog_row("Beta")]
    rows_dic = [
        dictionary_row("Worker", "Zeta", "Alpha; Beta"),
        dictionary_row("Worker", "Alpha Field", "Alpha"),
        dictionary_row("Worker", "Mid", "Alpha"),
    ]
    first = _link(rows_cat, rows_dic, tmp_path / "a")[1]
    second = _link(rows_cat, rows_dic, tmp_path / "b")[1]
    for key in first.linked_reports:
        assert [f.field_name for f in first.linked_reports[key]] == [
            f.field_name for f in second.linked_reports[key]
        ]


def test_fuzzy_is_off_by_default(tmp_path):
    reports, result = _link(
        [catalog_row("Alpha Report")],
        [dictionary_row("Worker", "Status", "Alpha Reprot")],  # typo
        tmp_path,
    )
    assert len(result.unmatched_where_used) == 1, "no fuzzy fallback unless enabled"


def test_fuzzy_when_enabled_matches_near_names(tmp_path):
    reports, result = _link(
        [catalog_row("Alpha Report")],
        [dictionary_row("Worker", "Status", "Alpha Reprot")],
        tmp_path,
        enable_fuzzy=True,
        fuzzy_threshold=0.8,
    )
    assert result.report_field_links[0].match_method is MatchMethod.FUZZY


# --- enrichment -------------------------------------------------------------


def _corpus(catalog_rows, dictionary_rows, tmp_path, **overrides):
    cat = write_catalog(catalog_rows, tmp_path / "c.xlsx")
    dic = write_dictionary(dictionary_rows, tmp_path / "d.xlsx")
    cfg = DEFAULT.with_overrides(
        ingest_mode="phase2_dual_file", catalog_path=cat, field_dictionary_path=dic,
        **overrides,
    )
    return build_corpus(cfg)


def test_enrich_report_with_multiple_fields(tmp_path):
    corpus, summary = _corpus(
        [catalog_row("Alpha")],
        [
            dictionary_row("Worker", "Status", "Alpha"),
            dictionary_row("Worker", "Hire Date", "Alpha"),
        ],
        tmp_path,
    )
    row = corpus.frame.iloc[0]
    assert sorted(row["fields"]) == ["Hire Date", "Status"]
    assert len(row["field_meta"]) == 2
    assert summary.report_field_links_created == 2


def test_enrich_field_used_by_multiple_reports(tmp_path):
    corpus, _ = _corpus(
        [catalog_row("Alpha"), catalog_row("Beta")],
        [dictionary_row("Worker", "Status", "Alpha; Beta")],
        tmp_path,
    )
    by_title = {r["title"]: r for _, r in corpus.frame.iterrows()}
    assert by_title["Alpha"]["fields"] == ["Status"]
    assert by_title["Beta"]["fields"] == ["Status"]


def test_enrich_report_with_zero_fields(tmp_path):
    corpus, summary = _corpus(
        [catalog_row("Alpha"), catalog_row("Orphan")],
        [dictionary_row("Worker", "Status", "Alpha")],
        tmp_path,
    )
    by_title = {r["title"]: r for _, r in corpus.frame.iterrows()}
    assert by_title["Orphan"]["fields"] == []
    assert summary.reports_with_zero_fields == 1


def test_enrich_preserves_field_provenance(tmp_path):
    corpus, _ = _corpus(
        [catalog_row("Alpha")], [dictionary_row("Worker", "Status", "Alpha")], tmp_path
    )
    meta = corpus.frame.iloc[0]["field_meta"][0]
    assert meta.source_file == "d.xlsx"
    assert meta.source_row == 4  # header row 3 => first data row is 4
    assert meta.business_object == "Worker"
    assert meta.match_method == MatchMethod.EXACT_NAME.value


def test_enriched_field_embedding_text_is_deterministic_and_skips_blanks(tmp_path):
    corpus, _ = _corpus(
        [catalog_row("Alpha")],
        [dictionary_row("Worker", "Status", "Alpha", **{"Related Business Object Name": ""})],
        tmp_path,
    )
    text = corpus.frame.iloc[0]["field_meta"][0].embedding_text()
    assert text.startswith("Field Name: Status")
    assert "Business Object: Worker" in text
    assert "Related Business Object" not in text, "blank attributes are omitted"


def test_import_summary_counts(tmp_path):
    _, summary = _corpus(
        [catalog_row("Alpha"), catalog_row("Beta")],
        [
            dictionary_row("Worker", "Status", "Alpha; Beta"),
            dictionary_row("Worker", "Hire Date", "Nonexistent"),
        ],
        tmp_path,
    )
    assert summary.report_rows_read == 2
    assert summary.field_rows_read == 2
    assert summary.unique_field_identities == 2
    assert summary.where_used_entries == 3
    assert summary.unmatched_where_used == 1
    assert summary.report_field_links_created == 2
    assert summary.fields_with_no_matched_reports == 1
    assert summary.import_duration_s >= 0
