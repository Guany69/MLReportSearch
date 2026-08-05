"""Typed models for Phase 2 ingestion.

Plain dataclasses, matching the style already used in `data.py` / `model.py`.
Every record carries its source file and source row so any downstream claim can be
traced back to a cell in a spreadsheet.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class IngestMode(str, Enum):
    """How the corpus is assembled. Both modes produce the same ReportCorpus."""

    LEGACY_SINGLE_FILE = "legacy_single_file"
    PHASE2_DUAL_FILE = "phase2_dual_file"


class MatchMethod(str, Enum):
    """How a report-field link was established. Recorded on every link."""

    EXACT_NAME = "exact_name"
    NORMALIZED_NAME = "normalized_name"
    COMPOSITE_BUSINESS_OBJECT = "composite_business_object"
    AMBIGUOUS_MULTI = "ambiguous_multi"
    FUZZY = "fuzzy"
    LEGACY_FIELDS_COLUMN = "legacy_fields_column"


class AmbiguityStatus(str, Enum):
    """Whether a link is uniquely determined by the source data."""

    RESOLVED = "resolved"
    AMBIGUOUS = "ambiguous"


class AmbiguityPolicy(str, Enum):
    """What to do when a report name matches several catalog rows.

    STRICT   -- withhold the link entirely (spec-literal). Reports whose links are
                all ambiguous end up with no fields.
    PERMISSIVE -- attach to every surviving candidate. Higher recall, at the cost of
                attaching a field to a report that may not truly have it. Links are
                still flagged AMBIGUOUS either way; this only controls attachment.
    """

    STRICT = "strict"
    PERMISSIVE = "permissive"


@dataclass(frozen=True)
class SourceRef:
    """Where a record came from."""

    source_file: str
    source_row: int  # 1-based physical row in the sheet, for human lookup


@dataclass
class ReportCatalogRecord:
    """One row of the report catalog (report-level metadata, no fields)."""

    report_key: str  # stable catalog-row identity, R####
    report_name: str  # display value
    source: SourceRef
    row_index: int  # 0-based position in the loaded frame; unique per row
    title_key: str = ""  # normalized title; intentionally not unique
    # Normalized data_source, precomputed for composite disambiguation.
    data_source_key: str = ""

    report_tags: str = ""
    category: str = ""
    worklet_landing_pages: str = ""
    owner: str = ""
    created_by: str = ""
    description: str = ""
    created_date: Any = None
    available_usage: str = ""
    number_of_times: Any = None
    last_updated_date: Any = None
    last_run_date: Any = None
    last_run_by: str = ""
    data_source: str = ""
    report_type: str = ""
    worklet: str = ""
    chart_type: str = ""
    shared: str = ""
    landing_page: str = ""
    report_prompts: str = ""
    area_where_used: str = ""

    # Phase 1 only. None in Phase 2 (the column does not exist there).
    fields_raw: str | None = None

    @property
    def report_key_id(self) -> str:
        """Compatibility spelling used by early relevance-data prototypes."""
        return self.report_key


@dataclass
class FieldDictionaryRecord:
    """One logical field, merged across however many rows defined it.

    Identity is business object + field name, so `Worker|Status` and
    `Job Requisition|Status` stay distinct fields.
    """

    field_key: str
    business_object: str
    field_name: str
    sources: list[SourceRef] = field(default_factory=list)

    description: str = ""
    report_field_type: str = ""
    related_business_object_name: str = ""
    built_in_prompts: str = ""
    domain: str = ""
    categories: str = ""
    authorized_usage: str = ""

    where_used_raw: str = ""
    where_used_report_names: list[str] = field(default_factory=list)

    # Non-empty metadata that disagreed across duplicate rows. Recorded rather than
    # overwritten: we do not get to pick a winner arbitrarily.
    conflicts: list[MetadataConflict] = field(default_factory=list)


@dataclass(frozen=True)
class MetadataConflict:
    """Two rows claimed different non-empty values for the same field attribute."""

    field_key: str
    attribute: str
    existing: str
    incoming: str
    source: SourceRef


@dataclass(frozen=True)
class ReportFieldLink:
    """A resolved (or knowingly ambiguous) edge between a report row and a field."""

    report_key: str
    field_key: str
    match_method: MatchMethod
    match_confidence: float
    ambiguity_status: AmbiguityStatus
    source_field_row: int
    report_row_index: int
    # All catalog rows the name could have meant. Length > 1 => ambiguous.
    candidate_row_indices: tuple[int, ...] = ()


@dataclass(frozen=True)
class UnmatchedWhereUsed:
    """A Where_Used entry naming a report that is not in the catalog."""

    report_name: str
    field_key: str
    source: SourceRef


@dataclass
class RowError:
    """A row-level problem. Collected, not fatal -- one bad row must not kill import."""

    source: SourceRef
    reason: str


@dataclass
class ImportSummary:
    """Observability payload returned at the end of ingestion."""

    mode: str = ""
    report_rows_read: int = 0
    valid_reports_loaded: int = 0
    # None = not measured on this path (rather than measured and found to be zero).
    duplicate_report_identities: int | None = None
    duplicate_report_rows: int | None = None

    field_rows_read: int = 0
    valid_field_rows_loaded: int = 0
    unique_field_identities: int = 0
    fields_with_no_where_used: int = 0

    report_field_links_created: int = 0
    exact_name_links: int = 0
    normalized_name_links: int = 0
    composite_links: int = 0
    fuzzy_links: int = 0
    ambiguous_links: int = 0
    duplicate_links_removed: int = 0

    where_used_entries: int = 0
    unmatched_where_used: int = 0
    reports_with_zero_fields: int = 0
    fields_with_no_matched_reports: int = 0

    metadata_conflicts: int = 0
    row_validation_errors: int = 0

    families_after_collapse: int = 0
    import_duration_s: float = 0.0

    def render(self) -> str:
        """Human-readable summary. Reports counts and identities, never field values."""
        reports_line = (
            f"  reports: {self.report_rows_read} read -> {self.valid_reports_loaded} valid"
        )
        if self.duplicate_report_identities is not None:
            reports_line += (
                f" | {self.duplicate_report_identities} duplicate identities"
                f" over {self.duplicate_report_rows} rows"
            )
        lines = [f"Import summary [{self.mode}]", reports_line]
        if self.field_rows_read:
            lines += [
                f"  fields:  {self.field_rows_read} rows read -> {self.valid_field_rows_loaded} valid"
                f" -> {self.unique_field_identities} unique identities",
                f"  links:   {self.report_field_links_created} created from"
                f" {self.where_used_entries} Where_Used entries",
                f"           exact={self.exact_name_links}"
                f" normalized={self.normalized_name_links}"
                f" composite={self.composite_links}"
                f" fuzzy={self.fuzzy_links}"
                f" ambiguous={self.ambiguous_links}",
                f"           {self.duplicate_links_removed} duplicate links removed",
                f"  gaps:    {self.unmatched_where_used} unmatched Where_Used"
                f" | {self.reports_with_zero_fields} reports with zero fields"
                f" | {self.fields_with_no_matched_reports} fields matched to no report"
                f" | {self.fields_with_no_where_used} fields with empty Where_Used",
                f"  quality: {self.metadata_conflicts} metadata conflicts"
                f" | {self.row_validation_errors} row errors",
            ]
        lines.append(
            f"  corpus:  {self.families_after_collapse} families"
            f" | {self.import_duration_s:.1f}s"
        )
        return "\n".join(lines)


@dataclass(frozen=True)
class EnrichedField:
    """A field as attached to a report, with provenance and link quality."""

    business_object: str
    field_name: str
    description: str = ""
    report_field_type: str = ""
    related_business_object_name: str = ""
    built_in_prompts: str = ""
    domain: str = ""
    categories: str = ""
    authorized_usage: str = ""

    field_key: str = ""
    match_method: str = ""
    match_confidence: float = 1.0
    ambiguity_status: str = AmbiguityStatus.RESOLVED.value
    source_file: str = ""
    source_row: int = 0

    @property
    def is_ambiguous(self) -> bool:
        return self.ambiguity_status == AmbiguityStatus.AMBIGUOUS.value

    def embedding_text(self) -> str:
        """Deterministic template for field text (per Phase 2 spec).

        Only non-empty attributes are emitted so sparse metadata (e.g. Related
        Business Object is ~22% filled) does not inject empty labels into the
        document and dilute the representation.
        """
        parts = [
            ("Field Name", self.field_name),
            ("Business Object", self.business_object),
            ("Description", self.description),
            ("Field Type", self.report_field_type),
            ("Related Business Object", self.related_business_object_name),
            ("Domain", self.domain),
            ("Categories", self.categories),
            ("Built-in Prompts", self.built_in_prompts),
        ]
        return "\n".join(f"{label}: {value}" for label, value in parts if value)


@dataclass
class LinkResult:
    """Output of the linker."""

    linked_reports: dict[int, list[EnrichedField]] = field(default_factory=dict)
    report_field_links: list[ReportFieldLink] = field(default_factory=list)
    unmatched_where_used: list[UnmatchedWhereUsed] = field(default_factory=list)
    ambiguous_matches: list[ReportFieldLink] = field(default_factory=list)
    validation_metrics: ImportSummary = field(default_factory=ImportSummary)
