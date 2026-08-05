"""Typed models for ingestion.

Plain dataclasses, matching the style already used in `data.py` / `model.py`.
Every record carries its source file and source row so any downstream claim can be
traced back to a cell in a spreadsheet.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any


class IngestMode(str, Enum):
    """How the corpus is assembled.

    One member since the dual-file path was removed. Kept as an enum rather than
    dropped because it is recorded on the corpus and in the bundle manifest as
    provenance, and because `IngestMode(value)` is what makes a config naming the
    removed mode fail loudly instead of silently reading the wrong file.
    """

    LEGACY_SINGLE_FILE = "legacy_single_file"


class MatchMethod(str, Enum):
    """How a report-field link was established. Recorded on every link.

    The workbook states each row's fields directly, so there is one way to
    establish a link and it is exact. The reconstruction methods this once had
    (normalized, composite, fuzzy, ambiguous) belonged to the removed dual-file
    path, where links were inferred rather than read.
    """

    LEGACY_FIELDS_COLUMN = "legacy_fields_column"


class AmbiguityStatus(str, Enum):
    """Whether a link is uniquely determined by the source data."""

    RESOLVED = "resolved"
    AMBIGUOUS = "ambiguous"


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

    # The workbook column that states this row's fields.
    fields_raw: str | None = None

    @property
    def report_key_id(self) -> str:
        """Compatibility spelling used by early relevance-data prototypes."""
        return self.report_key






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

    report_field_links_created: int = 0
    reports_with_zero_fields: int = 0

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
        if self.report_field_links_created:
            lines += [
                f"  fields:  {self.report_field_links_created} report-field links"
                f" | {self.reports_with_zero_fields} reports with zero fields",
                f"  quality: {self.row_validation_errors} row errors",
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
        """Deterministic template for field text.

        Only non-empty attributes are emitted so sparse metadata does not inject
        empty labels into the document and dilute the representation.
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


