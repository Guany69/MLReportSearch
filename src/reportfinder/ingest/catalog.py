"""Report catalog loader: one row of report metadata per catalog row.

Reads the export's quirky layout -- banner row, "End Date" row, real header on the
third physical row -- and tolerates the absence of `Fields`.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from .models import ReportCatalogRecord, RowError, SourceRef
from .normalize import normalize_display, normalize_match, resolve_columns

# Logical name -> accepted header spellings. Resolution is by normalized header,
# so case/spacing/underscore variants are already covered; these are genuine
# alternate namings.
COLUMN_ALIASES: dict[str, tuple[str, ...]] = {
    "report_name": ("Custom Report", "Report Name", "Report"),
    "report_tags": ("Report Tag(s)", "Report Tags", "Tags"),
    "category": ("Category",),
    "worklet_landing_pages": ("Worklet Landing Pages",),
    "owner": ("Owner",),
    "created_by": ("Created by", "Created By"),
    "description": ("Description",),
    "created_date": ("Created Date",),
    "available_usage": ("Available Usage",),
    # "Number of Times Executed" is what a real exported catalog header says.
    # Without it the column resolved to nothing, so every downstream run count --
    # and the family-rank ordering that uses it -- was NaN on the real estate.
    "number_of_times": (
        "Number of Times", "Number of Times Executed", "Number of Times Run",
        "Run Count",
    ),
    "last_updated_date": ("Last Updated Date",),
    "last_run_date": ("Last Run Date",),
    "last_run_by": ("Last Run By",),
    "data_source": ("Data Source",),
    "report_type": ("Report Type",),
    "worklet": ("Worklet",),
    "chart_type": ("Chart Type",),
    "shared": ("Shared",),
    "landing_page": ("Landing Page",),
    "report_prompts": ("Report Prompts", "Prompts"),
    "area_where_used": ("Area Where Used",),
    # Optional: the column that states each row's fields directly.
    "fields": ("Fields",),
}

# The only hard requirement: we cannot identify a report without its name.
REQUIRED = ("report_name",)

HEADER_ROW = 2  # 0-based: the real header is the third physical row.
_FIRST_DATA_ROW = HEADER_ROW + 2  # 1-based physical row of the first data record.


class ReportCatalogLoader:
    """Loads report-level metadata from the catalog workbook."""

    def __init__(self, header_row: int = HEADER_ROW):
        self.header_row = header_row
        self.errors: list[RowError] = []
        self.rows_read: int = 0

    def load(self, path: Path | str) -> list[ReportCatalogRecord]:
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(
                f"Report catalog not found at {path}\n"
                "Pass --catalog /path/to/catalog.xlsx"
            )

        frame = pd.read_excel(path, header=self.header_row)
        frame.columns = [str(c).strip() for c in frame.columns]
        cols = resolve_columns(list(frame.columns), COLUMN_ALIASES)

        missing = [name for name in REQUIRED if name not in cols]
        if missing:
            raise ValueError(
                f"Report catalog is missing required column(s): {missing}\n"
                f"Columns found (header row {self.header_row + 1}): {list(frame.columns)}\n"
                "The loader expects the real header on the THIRD physical row."
            )

        self.errors = []
        self.rows_read = len(frame)
        source_file = path.name

        def get(row, logical: str) -> str:
            col = cols.get(logical)
            return normalize_display(row[col]) if col else ""

        def get_raw(row, logical):
            col = cols.get(logical)
            return row[col] if col else None

        records: list[ReportCatalogRecord] = []
        for position, (_, row) in enumerate(frame.iterrows()):
            physical_row = _FIRST_DATA_ROW + position
            source = SourceRef(source_file=source_file, source_row=physical_row)

            name = get(row, "report_name")
            if not name:
                # Blank spacer/footer rows are expected in this export; they are
                # skipped quietly rather than reported as data errors.
                continue

            fields_col = cols.get("fields")
            records.append(
                ReportCatalogRecord(
                    report_key=f"R{physical_row:04d}",
                    title_key=normalize_match(name),
                    report_name=name,
                    source=source,
                    row_index=len(records),
                    data_source_key=normalize_match(get(row, "data_source")),
                    report_tags=get(row, "report_tags"),
                    category=get(row, "category"),
                    worklet_landing_pages=get(row, "worklet_landing_pages"),
                    owner=get(row, "owner"),
                    created_by=get(row, "created_by"),
                    description=get(row, "description"),
                    created_date=get_raw(row, "created_date"),
                    available_usage=get(row, "available_usage"),
                    number_of_times=get_raw(row, "number_of_times"),
                    last_updated_date=get_raw(row, "last_updated_date"),
                    last_run_date=get_raw(row, "last_run_date"),
                    last_run_by=get(row, "last_run_by"),
                    data_source=get(row, "data_source"),
                    report_type=get(row, "report_type"),
                    worklet=get(row, "worklet"),
                    chart_type=get(row, "chart_type"),
                    shared=get(row, "shared"),
                    landing_page=get(row, "landing_page"),
                    report_prompts=get(row, "report_prompts"),
                    area_where_used=get(row, "area_where_used"),
                    fields_raw=(
                        normalize_display(row[fields_col]) if fields_col else None
                    ),
                )
            )
        keys = [record.report_key for record in records]
        duplicates = sorted({key for key in keys if keys.count(key) > 1})
        if duplicates:
            raise ValueError(f"Duplicate catalog report_key values: {duplicates[:10]}")
        return records
