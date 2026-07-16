"""Builders for small, hand-controlled Phase 2 fixture workbooks.

These are intentionally tiny and explicit: every linking edge case in the tests
(duplicate titles, commas in names, versioned reports, conflicting metadata) is
visible in the fixture rather than buried in 4000 synthetic rows.

Both writers reproduce the real files' quirk: two banner rows above the header, so
the real header lands on the third physical row.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

CATALOG_COLUMNS = [
    "Custom Report", "Report Tag(s)", "Category", "Worklet Landing Pages", "Owner",
    "Created by", "Description", "Created Date", "Available Usage", "Number of Times",
    "Last Updated Date", "Last Run Date", "Last Run By", "Data Source", "Report Type",
    "Worklet", "Chart Type", "Shared", "Landing Page", "Report Prompts",
    "Area Where Used",
]

DICTIONARY_COLUMNS = [
    "Business Object", "Field Name", "Description", "Report Field Type",
    "Related Business Object Name", "Built-in Prompts", "Domain", "Categories",
    "Authorized Usage", "Where_Used",
]


def _write_with_banner(frame: pd.DataFrame, path: Path, banner: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    n_cols = len(frame.columns)
    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        pd.DataFrame([[banner] + [None] * (n_cols - 1)]).to_excel(
            writer, index=False, header=False, startrow=0, sheet_name="Sheet1"
        )
        pd.DataFrame([["End Date", "2026-06-30"] + [None] * (n_cols - 2)]).to_excel(
            writer, index=False, header=False, startrow=1, sheet_name="Sheet1"
        )
        frame.to_excel(
            writer, index=False, header=True, startrow=2, sheet_name="Sheet1"
        )


def catalog_row(name: str, **overrides) -> dict:
    row = {col: "" for col in CATALOG_COLUMNS}
    row["Custom Report"] = name
    row["Category"] = "Worker Data"
    row["Data Source"] = "All Workers"
    row["Report Type"] = "Advanced"
    row["Number of Times"] = 1
    row["Shared"] = "Yes"
    row["Last Updated Date"] = pd.Timestamp("2025-01-01")
    row["Last Run Date"] = pd.Timestamp("2025-06-01")
    row.update(overrides)
    return row


def dictionary_row(business_object: str, field_name: str, where_used: str, **overrides) -> dict:
    row = {col: "" for col in DICTIONARY_COLUMNS}
    row["Business Object"] = business_object
    row["Field Name"] = field_name
    row["Where_Used"] = where_used
    row["Description"] = f"The {field_name.lower()} of the record."
    row["Report Field Type"] = "Text"
    row["Domain"] = "Worker Data"
    row["Categories"] = "General Reporting"
    row["Authorized Usage"] = "Default Areas"
    row.update(overrides)
    return row


def write_catalog(rows: list[dict], path: Path) -> Path:
    frame = pd.DataFrame(rows, columns=CATALOG_COLUMNS)
    _write_with_banner(frame, path, "All Custom Reports — Synthetic Estate")
    return path


def write_dictionary(rows: list[dict], path: Path) -> Path:
    frame = pd.DataFrame(rows, columns=DICTIONARY_COLUMNS)
    _write_with_banner(frame, path, "Workday Data Dictionary — Synthetic Phase 2")
    return path


def write_legacy(rows: list[dict], path: Path) -> Path:
    """Phase 1 single-file equivalent: catalog columns plus a Fields column."""
    columns = CATALOG_COLUMNS + ["Fields"]
    frame = pd.DataFrame(rows, columns=columns)
    _write_with_banner(frame, path, "All Custom Reports — Synthetic Estate")
    return path
