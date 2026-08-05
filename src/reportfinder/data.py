"""Load and parse the custom-reports workbook.

The workbook is quirky: the real header sits on the third physical row, beneath a
banner row and an "End Date" row. Everything downstream depends on parsing it the
same way every time, so that assumption is asserted here rather than guessed at.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

# Canonical column names as they appear in the header row.
COL_TITLE = "Custom Report"
COL_FIELDS = "Fields"
COL_PROMPTS = "Report Prompts"
COL_CATEGORY = "Category"
COL_DATA_SOURCE = "Data Source"
COL_REPORT_TYPE = "Report Type"
COL_TAGS = "Report Tag(s)"
COL_RUNS = "Number of Times"
COL_LAST_RUN = "Last Run Date"
COL_LAST_UPDATED = "Last Updated Date"
COL_SHARED = "Shared"

# Legacy (Phase 1) requires Fields: it is the sole source of the field signal in a
# single-file import. The Phase 2 catalog has no Fields column -- that path goes
# through `ingest/` instead, which reconstructs fields from the field dictionary.
REQUIRED_COLUMNS = [COL_TITLE, COL_FIELDS]

# Description is deliberately excluded: only 7 distinct values across 4000 rows,
# so it carries no discriminating signal for matching.
IGNORED_FOR_MATCHING = ("Description",)


def split_list_cell(value) -> list[str]:
    """Split a ';'-separated cell into a clean list of items.

    Empty/NaN cells yield []. Whitespace is stripped and empty fragments dropped,
    so "A; ; B;" -> ["A", "B"].
    """
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return []
    text = str(value).strip()
    if not text or text.lower() in {"nan", "none"}:
        return []
    return [part.strip() for part in text.split(";") if part.strip()]


def _coerce_runs(series: pd.Series) -> pd.Series:
    """Run counts to a nullable integer. Non-numeric junk becomes NA, not 0."""
    return pd.to_numeric(series, errors="coerce").astype("Int64")


def _coerce_date(series: pd.Series) -> pd.Series:
    """Dates to datetime; unparseable values become NaT (never zero-filled)."""
    return pd.to_datetime(series, errors="coerce")


@dataclass
class ReportCorpus:
    """The parsed, family-collapsed report corpus.

    Attributes:
        frame: One row per report *family* (exact-duplicate definitions merged).
        raw_row_count: Number of data rows read from the workbook before collapsing.
    """

    frame: pd.DataFrame
    raw_row_count: int

    def __len__(self) -> int:
        return len(self.frame)


def load_raw(path: Path) -> pd.DataFrame:
    """Read the workbook with the header on the third physical row."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Report workbook not found at {path}\n"
            "Place the .xlsx there, or pass --data /path/to/file.xlsx"
        )

    frame = pd.read_excel(path, header=2)
    frame.columns = [str(c).strip() for c in frame.columns]

    missing = [c for c in REQUIRED_COLUMNS if c not in frame.columns]
    if missing:
        hint = ""
        if missing == [COL_FIELDS] and COL_TITLE in frame.columns:
            # Almost certainly a Phase 2 catalog fed to the legacy loader.
            hint = (
                "\nThis looks like a Phase 2 report catalog (no Fields column).\n"
                "Use Phase 2 mode, which reconstructs fields from the field dictionary:\n"
                "  python -m reportfinder --mode phase2_dual_file \"your query\""
            )
        raise ValueError(
            f"Workbook is missing required column(s): {missing}\n"
            f"Columns found (header=2): {list(frame.columns)}\n"
            "The parser expects the real header on the THIRD physical row." + hint
        )

    # Drop rows with no title: fully blank spacer rows and any trailing footer.
    frame = frame[frame[COL_TITLE].notna()]
    frame = frame[frame[COL_TITLE].astype(str).str.strip() != ""]
    return frame.reset_index(drop=True)


def _family_key(title: str, fields: list[str]) -> str:
    """Identity of a report definition: same title + same field *set*.

    Field order is not meaningful for identity, so the set is sorted and
    case-folded before hashing into the key.
    """
    normalized_fields = sorted({f.casefold() for f in fields})
    return f"{title.strip().casefold()}||{'|'.join(normalized_fields)}"


def collapse_families(frame: pd.DataFrame) -> pd.DataFrame:
    """Collapse exact-duplicate report definitions into one representative row.

    Near-identical reports would otherwise flood the top-k with the same answer
    repeated. The representative is the family member with the highest run count
    (ties broken by most-recent run), since that's the copy people actually use.
    Aggregate usage across the family is preserved separately.
    """
    work = frame.copy()
    work["_fields_list"] = work[COL_FIELDS].map(split_list_cell)
    work["_prompts_list"] = (
        work[COL_PROMPTS].map(split_list_cell)
        if COL_PROMPTS in work.columns
        else [[] for _ in range(len(work))]
    )
    work["_family_key"] = [
        _family_key(str(t), f)
        for t, f in zip(work[COL_TITLE], work["_fields_list"], strict=True)
    ]

    runs = (
        _coerce_runs(work[COL_RUNS])
        if COL_RUNS in work.columns
        else pd.Series([pd.NA] * len(work), dtype="Int64")
    )
    last_run = (
        _coerce_date(work[COL_LAST_RUN])
        if COL_LAST_RUN in work.columns
        else pd.Series([pd.NaT] * len(work))
    )
    last_updated = (
        _coerce_date(work[COL_LAST_UPDATED])
        if COL_LAST_UPDATED in work.columns
        else pd.Series([pd.NaT] * len(work))
    )
    work["_runs"] = runs
    work["_last_run"] = last_run
    work["_last_updated"] = last_updated

    # Representative selection: most-run copy, then most-recently-run.
    # NA run counts sort last rather than being treated as zero.
    order = work.sort_values(
        by=["_runs", "_last_run"],
        ascending=[False, False],
        na_position="last",
        kind="mergesort",
    )
    representatives = order.groupby("_family_key", sort=False).head(1)

    stats = work.groupby("_family_key", sort=False).agg(
        family_size=("_family_key", "size"),
        runs_total=("_runs", "sum"),
        last_run_any=("_last_run", "max"),
        last_updated_any=("_last_updated", "max"),
    )

    merged = representatives.merge(
        stats, left_on="_family_key", right_index=True, how="left"
    )

    out = pd.DataFrame(
        {
            "family_key": merged["_family_key"].values,
            "title": merged[COL_TITLE].astype(str).str.strip().values,
            "fields": merged["_fields_list"].values,
            "prompts": merged["_prompts_list"].values,
            "family_size": merged["family_size"].astype(int).values,
        }
    )

    for col, name in [
        (COL_CATEGORY, "category"),
        (COL_DATA_SOURCE, "data_source"),
        (COL_REPORT_TYPE, "report_type"),
        (COL_TAGS, "tags"),
        (COL_SHARED, "shared"),
    ]:
        if col in merged.columns:
            values = merged[col]
            out[name] = [
                "" if pd.isna(v) else str(v).strip() for v in values
            ]
        else:
            out[name] = ""

    # Usage/recency travel with the report for display. Nulls stay null and are
    # flagged explicitly -- a report that has never run is not a report that ran
    # zero times on the epoch.
    out["runs"] = merged["_runs"].values
    out["runs_total"] = merged["runs_total"].values
    out["last_run"] = merged["_last_run"].values
    out["last_updated"] = merged["_last_updated"].values
    out["last_run_missing"] = merged["_last_run"].isna().values
    out["last_updated_missing"] = merged["_last_updated"].isna().values
    out["runs_missing"] = merged["_runs"].isna().values

    return out.reset_index(drop=True)


def load_corpus(path: Path) -> ReportCorpus:
    """Read the workbook and collapse it to one row per report family."""
    raw = load_raw(path)
    collapsed = collapse_families(raw)
    return ReportCorpus(frame=collapsed, raw_row_count=len(raw))
