"""Field dictionary loader (Phase 2 supplemental workbook).

Each row describes one field and lists, in `Where_Used`, the reports that use it.
Rows may repeat for the same logical field; those are merged here -- compatibly,
without picking an arbitrary winner when two rows disagree.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from .models import (
    FieldDictionaryRecord,
    MetadataConflict,
    RowError,
    SourceRef,
)
from .normalize import normalize_display, normalize_match, resolve_columns, split_multi

COLUMN_ALIASES: dict[str, tuple[str, ...]] = {
    "business_object": ("Business Object", "BusinessObject"),
    "field_name": ("Field Name", "FieldName", "Field"),
    "description": ("Description",),
    "report_field_type": ("Report Field Type", "Field Type"),
    "related_business_object_name": (
        "Related Business Object Name",
        "Related Business Object",
    ),
    "built_in_prompts": ("Built-in Prompts", "Built in Prompts", "Builtin Prompts"),
    "domain": ("Domain",),
    "categories": ("Categories", "Category"),
    "authorized_usage": ("Authorized Usage",),
    "where_used": ("Where_Used", "Where Used"),
}

# Without these three there is no field identity and no link to reconstruct.
REQUIRED = ("business_object", "field_name", "where_used")

# Merged compatibly across duplicate rows; disagreement is recorded, not resolved.
_MERGEABLE = (
    "description",
    "report_field_type",
    "related_business_object_name",
    "built_in_prompts",
    "domain",
    "categories",
    "authorized_usage",
)

HEADER_ROW = 2
_FIRST_DATA_ROW = HEADER_ROW + 2


def make_field_key(business_object: str, field_name: str) -> str:
    """Deterministic field identity.

    Business object is part of the key on purpose: `Worker|Status` and
    `Job Requisition|Status` are different fields that merely share a display name.
    """
    return f"{normalize_match(business_object)}|{normalize_match(field_name)}"


class FieldDictionaryLoader:
    """Loads and merges field-level metadata."""

    def __init__(self, header_row: int = HEADER_ROW):
        self.header_row = header_row
        self.errors: list[RowError] = []
        self.rows_read: int = 0
        self.valid_rows: int = 0
        self.conflicts: list[MetadataConflict] = []

    def load(self, path: Path | str) -> list[FieldDictionaryRecord]:
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(
                f"Field dictionary not found at {path}\n"
                "Pass --field-dictionary /path/to/dictionary.xlsx"
            )

        frame = pd.read_excel(path, header=self.header_row)
        frame.columns = [str(c).strip() for c in frame.columns]
        cols = resolve_columns(list(frame.columns), COLUMN_ALIASES)

        missing = [name for name in REQUIRED if name not in cols]
        if missing:
            raise ValueError(
                f"Field dictionary is missing required column(s): {missing}\n"
                f"Columns found (header row {self.header_row + 1}): {list(frame.columns)}\n"
                "Required: Business Object, Field Name, Where_Used."
            )

        self.errors = []
        self.conflicts = []
        self.rows_read = len(frame)
        self.valid_rows = 0
        source_file = path.name

        def get(row, logical: str) -> str:
            col = cols.get(logical)
            return normalize_display(row[col]) if col else ""

        merged: dict[str, FieldDictionaryRecord] = {}

        for position, (_, row) in enumerate(frame.iterrows()):
            physical_row = _FIRST_DATA_ROW + position
            source = SourceRef(source_file=source_file, source_row=physical_row)

            business_object = get(row, "business_object")
            field_name = get(row, "field_name")

            # A row without identity cannot be merged or linked. Record and skip --
            # one malformed row must not abort the whole import.
            if not business_object or not field_name:
                if business_object or field_name or get(row, "where_used"):
                    self.errors.append(
                        RowError(
                            source=source,
                            reason=(
                                "missing Business Object"
                                if not business_object
                                else "missing Field Name"
                            ),
                        )
                    )
                continue

            self.valid_rows += 1
            key = make_field_key(business_object, field_name)
            where_used_raw = get(row, "where_used")
            names = split_multi(where_used_raw)

            existing = merged.get(key)
            if existing is None:
                merged[key] = FieldDictionaryRecord(
                    field_key=key,
                    business_object=business_object,
                    field_name=field_name,
                    sources=[source],
                    where_used_raw=where_used_raw,
                    where_used_report_names=names,
                    **{attr: get(row, attr) for attr in _MERGEABLE},
                )
                continue

            # Repeated row for a known field: merge, keep all provenance.
            existing.sources.append(source)
            self._merge_metadata(existing, row, get, source)
            self._merge_where_used(existing, names, where_used_raw)

        for record in merged.values():
            self.conflicts.extend(record.conflicts)
        return list(merged.values())

    def _merge_metadata(self, existing, row, get, source: SourceRef) -> None:
        """Fill blanks from the new row; record disagreements without overwriting."""
        for attr in _MERGEABLE:
            incoming = get(row, attr)
            if not incoming:
                continue
            current = getattr(existing, attr)
            if not current:
                setattr(existing, attr, incoming)
            elif normalize_match(current) != normalize_match(incoming):
                # Two rows assert different non-empty values. We are not entitled to
                # pick one, so the first is kept and the clash is surfaced.
                existing.conflicts.append(
                    MetadataConflict(
                        field_key=existing.field_key,
                        attribute=attr,
                        existing=current,
                        incoming=incoming,
                        source=source,
                    )
                )

    @staticmethod
    def _merge_where_used(existing, names: list[str], raw: str) -> None:
        """Union the report names, preserving first-appearance order."""
        seen = {normalize_match(n) for n in existing.where_used_report_names}
        for name in names:
            key = normalize_match(name)
            if key not in seen:
                seen.add(key)
                existing.where_used_report_names.append(name)
        if raw and raw not in existing.where_used_raw:
            existing.where_used_raw = f"{existing.where_used_raw}; {raw}".strip("; ")
