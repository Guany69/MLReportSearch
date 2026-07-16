"""Enrichment: turn catalog records + linked fields into the ML-facing corpus.

This is the seam of Phase 2. Both ingestion modes converge here on one contract --
the same `ReportCorpus` the Phase 1 pipeline already consumes -- so no mode
conditional ever reaches the ranking code.

The contract downstream depends on (`represent.py`, `model.py`, `render.py`,
`app.py`) is that `frame["fields"]` is a `list[str]` of field names. That column
means exactly what it meant in Phase 1. Phase 2 metadata is *additive*: it arrives
in new columns that are empty in legacy mode.
"""

from __future__ import annotations

import pandas as pd

from ..data import ReportCorpus, collapse_families
from .models import EnrichedField, ImportSummary, ReportCatalogRecord


def _records_to_frame(records: list[ReportCatalogRecord]) -> pd.DataFrame:
    """Rebuild the raw, Phase-1-shaped frame that `collapse_families` expects.

    Reusing the existing collapse logic (rather than reimplementing it) is what
    keeps family semantics identical across modes.
    """
    return pd.DataFrame(
        {
            "Custom Report": [r.report_name for r in records],
            "Fields": [r.fields_raw or "" for r in records],
            "Report Prompts": [r.report_prompts for r in records],
            "Category": [r.category for r in records],
            "Data Source": [r.data_source for r in records],
            "Report Type": [r.report_type for r in records],
            "Report Tag(s)": [r.report_tags for r in records],
            "Number of Times": [r.number_of_times for r in records],
            "Last Run Date": [r.last_run_date for r in records],
            "Last Updated Date": [r.last_updated_date for r in records],
            "Shared": [r.shared for r in records],
        }
    )


def build_enriched_frame(
    records: list[ReportCatalogRecord],
    linked_fields: dict[int, list[EnrichedField]],
    summary: ImportSummary,
) -> ReportCorpus:
    """Attach reconstructed fields to catalog rows, then collapse into families.

    Fields are written into the raw frame's `Fields` column *before* collapsing, so
    family identity (title + field set) is computed on the reconstructed fields --
    exactly as Phase 1 computes it on the authored ones.
    """
    frame = _records_to_frame(records)

    # Reconstructed field names, in the linker's deterministic order.
    frame["Fields"] = [
        "; ".join(f.field_name for f in linked_fields.get(i, []))
        for i in range(len(records))
    ]

    collapsed = collapse_families(frame)

    # Map each surviving family back to its representative row so the Phase 2
    # metadata travels with it. collapse_families picks the most-run copy, so we
    # re-derive the mapping by family key rather than assuming positions.
    frame["_row_index"] = range(len(records))
    frame["_fields_list"] = [
        [f.field_name for f in linked_fields.get(i, [])] for i in range(len(records))
    ]
    from ..data import _family_key  # local import: internal helper, avoids cycle

    frame["_family_key"] = [
        _family_key(str(t), f)
        for t, f in zip(frame["Custom Report"], frame["_fields_list"])
    ]
    # Representative = first row of each family in collapse order; collapse_families
    # sorts by runs desc, so mirror that selection here.
    runs = pd.to_numeric(frame["Number of Times"], errors="coerce")
    order = frame.assign(_runs=runs).sort_values(
        by="_runs", ascending=False, na_position="last", kind="mergesort"
    )
    rep_row = order.groupby("_family_key", sort=False)["_row_index"].first()

    rep_indices = [rep_row[key] for key in collapsed["family_key"]]

    field_meta: list[list[EnrichedField]] = [
        linked_fields.get(int(i), []) for i in rep_indices
    ]
    collapsed["field_meta"] = field_meta
    collapsed["has_ambiguous_fields"] = [
        any(f.is_ambiguous for f in metas) for metas in field_meta
    ]
    collapsed["ambiguous_field_count"] = [
        sum(1 for f in metas if f.is_ambiguous) for metas in field_meta
    ]
    collapsed["source_row"] = [records[int(i)].source.source_row for i in rep_indices]

    summary.families_after_collapse = len(collapsed)
    return ReportCorpus(frame=collapsed, raw_row_count=len(records))


def attach_empty_enrichment(corpus: ReportCorpus) -> ReportCorpus:
    """Give a legacy corpus the Phase 2 columns, empty.

    Downstream code can then read the same columns in both modes without asking
    which mode produced them.
    """
    n = len(corpus.frame)
    corpus.frame["field_meta"] = [[] for _ in range(n)]
    corpus.frame["has_ambiguous_fields"] = [False] * n
    corpus.frame["ambiguous_field_count"] = [0] * n
    if "source_row" not in corpus.frame.columns:
        corpus.frame["source_row"] = [0] * n
    return corpus
