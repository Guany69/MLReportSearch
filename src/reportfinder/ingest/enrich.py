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
    frame = pd.DataFrame(
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
    # Preserve the complete high-signal catalog record. These columns survive
    # family collapse below by being copied from its chosen representative.
    extra = {
        "report_id": [r.row_index for r in records],
        "description": [r.description for r in records],
        "area_where_used": [r.area_where_used for r in records],
        "landing_page": [r.landing_page for r in records],
        "chart_type": [r.chart_type for r in records],
        "owner": [r.owner for r in records],
        "created_by": [r.created_by for r in records],
        "created_date": [r.created_date for r in records],
        "available_usage": [r.available_usage for r in records],
        "last_run_by": [r.last_run_by for r in records],
        "worklet": [r.worklet for r in records],
        "worklet_landing_pages": [r.worklet_landing_pages for r in records],
    }
    for name, values in extra.items():
        frame[name] = values
    return frame


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
        for t, f in zip(frame["Custom Report"], frame["_fields_list"], strict=True)
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
    metadata = [
        "report_id", "description", "area_where_used", "landing_page",
        "chart_type", "owner", "created_by", "created_date",
        "available_usage", "last_run_by", "worklet", "worklet_landing_pages",
    ]
    for name in metadata:
        collapsed[name] = [frame.iloc[int(i)][name] for i in rep_indices]
    collapsed["field_link_confidence"] = [
        (sum(f.match_confidence for f in metas) / len(metas)) if metas else 0.0
        for metas in field_meta
    ]
    collapsed["ambiguous_link_fraction"] = [
        (sum(f.is_ambiguous for f in metas) / len(metas)) if metas else 0.0
        for metas in field_meta
    ]

    summary.families_after_collapse = len(collapsed)
    return ReportCorpus(frame=collapsed, raw_row_count=len(records))


def build_row_level_frame(
    records: list[ReportCatalogRecord],
    linked_fields: dict[int, list[EnrichedField]],
    summary: ImportSummary,
) -> ReportCorpus:
    """Build the retrieval corpus with one independently addressable catalog row."""
    raw = _records_to_frame(records)
    fields = [[f.field_name for f in linked_fields.get(i, [])] for i in range(len(records))]
    raw["Fields"] = ["; ".join(values) for values in fields]
    from ..data import _family_key, split_list_cell

    family_keys = [_family_key(r.report_name, fields[i]) for i, r in enumerate(records)]
    family_sizes = pd.Series(family_keys).value_counts().to_dict()
    runs = pd.to_numeric(raw["Number of Times"], errors="coerce")
    family_rank = (
        pd.DataFrame({"family": family_keys, "runs": runs, "position": range(len(records))})
        .sort_values(["family", "runs", "position"], ascending=[True, False, True],
                     na_position="last", kind="mergesort")
        .groupby("family", sort=False).cumcount().add(1)
    )
    ranks_by_position = dict(zip(
        pd.DataFrame({"family": family_keys, "runs": runs, "position": range(len(records))})
        .sort_values(["family", "runs", "position"], ascending=[True, False, True],
                     na_position="last", kind="mergesort")["position"],
        family_rank,
        strict=True,
    ))
    out = pd.DataFrame({
        "report_key": [r.report_key for r in records],
        "title_key": [r.title_key for r in records],
        "family_key": family_keys,
        "family_size": [family_sizes[key] for key in family_keys],
        "family_rank": [int(ranks_by_position[i]) for i in range(len(records))],
        "source_row": [r.source.source_row for r in records],
        "report_id": [r.row_index for r in records],
        "title": [r.report_name for r in records],
        "fields": fields,
        "prompts": [split_list_cell(r.report_prompts) for r in records],
        "category": [r.category for r in records],
        "data_source": [r.data_source for r in records],
        "report_type": [r.report_type for r in records],
        "tags": [r.report_tags for r in records],
        "shared": [r.shared for r in records],
        "runs": pd.to_numeric(raw["Number of Times"], errors="coerce").astype("Int64"),
        "last_run": pd.to_datetime(raw["Last Run Date"], errors="coerce"),
        "last_updated": pd.to_datetime(raw["Last Updated Date"], errors="coerce"),
    })
    out["runs_total"] = out.groupby("family_key")["runs"].transform("sum")
    out["last_run_missing"] = out["last_run"].isna()
    out["last_updated_missing"] = out["last_updated"].isna()
    out["runs_missing"] = out["runs"].isna()
    out["field_meta"] = [linked_fields.get(i, []) for i in range(len(records))]
    out["has_ambiguous_fields"] = [any(f.is_ambiguous for f in values) for values in out["field_meta"]]
    out["ambiguous_field_count"] = [sum(f.is_ambiguous for f in values) for values in out["field_meta"]]
    out["field_link_confidence"] = [sum(f.match_confidence for f in values) / len(values) if values else 0.0 for values in out["field_meta"]]
    out["ambiguous_link_fraction"] = [sum(f.is_ambiguous for f in values) / len(values) if values else 0.0 for values in out["field_meta"]]
    for name in ("description", "area_where_used", "landing_page", "chart_type", "owner",
                 "created_by", "created_date", "available_usage", "last_run_by", "worklet",
                 "worklet_landing_pages"):
        out[name] = [getattr(r, name) for r in records]
    summary.families_after_collapse = len(set(family_keys))
    return ReportCorpus(frame=out.reset_index(drop=True), raw_row_count=len(records))


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
    defaults = {
        "report_id": list(range(n)), "description": [""] * n,
        "area_where_used": [""] * n, "landing_page": [""] * n,
        "chart_type": [""] * n, "owner": [""] * n,
        "created_by": [""] * n, "created_date": [pd.NaT] * n,
        "available_usage": [""] * n, "last_run_by": [""] * n,
        "worklet": [""] * n, "worklet_landing_pages": [""] * n,
        "field_link_confidence": [0.0] * n,
        "ambiguous_link_fraction": [0.0] * n,
    }
    for name, values in defaults.items():
        if name not in corpus.frame:
            corpus.frame[name] = values
    return corpus
