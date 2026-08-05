"""Ingestion package: the report workbook in, one ML-facing corpus out.

`build_corpus(cfg)` is the only entry point the rest of the application needs.

A dual-file mode once lived here, reading a report catalog plus a separate field
dictionary and reconstructing report->field links by inverting the dictionary's
`Where_Used` column. It has been removed along with its workbooks; the single
file states each row's fields directly, so no link reconstruction is needed. The
*shape* that mode required -- row-level instances collapsed into title families
-- is unchanged and is now the architecture, so `corpus_granularity` still
matters even though there is one ingest mode left.
"""

from __future__ import annotations

import time

from ..data import load_corpus as load_legacy_corpus
from .catalog import ReportCatalogLoader
from .enrich import attach_empty_enrichment, build_row_level_frame
from .legacy_rows import fields_from_column
from .models import (
    EnrichedField,
    ImportSummary,
    IngestMode,
    ReportCatalogRecord,
)

__all__ = [
    "build_corpus",
    "fields_from_column",
    "ReportCatalogLoader",
    "IngestMode",
    "EnrichedField",
    "ImportSummary",
    "ReportCatalogRecord",
]


def build_corpus(cfg, verbose: bool = False):
    """Assemble the corpus. Returns (ReportCorpus, ImportSummary)."""
    started = time.perf_counter()
    # Still validated rather than assumed: a config naming the removed dual-file
    # mode must fail here and say so, not fall through to reading the wrong file.
    mode = IngestMode(cfg.ingest_mode)

    if cfg.corpus_granularity == "family":
        corpus = attach_empty_enrichment(load_legacy_corpus(cfg.data_path))
        summary = ImportSummary(
            mode=mode.value,
            report_rows_read=corpus.raw_row_count,
            valid_reports_loaded=corpus.raw_row_count,
            families_after_collapse=len(corpus.frame),
            import_duration_s=time.perf_counter() - started,
        )
        if verbose:
            print(summary.render())
        return corpus, summary

    # Row-level import. The workbook's own `Fields` column already states each
    # row's fields, so the corpus reaches the two-level family/instance shape
    # without any link reconstruction.
    catalog_loader = ReportCatalogLoader()
    records = catalog_loader.load(cfg.data_path)
    linked = fields_from_column(records, source_file=str(cfg.data_path))

    summary = ImportSummary(mode=mode.value)
    corpus = build_row_level_frame(records, linked, summary)

    summary.report_rows_read = catalog_loader.rows_read
    summary.valid_reports_loaded = len(records)
    summary.row_validation_errors = len(catalog_loader.errors)
    duplicate_keys = _count_duplicate_identities(records)
    summary.duplicate_report_identities = duplicate_keys[0]
    summary.duplicate_report_rows = duplicate_keys[1]
    summary.import_duration_s = time.perf_counter() - started

    if verbose:
        print(summary.render())
    return corpus, summary


def _count_duplicate_identities(records: list[ReportCatalogRecord]) -> tuple[int, int]:
    """How many report identities repeat, and over how many rows."""
    import collections

    counts = collections.Counter(r.title_key for r in records)
    dupes = {k: v for k, v in counts.items() if v > 1}
    return len(dupes), sum(dupes.values())
