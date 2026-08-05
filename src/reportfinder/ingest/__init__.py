"""Ingestion package: spreadsheets in, one ML-facing corpus out.

`build_corpus(cfg)` is the only entry point the rest of the application needs. It
dispatches on ingest mode and returns the same `ReportCorpus` either way, so the
ranking pipeline never learns which spreadsheets produced it.
"""

from __future__ import annotations

import time

from ..data import load_corpus as load_legacy_corpus
from .catalog import ReportCatalogLoader
from .enrich import attach_empty_enrichment, build_enriched_frame, build_row_level_frame
from .field_dictionary import FieldDictionaryLoader
from .legacy_rows import fields_from_column
from .linker import ReportFieldLinker
from .models import (
    AmbiguityPolicy,
    AmbiguityStatus,
    EnrichedField,
    FieldDictionaryRecord,
    ImportSummary,
    IngestMode,
    LinkResult,
    MatchMethod,
    ReportCatalogRecord,
    ReportFieldLink,
)

__all__ = [
    "build_corpus",
    "fields_from_column",
    "ReportCatalogLoader",
    "FieldDictionaryLoader",
    "ReportFieldLinker",
    "IngestMode",
    "AmbiguityPolicy",
    "AmbiguityStatus",
    "MatchMethod",
    "EnrichedField",
    "ImportSummary",
    "LinkResult",
    "ReportCatalogRecord",
    "FieldDictionaryRecord",
    "ReportFieldLink",
]


def build_corpus(cfg, verbose: bool = False):
    """Assemble the corpus for the configured mode.

    Returns (ReportCorpus, ImportSummary). Both modes produce an identical corpus
    contract; only the summary reveals how it was built.
    """
    started = time.perf_counter()
    mode = IngestMode(cfg.ingest_mode)

    if mode is IngestMode.LEGACY_SINGLE_FILE:
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

        # Row-level legacy import. The workbook's own `Fields` column already
        # states each row's fields, so no linker is needed -- but the resulting
        # corpus has the same instance-level shape Phase 2 produces, which is what
        # the two-level family/instance model requires.
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

    # -- Phase 2 --------------------------------------------------------
    catalog_loader = ReportCatalogLoader()
    records = catalog_loader.load(cfg.catalog_path)

    dictionary_loader = FieldDictionaryLoader()
    field_records = dictionary_loader.load(cfg.field_dictionary_path)

    linker = ReportFieldLinker(
        policy=AmbiguityPolicy(cfg.ambiguity_policy),
        enable_composite=cfg.enable_composite_match,
        enable_fuzzy=cfg.enable_fuzzy_match,
        fuzzy_threshold=cfg.fuzzy_threshold,
    )
    result = linker.link(records, field_records)

    builder = build_row_level_frame if cfg.corpus_granularity == "report_row" else build_enriched_frame
    corpus = builder(records, result.linked_reports, result.validation_metrics)

    summary = result.validation_metrics
    summary.mode = mode.value
    summary.report_rows_read = catalog_loader.rows_read
    summary.valid_reports_loaded = len(records)
    summary.field_rows_read = dictionary_loader.rows_read
    summary.valid_field_rows_loaded = dictionary_loader.valid_rows
    summary.unique_field_identities = len(field_records)
    summary.metadata_conflicts = len(dictionary_loader.conflicts)
    summary.row_validation_errors = len(dictionary_loader.errors) + len(
        catalog_loader.errors
    )

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
