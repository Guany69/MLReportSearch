"""Row-level ingestion for the legacy single-file workbook.

The two-level Report Family / Report Instance model needs one addressable row per
catalog row. Legacy mode used to collapse to families before anything downstream
could see the instances, so this file pins the row-level contract:

1. One instance per catalog row, with a stable `report_key`.
2. Fields come from the workbook's own `Fields` column, deduplicated.
3. `corpus_granularity="family"` still collapses, unchanged.
4. Every column the renderers index off `Candidate.row` survives.

The last test is guarded on the real workbook because it checks the identity rule
the relevance dataset depends on -- a synthetic fixture cannot prove that.

Run: uv run pytest tests/test_legacy_rows.py -v
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from reportfinder.config import DEFAULT
from reportfinder.ingest import build_corpus
from reportfinder.ingest.catalog import ReportCatalogLoader
from reportfinder.ingest.legacy_rows import fields_from_column
from reportfinder.ingest.models import AmbiguityStatus, MatchMethod

from .ingest_fixtures import catalog_row, write_legacy

REAL_LEGACY = DEFAULT.data_path
SEED_JUDGMENTS = Path("data/relevance/raw/axis_report_search_seed_judgments.jsonl")

requires_real_legacy = pytest.mark.skipif(
    not REAL_LEGACY.exists(), reason="real legacy workbook not present in data/"
)
requires_relevance_bundle = pytest.mark.skipif(
    not (REAL_LEGACY.exists() and SEED_JUDGMENTS.exists()),
    reason="legacy workbook or relevance seed bundle not present in data/",
)


def _legacy_workbook(tmp_path: Path) -> Path:
    """Two reports sharing a title, one distinct -- so families are non-trivial."""
    rows = [
        catalog_row(
            "Headcount by Organization",
            Fields="Worker ID; Supervisory Organization; Headcount",
            **{"Number of Times": 40},
        ),
        catalog_row(
            "Headcount by Organization",
            Fields="Worker ID; Company; Headcount",
            **{"Number of Times": 5},
        ),
        catalog_row(
            "Termination Detail",
            # Duplicate field name: the repetition carries no retrieval signal.
            Fields="Worker ID; Termination Reason; worker id",
            **{"Number of Times": 12},
        ),
    ]
    return write_legacy(rows, tmp_path / "legacy.xlsx")


def _row_level(tmp_path: Path):
    cfg = DEFAULT.with_overrides(
        ingest_mode="legacy_single_file",
        corpus_granularity="report_row",
        data_path=_legacy_workbook(tmp_path),
    )
    return build_corpus(cfg)


# --- instance identity ------------------------------------------------------


def test_row_level_keeps_one_instance_per_catalog_row(tmp_path):
    corpus, summary = _row_level(tmp_path)
    assert len(corpus.frame) == 3
    assert corpus.raw_row_count == 3
    assert summary.valid_reports_loaded == 3


def test_report_keys_encode_the_physical_spreadsheet_row(tmp_path):
    """R#### is the physical row, which is what the relevance dataset keys on."""
    corpus, _ = _row_level(tmp_path)
    # Banner, End Date, header, then data: the first data row is physical row 4.
    assert list(corpus.frame["report_key"]) == ["R0004", "R0005", "R0006"]
    assert list(corpus.frame["source_row"]) == [4, 5, 6]


def test_report_id_is_dense_and_distinct_from_report_key(tmp_path):
    """`report_id` is a positional index; `report_key` is a physical-row identity.

    They coincide only when no rows are skipped, so nothing may treat them as
    interchangeable.
    """
    corpus, _ = _row_level(tmp_path)
    assert list(corpus.frame["report_id"]) == [0, 1, 2]
    assert list(corpus.frame["report_key"]) != list(corpus.frame["report_id"])


def test_duplicate_titles_stay_separate_instances(tmp_path):
    corpus, summary = _row_level(tmp_path)
    frame = corpus.frame
    assert frame["title_key"].nunique() == 2
    assert frame["report_key"].nunique() == 3
    # The two Headcount rows are one title family with two distinct instances.
    headcount = frame[frame["title_key"] == "headcount by organization"]
    assert len(headcount) == 2
    assert summary.duplicate_report_identities == 1
    assert summary.duplicate_report_rows == 2


# --- fields -----------------------------------------------------------------


def test_fields_come_from_the_workbook_column_and_dedupe(tmp_path):
    corpus, _ = _row_level(tmp_path)
    frame = corpus.frame
    assert frame["fields"][0] == ["Worker ID", "Supervisory Organization", "Headcount"]
    # "worker id" repeats "Worker ID" under normalized identity.
    assert frame["fields"][2] == ["Worker ID", "Termination Reason"]


def test_legacy_field_links_are_marked_as_column_sourced(tmp_path):
    """Provenance must distinguish "no business object" from "linker found none"."""
    cfg = DEFAULT.with_overrides(data_path=_legacy_workbook(tmp_path))
    records = ReportCatalogLoader().load(cfg.data_path)
    linked = fields_from_column(records, source_file=str(cfg.data_path))

    every = [f for fields in linked.values() for f in fields]
    assert every, "fixture should produce field links"
    assert all(f.match_method is MatchMethod.LEGACY_FIELDS_COLUMN for f in every)
    assert all(f.ambiguity_status is AmbiguityStatus.RESOLVED for f in every)
    assert all(f.business_object == "" for f in every)
    assert all(f.field_key.startswith("|") for f in every)


def test_legacy_rows_report_no_ambiguous_links(tmp_path):
    """The Fields column is stated by the source, so nothing is inferred."""
    corpus, _ = _row_level(tmp_path)
    assert not corpus.frame["has_ambiguous_fields"].any()
    assert (corpus.frame["ambiguous_link_fraction"] == 0.0).all()


# --- granularity switch -----------------------------------------------------


def test_family_granularity_still_collapses(tmp_path):
    """The old behavior must remain reachable and unchanged."""
    cfg = DEFAULT.with_overrides(
        ingest_mode="legacy_single_file",
        corpus_granularity="family",
        # About ingestion granularity, not retrieval. `generators` is the default
        # and requires row-level instances, so the compatible mode is stated here.
        retrieval_mode="hybrid",
        data_path=_legacy_workbook(tmp_path),
    )
    corpus, _ = build_corpus(cfg)
    # Three rows, three distinct title+field-set definitions.
    assert len(corpus.frame) == 3
    assert "report_key" not in corpus.frame.columns


def test_row_level_frame_carries_every_rendered_column(tmp_path):
    """`Candidate.row` is a raw Series; the renderers index it by name."""
    corpus, _ = _row_level(tmp_path)
    required = {
        "title", "fields", "prompts", "category", "data_source", "report_type",
        "tags", "shared", "runs", "runs_total", "last_run", "last_updated",
        "last_run_missing", "last_updated_missing", "runs_missing",
        "report_key", "title_key", "family_key", "family_size", "family_rank",
        "description", "area_where_used", "worklet", "chart_type", "landing_page",
    }
    assert required <= set(corpus.frame.columns)


# --- the identity rule the relevance dataset depends on ---------------------


@requires_real_legacy
def test_real_legacy_workbook_loads_at_row_level():
    cfg = DEFAULT.with_overrides(
        ingest_mode="legacy_single_file", corpus_granularity="report_row"
    )
    corpus, summary = build_corpus(cfg)
    assert len(corpus.frame) == 4000
    assert summary.row_validation_errors == 0
    assert corpus.frame["report_key"].is_unique
    # Family identity is the title, not title+field-set: the latter collapses
    # 4000 rows to 3999, which would make family aggregation a no-op.
    assert corpus.frame["title_key"].nunique() == 3280


@requires_relevance_bundle
def test_seed_judgments_resolve_against_legacy_report_keys():
    """The relevance bundle keys on R#### physical rows.

    If row-level ingestion ever stops reproducing that mapping, every label in the
    training set silently points at the wrong report -- so this is asserted rather
    than assumed.
    """
    cfg = DEFAULT.with_overrides(
        ingest_mode="legacy_single_file", corpus_granularity="report_row"
    )
    corpus, _ = build_corpus(cfg)
    titles = dict(
        zip(corpus.frame["report_key"], corpus.frame["title"], strict=True)
    )

    checked = mismatched = absent = 0
    for line in SEED_JUDGMENTS.read_text().splitlines():
        if not line.strip():
            continue
        judgment = json.loads(line)
        checked += 1
        key = judgment["Report Key"]
        if key not in titles:
            absent += 1
        elif str(titles[key]).strip() != str(judgment["Report Title"]).strip():
            mismatched += 1

    assert checked == 10874
    assert absent == 0
    assert mismatched == 0
