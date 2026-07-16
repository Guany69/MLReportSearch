"""Backward compatibility and search regression for Phase 2.

Two questions this file exists to answer:

1. Does Phase 2 reconstruct what Phase 1 read directly? (equivalence)
2. Does adding Phase 2 metadata break or distort existing ranking? (regression)

The equivalence test runs against the *real* workbooks when present, because a
synthetic fixture can only prove the code is self-consistent -- not that it
reconstructs the actual estate. It falls back to a controlled fixture otherwise.

Run: uv run pytest tests/test_phase2_regression.py -v
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from reportfinder.config import DEFAULT
from reportfinder.ingest import build_corpus
from reportfinder.represent import build_doc

from .phase2_fixtures import (
    catalog_row,
    dictionary_row,
    write_catalog,
    write_dictionary,
    write_legacy,
)

REAL_LEGACY = DEFAULT.data_path
REAL_CATALOG = DEFAULT.catalog_path
REAL_DICTIONARY = DEFAULT.field_dictionary_path

have_real_data = REAL_LEGACY.exists() and REAL_CATALOG.exists() and REAL_DICTIONARY.exists()
requires_real_data = pytest.mark.skipif(
    not have_real_data, reason="real Phase 1/Phase 2 workbooks not present in data/"
)


# --- backward compatibility: controlled fixture -----------------------------


def test_legacy_and_phase2_agree_on_controlled_fixture(tmp_path):
    """Same underlying truth expressed both ways must yield the same fields."""
    legacy_rows = [
        dict(catalog_row("Alpha"), Fields="Status; Hire Date"),
        dict(catalog_row("Beta"), Fields="Status"),
    ]
    legacy_path = write_legacy(legacy_rows, tmp_path / "legacy.xlsx")

    catalog = write_catalog(
        [catalog_row("Alpha"), catalog_row("Beta")], tmp_path / "cat.xlsx"
    )
    dictionary = write_dictionary(
        [
            dictionary_row("Worker", "Status", "Alpha; Beta"),
            dictionary_row("Worker", "Hire Date", "Alpha"),
        ],
        tmp_path / "dict.xlsx",
    )

    legacy_corpus, _ = build_corpus(
        DEFAULT.with_overrides(ingest_mode="legacy_single_file", data_path=legacy_path)
    )
    phase2_corpus, _ = build_corpus(
        DEFAULT.with_overrides(
            ingest_mode="phase2_dual_file",
            catalog_path=catalog,
            field_dictionary_path=dictionary,
        )
    )

    def fields_by_title(corpus):
        return {
            r["title"]: set(f.casefold() for f in r["fields"])
            for _, r in corpus.frame.iterrows()
        }

    assert fields_by_title(legacy_corpus) == fields_by_title(phase2_corpus)


def test_both_modes_produce_the_same_corpus_contract(tmp_path):
    """The ML pipeline must not be able to tell which mode produced the corpus."""
    legacy_path = write_legacy(
        [dict(catalog_row("Alpha"), Fields="Status")], tmp_path / "legacy.xlsx"
    )
    catalog = write_catalog([catalog_row("Alpha")], tmp_path / "cat.xlsx")
    dictionary = write_dictionary(
        [dictionary_row("Worker", "Status", "Alpha")], tmp_path / "dict.xlsx"
    )

    legacy_corpus, _ = build_corpus(
        DEFAULT.with_overrides(ingest_mode="legacy_single_file", data_path=legacy_path)
    )
    phase2_corpus, _ = build_corpus(
        DEFAULT.with_overrides(
            ingest_mode="phase2_dual_file",
            catalog_path=catalog,
            field_dictionary_path=dictionary,
        )
    )

    required = {
        "title", "fields", "prompts", "category", "data_source", "report_type",
        "tags", "family_size", "runs", "last_run", "last_updated",
        "field_meta", "has_ambiguous_fields", "ambiguous_field_count",
    }
    assert required <= set(legacy_corpus.frame.columns)
    assert required <= set(phase2_corpus.frame.columns)
    # fields is list[str] in both modes -- the contract represent/model rely on.
    for corpus in (legacy_corpus, phase2_corpus):
        value = corpus.frame.iloc[0]["fields"]
        assert isinstance(value, list) and all(isinstance(v, str) for v in value)


def test_legacy_mode_has_empty_phase2_enrichment(tmp_path):
    legacy_path = write_legacy(
        [dict(catalog_row("Alpha"), Fields="Status")], tmp_path / "legacy.xlsx"
    )
    corpus, _ = build_corpus(
        DEFAULT.with_overrides(ingest_mode="legacy_single_file", data_path=legacy_path)
    )
    row = corpus.frame.iloc[0]
    assert row["field_meta"] == []
    # Value, not identity: pandas stores this column as numpy bool.
    assert not row["has_ambiguous_fields"]
    assert row["ambiguous_field_count"] == 0


# --- doc(r) regression ------------------------------------------------------


def test_legacy_doc_is_unchanged_by_phase2_code(tmp_path):
    """Legacy doc(r) must be byte-identical to the Phase 1 representation.

    Phase 2 zones are additive; with no field metadata they must contribute
    nothing at all, or every cached vector and tuned threshold shifts silently.
    """
    legacy_path = write_legacy(
        [dict(catalog_row("Alpha"), Fields="Status; Hire Date")], tmp_path / "legacy.xlsx"
    )
    corpus, _ = build_corpus(
        DEFAULT.with_overrides(ingest_mode="legacy_single_file", data_path=legacy_path)
    )
    row = corpus.frame.iloc[0]
    doc = build_doc(row, DEFAULT)

    expected_zones = (
        [row["title"]] * DEFAULT.w_title
        + [", ".join(row["fields"])] * DEFAULT.w_fields
        + [row["category"]] * DEFAULT.w_category
        + [row["data_source"]] * DEFAULT.w_data_source
        + [row["report_type"]] * DEFAULT.w_report_type
    )
    assert doc == " . ".join(expected_zones)


def test_phase2_zones_appear_only_with_field_metadata(tmp_path):
    catalog = write_catalog([catalog_row("Alpha")], tmp_path / "cat.xlsx")
    dictionary = write_dictionary(
        [
            dictionary_row(
                "Worker", "Status", "Alpha",
                Description="The employment status of the worker.",
                Domain="Worker Data",
            )
        ],
        tmp_path / "dict.xlsx",
    )
    corpus, _ = build_corpus(
        DEFAULT.with_overrides(
            ingest_mode="phase2_dual_file",
            catalog_path=catalog,
            field_dictionary_path=dictionary,
        )
    )
    doc = build_doc(corpus.frame.iloc[0], DEFAULT)
    assert "The employment status of the worker." in doc
    assert "Worker Data" in doc


def test_phase2_zone_weights_are_configurable_and_can_be_disabled(tmp_path):
    catalog = write_catalog([catalog_row("Alpha")], tmp_path / "cat.xlsx")
    dictionary = write_dictionary(
        [dictionary_row("Worker", "Status", "Alpha", Description="Unique description text.")],
        tmp_path / "dict.xlsx",
    )
    corpus, _ = build_corpus(
        DEFAULT.with_overrides(
            ingest_mode="phase2_dual_file",
            catalog_path=catalog,
            field_dictionary_path=dictionary,
        )
    )
    row = corpus.frame.iloc[0]
    off = DEFAULT.with_overrides(
        w_field_description=0, w_business_object=0, w_domain=0,
        w_field_categories=0, w_field_type=0, w_builtin_prompts=0,
        w_related_business_object=0,
    )
    assert "Unique description text." not in build_doc(row, off)
    assert "Unique description text." in build_doc(row, DEFAULT)


def test_authorized_usage_never_enters_the_document(tmp_path):
    """1 distinct value across the real dictionary => zero signal, must stay out."""
    catalog = write_catalog([catalog_row("Alpha")], tmp_path / "cat.xlsx")
    dictionary = write_dictionary(
        [dictionary_row("Worker", "Status", "Alpha", **{"Authorized Usage": "Default Areas"})],
        tmp_path / "dict.xlsx",
    )
    corpus, _ = build_corpus(
        DEFAULT.with_overrides(
            ingest_mode="phase2_dual_file",
            catalog_path=catalog,
            field_dictionary_path=dictionary,
        )
    )
    row = corpus.frame.iloc[0]
    assert row["field_meta"][0].authorized_usage == "Default Areas", "kept as metadata"
    assert "Default Areas" not in build_doc(row, DEFAULT), "but never in doc(r)"


def test_phase2_zones_dedupe_shared_values(tmp_path):
    """Many fields sharing a domain must not re-weight that zone per field."""
    catalog = write_catalog([catalog_row("Alpha")], tmp_path / "cat.xlsx")
    dictionary = write_dictionary(
        [
            dictionary_row("Worker", "Status", "Alpha", Domain="Worker Data"),
            dictionary_row("Worker", "Hire Date", "Alpha", Domain="Worker Data"),
            dictionary_row("Worker", "Location", "Alpha", Domain="Worker Data"),
        ],
        tmp_path / "dict.xlsx",
    )
    corpus, _ = build_corpus(
        DEFAULT.with_overrides(
            ingest_mode="phase2_dual_file",
            catalog_path=catalog,
            field_dictionary_path=dictionary,
        )
    )
    doc = build_doc(corpus.frame.iloc[0], DEFAULT)
    # Domain zone weight is 1; three fields share the domain but it appears once
    # in that zone (the category zone contributes the other occurrence).
    assert doc.count("Worker Data") <= 2


# --- real-data equivalence --------------------------------------------------


@requires_real_data
def test_phase2_reconstructs_real_phase1_fields():
    """The acceptance gate: reconstruction vs authored ground truth.

    Compared per report *title*. Where_Used records that a field is used by a
    report of a given name; it cannot say which duplicate-titled row. Title level
    is therefore the granularity the source data actually supports -- and is the
    granularity family collapse operates at.
    """
    legacy, _ = build_corpus(DEFAULT.with_overrides(ingest_mode="legacy_single_file"))
    phase2, _ = build_corpus(DEFAULT.with_overrides(ingest_mode="phase2_dual_file"))

    def by_title(corpus):
        out: dict[str, set[str]] = {}
        for _, row in corpus.frame.iterrows():
            out.setdefault(row["title"].casefold(), set()).update(
                f.casefold() for f in row["fields"]
            )
        return out

    truth, got = by_title(legacy), by_title(phase2)
    assert set(truth) == set(got), "no report titles lost or invented"

    precision, recall = [], []
    for key, want in truth.items():
        have = got[key]
        if not want:
            continue
        precision.append(len(want & have) / len(have) if have else 0.0)
        recall.append(len(want & have) / len(want))

    assert np.mean(recall) == 1.0, "every authored field must be reconstructed"
    assert np.mean(precision) == 1.0, "and no field invented"


@requires_real_data
def test_real_import_metrics_are_within_expected_bounds():
    """Guards the linking result on the real estate against silent drift."""
    _, summary = build_corpus(DEFAULT.with_overrides(ingest_mode="phase2_dual_file"))

    assert summary.report_rows_read == 4000
    assert summary.unmatched_where_used == 0, "every Where_Used name exists in catalog"
    assert summary.reports_with_zero_fields == 0, "permissive leaves no report field-blind"
    assert summary.exact_name_links > 30000
    assert summary.composite_links > 10000, "business-object corroboration is carrying weight"
    assert summary.ambiguous_links > 0, "genuinely undecidable links exist and are reported"
    assert summary.duplicate_report_identities == 533
    assert summary.import_duration_s < 60


@requires_real_data
def test_strict_policy_leaves_reports_field_blind_on_real_data():
    """Documents the trade-off behind choosing permissive, on real data."""
    _, strict = build_corpus(
        DEFAULT.with_overrides(
            ingest_mode="phase2_dual_file", ambiguity_policy="strict"
        )
    )
    assert strict.reports_with_zero_fields > 0, (
        "strict withholds undecidable links, so some reports get no fields at all"
    )


@requires_real_data
def test_composite_matching_can_be_disabled():
    _, without = build_corpus(
        DEFAULT.with_overrides(
            ingest_mode="phase2_dual_file", enable_composite_match=False
        )
    )
    assert without.composite_links == 0
    assert without.ambiguous_links > 1936, "without corroboration, more stays ambiguous"
