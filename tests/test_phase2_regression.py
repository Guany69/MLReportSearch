"""Backward compatibility and search regression for the dual-file ingest path.

Two questions this file exists to answer:

1. Does Phase 2 reconstruct what Phase 1 read directly? (equivalence)
2. Does adding Phase 2 metadata break or distort existing ranking? (regression)

Both are answered against controlled synthetic fixtures built by
`phase2_fixtures.py`, which is what lets them run at all: the real Phase 2
workbooks are not in this tree.

Four further tests used to assert against those real workbooks -- reconstruction
precision/recall of 1.0, the exact link counters, and the strict/composite policy
trade-offs. They were permanently skipped and have been removed rather than left
as decoration. What they measured was the linker's behaviour *on one specific
estate*; the mechanics they exercised are covered here synthetically. Restoring
them means restoring the workbooks, at which point they are worth rewriting
against whatever the new estate actually contains rather than against numbers
copied from a dataset nobody has.

Run: uv run pytest tests/test_phase2_regression.py -v
"""

from __future__ import annotations

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


def test_phase2_preserves_catalog_metadata_and_link_aggregates(tmp_path):
    catalog = write_catalog([catalog_row(
        "Alpha", Description="Useful report description", **{
            "Area Where Used": "Benefits Hub", "Landing Page": "People",
            "Chart Type": "Bar", "Owner": "Reporting Team",
        })], tmp_path / "cat.xlsx")
    dictionary = write_dictionary(
        [dictionary_row("Worker", "Status", "Alpha")], tmp_path / "dict.xlsx"
    )
    corpus, _ = build_corpus(DEFAULT.with_overrides(
        ingest_mode="phase2_dual_file", catalog_path=catalog,
        field_dictionary_path=dictionary,
    ))
    row = corpus.frame.iloc[0]
    assert row["report_id"] == 0
    assert row["description"] == "Useful report description"
    assert row["area_where_used"] == "Benefits Hub"
    assert row["landing_page"] == "People" and row["chart_type"] == "Bar"
    assert row["field_link_confidence"] == 1.0
    assert row["ambiguous_link_fraction"] == 0.0


def test_legacy_family_mode_has_empty_phase2_enrichment(tmp_path):
    """Family granularity still attaches the Phase 2 columns empty."""
    legacy_path = write_legacy(
        [dict(catalog_row("Alpha"), Fields="Status")], tmp_path / "legacy.xlsx"
    )
    corpus, _ = build_corpus(
        DEFAULT.with_overrides(
            ingest_mode="legacy_single_file",
            corpus_granularity="family",
            # Family granularity cannot run under the `generators` default.
            retrieval_mode="hybrid",
            data_path=legacy_path,
        )
    )
    row = corpus.frame.iloc[0]
    assert row["field_meta"] == []
    # Value, not identity: pandas stores this column as numpy bool.
    assert not row["has_ambiguous_fields"]
    assert row["ambiguous_field_count"] == 0


def test_legacy_row_mode_links_fields_without_phase2_metadata(tmp_path):
    """Row granularity populates `field_meta`, but with no Phase 2 *content*.

    Row-level legacy ingestion needs per-field provenance to give instances an
    identity. What it must not do is invent Phase 2 metadata it never read: the
    business object, domain, categories and field descriptions stay empty, which
    is what keeps `doc(r)` byte-identical (asserted separately below).
    """
    legacy_path = write_legacy(
        [dict(catalog_row("Alpha"), Fields="Status")], tmp_path / "legacy.xlsx"
    )
    corpus, _ = build_corpus(
        DEFAULT.with_overrides(
            ingest_mode="legacy_single_file",
            corpus_granularity="report_row",
            data_path=legacy_path,
        )
    )
    row = corpus.frame.iloc[0]
    assert [f.field_name for f in row["field_meta"]] == ["Status"]
    assert all(
        f.business_object == ""
        and f.domain == ""
        and f.categories == ""
        and f.description == ""
        and f.built_in_prompts == ""
        for f in row["field_meta"]
    )
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
