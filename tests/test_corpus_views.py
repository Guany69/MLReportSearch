"""The two-level model and the four searchable views.

The load-bearing claims here:

1. Family identity is the normalized title, so families are real groupings rather
   than one-instance-each.
2. Views are deterministic and independently hashed, so an edit to one view does
   not invalidate the other three.
3. Operational metadata (owner, run counts, last-run dates) never becomes view
   text. This is the one that silently degrades retrieval if it breaks: embedding
   usage data teaches the retriever that popular reports are closer to every query.

Run: uv run pytest tests/test_corpus_views.py -v
"""

from __future__ import annotations

import pandas as pd
import pytest

from reportfinder.config import DEFAULT
from reportfinder.corpus import (
    ALL_VIEW_TYPES,
    UNRESTRICTED_ACL_KEY,
    LifecycleState,
    ViewType,
    authoritative_text,
    build_corpus_model,
    build_views,
)
from reportfinder.corpus.identity import instance_from_row
from reportfinder.ingest import build_corpus

OPERATIONAL_VALUES = {
    "owner": "Dana Fielding",
    "created_by": "Ravi Menon",
    "last_run_by": "Priya Shah",
    "available_usage": "Default Areas",
}


def _row(**overrides) -> pd.Series:
    row = {
        "report_key": "R0004",
        "title_key": "headcount by organization",
        "source_row": 4,
        "title": "Headcount by Organization",
        "description": "Embedded on tasks.",
        "category": "Worker Data",
        "data_source": "All Workers",
        "report_type": "Advanced",
        "prompts": ["Effective Date", "Organization"],
        "fields": ["Worker ID", "Headcount", "Supervisory Organization"],
        "tags": "HR; Headcount",
        "area_where_used": "People Hub",
        "worklet": "Org Insights",
        "chart_type": "Bar",
        "landing_page": "People",
        "worklet_landing_pages": "People; Manager Hub",
        "shared": "Yes",
        "runs": 4211,
        "runs_total": 5000,
        "last_run": "2026-06-01",
        "last_updated": "2026-01-01",
        **OPERATIONAL_VALUES,
    }
    row.update(overrides)
    return pd.Series(row)


def _instance(**overrides):
    return instance_from_row(
        _row(**overrides),
        catalog_version="cafe1234",
        ingest_mode="legacy_single_file",
        source_file="data/Reports.xlsx",
        loader="test",
    )


# --- instance identity ------------------------------------------------------


def test_instance_carries_the_specified_identity_fields():
    instance = _instance()
    assert instance.report_instance_id == "R0004"
    assert instance.instance_id == "R0004"
    assert instance.family_id == "headcount by organization"
    assert instance.fields == ("Worker ID", "Headcount", "Supervisory Organization")
    assert instance.prompts == ("Effective Date", "Organization")
    assert instance.interface_metadata["chart_type"] == "Bar"
    assert instance.catalog_version == "cafe1234"
    assert instance.provenance.source_row == 4
    assert instance.provenance.ingest_mode == "legacy_single_file"


def test_placeholders_are_explicit_rather_than_invented():
    """No lifecycle column and no ACL source exist; both must say so."""
    instance = _instance()
    assert instance.lifecycle_state is LifecycleState.UNKNOWN
    assert instance.acl_key == UNRESTRICTED_ACL_KEY


def test_instance_requires_row_level_granularity():
    with pytest.raises(ValueError, match="row-level corpus"):
        instance_from_row(
            pd.Series({"title": "Alpha"}),
            catalog_version="v",
            ingest_mode="legacy_single_file",
            source_file="f",
            loader="test",
        )


# --- views ------------------------------------------------------------------


def test_each_view_holds_only_its_own_facet():
    views = build_views(_instance())
    identity = views[ViewType.IDENTITY].text
    purpose = views[ViewType.PURPOSE].text
    schema = views[ViewType.SCHEMA].text
    interface = views[ViewType.INTERFACE].text

    assert "Headcount by Organization" in identity and "Worker Data" in identity
    assert "Embedded on tasks." in purpose and "People Hub" in purpose
    assert "Supervisory Organization" in schema and "Effective Date" in schema
    assert "Bar" in interface and "Org Insights" in interface

    # Facets must not bleed: a field name in the identity view would let schema
    # matches win identity retrieval and vice versa.
    assert "Supervisory Organization" not in identity
    assert "Bar" not in schema


def test_views_are_deterministic():
    first = build_views(_instance())
    second = build_views(_instance())
    for view_type in ALL_VIEW_TYPES:
        assert first[view_type].content_hash == second[view_type].content_hash
        assert first[view_type].text == second[view_type].text


def test_editing_one_view_does_not_change_the_others():
    """This is what makes incremental re-embedding possible."""
    base = build_views(_instance())
    retitled = build_views(_instance(title="Headcount by Organisation"))

    assert retitled[ViewType.IDENTITY].content_hash != base[ViewType.IDENTITY].content_hash
    for view_type in (ViewType.PURPOSE, ViewType.SCHEMA, ViewType.INTERFACE):
        assert retitled[view_type].content_hash == base[view_type].content_hash


def test_aliases_reach_the_identity_view():
    views = build_views(_instance(), aliases=("Headcount By Org",))
    assert "Headcount By Org" in views[ViewType.IDENTITY].text


def test_empty_view_is_flagged_not_silently_embedded():
    views = build_views(_instance(description="", area_where_used=""))
    assert views[ViewType.PURPOSE].is_empty
    assert not views[ViewType.SCHEMA].is_empty


@pytest.mark.parametrize("view_type", list(ALL_VIEW_TYPES))
def test_operational_metadata_never_becomes_view_text(view_type):
    """Owner, creator, last-run user and usage are not report meaning.

    If this fails, the dense index has learned that frequently-run reports are
    semantically nearer to every query, which is invisible in ranking metrics and
    very hard to diagnose later.
    """
    text = build_views(_instance())[view_type].text
    for value in OPERATIONAL_VALUES.values():
        assert value not in text
    assert "4211" not in text and "5000" not in text
    assert "2026-06-01" not in text


def test_authoritative_text_excludes_operational_metadata_and_bounds_length():
    instance = _instance()
    text = authoritative_text(instance)
    assert "Headcount by Organization" in text
    assert "Supervisory Organization" in text
    for value in OPERATIONAL_VALUES.values():
        assert value not in text

    long_text = authoritative_text(_instance(fields=[f"Field {i}" for i in range(400)]),
                                   max_chars=200)
    assert len(long_text) <= 200
    # Truncation lands on a separator, never mid-field.
    assert not long_text.endswith("Field")


# --- corpus model -----------------------------------------------------------


def _model(rows: list[dict]):
    frame = pd.DataFrame([_row(**r) for r in rows])
    return build_corpus_model(
        frame, ingest_mode="legacy_single_file", source_file="data/Reports.xlsx"
    )


def test_families_group_by_normalized_title():
    model = _model([
        {"report_key": "R0004", "title": "Headcount by Organization"},
        {"report_key": "R0005", "title": "HEADCOUNT BY ORGANIZATION"},
        {"report_key": "R0006", "title": "Termination Detail",
         "title_key": "termination detail"},
    ])
    assert len(model) == 3
    assert len(model.families) == 2
    family = model.families["headcount by organization"]
    assert family.instance_ids == ("R0004", "R0005")
    assert family.canonical_title == "Headcount by Organization"
    # A genuine cosmetic variant, not an invented synonym.
    assert family.aliases == ("HEADCOUNT BY ORGANIZATION",)


def test_position_mapping_round_trips():
    model = _model([
        {"report_key": "R0004"}, {"report_key": "R0005"}, {"report_key": "R0006"},
    ])
    for position, instance in enumerate(model.instances):
        assert model.position_of(instance.report_instance_id) == position
        assert model.instance(instance.report_instance_id) is instance


def test_content_hash_is_order_independent_but_content_sensitive():
    a = _model([{"report_key": "R0004"}, {"report_key": "R0005"}])
    same = _model([{"report_key": "R0004"}, {"report_key": "R0005"}])
    changed = _model([{"report_key": "R0004"}, {"report_key": "R0005", "title": "Other"}])
    assert a.content_hash == same.content_hash
    assert a.content_hash != changed.content_hash


def test_family_collapsed_frame_is_rejected():
    frame = pd.DataFrame({"title": ["Alpha"], "fields": [["Status"]]})
    with pytest.raises(ValueError, match="row-level corpus"):
        build_corpus_model(frame, ingest_mode="legacy_single_file", source_file="f")


# --- against the real estate ------------------------------------------------


@pytest.mark.skipif(not DEFAULT.data_path.exists(), reason="legacy workbook absent")
def test_real_estate_yields_the_expected_two_level_shape():
    cfg = DEFAULT.with_overrides(
        ingest_mode="legacy_single_file", corpus_granularity="report_row"
    )
    corpus, _ = build_corpus(cfg)
    model = build_corpus_model(
        corpus.frame, ingest_mode=cfg.ingest_mode, source_file=str(cfg.data_path)
    )
    assert len(model) == 4000
    # 3280 families over 4000 instances: family aggregation is meaningful here,
    # which it would not be under title+field-set identity (that yields 3999).
    assert len(model.families) == 3280
    assert max(len(f.instance_ids) for f in model.families.values()) == 7
    assert model.catalog_version != "absent"
    assert len(model.content_hash) == 16
