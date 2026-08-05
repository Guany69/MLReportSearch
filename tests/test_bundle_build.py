"""Building and loading a bundle.

Runs entirely on fake encoders: the point is the build/reuse/readiness logic, not
the models. A real-model build is exercised by `reportfinder-bundle build` and
recorded in the manifest, not by the unit suite.

Run: uv run pytest tests/test_bundle_build.py -v
"""

from __future__ import annotations

import pandas as pd
import pytest

from reportfinder.config import DEFAULT, from_mapping
from reportfinder.corpus import ViewType, build_corpus_model
from reportfinder.index import BundleNotReady, ComponentStatus
from reportfinder.index.build import build_bundle, bundle_id, load_bundle

from .fakes import FakeEncoder, FakeSparseEncoder, FakeTokenEncoder

TITLES = [
    "Attrition and Replacement Lag Analysis",
    "Headcount by Supervisory Organization",
    "Payroll Earnings Detail",
]


def _corpus(titles=None):
    """A corpus shaped like the real estate.

    Identity and schema discriminate; purpose is thin but varied; interface is
    identical on every row. That last part is not incidental -- on the real
    catalog the interface view has a distinct-text ratio of 0.079, and this
    fixture reproduces that shape.

    Note the metric is a *ratio*: with only three rows, one distinct interface
    value still scores 0.33, which is correctly "as discriminating as a
    three-row corpus can be". `_wide_corpus` is used where the degeneracy
    threshold itself is under test.
    """
    titles = titles or TITLES
    frame = pd.DataFrame([
        {
            "report_key": f"R{4 + i:04d}", "title_key": t.casefold(), "source_row": 4 + i,
            "title": t, "description": f"Used for {t.split()[0].lower()} review.",
            "category": "Worker Data",
            "data_source": "All Workers", "report_type": "Advanced",
            "prompts": ["Effective Date"], "fields": [f"{t.split()[0]} Field"],
            "tags": "", "area_where_used": "",
            # Identical across every instance, as in the real catalog.
            "worklet": "Standard", "chart_type": "Bar", "landing_page": "People",
            "worklet_landing_pages": "", "shared": "Yes",
        }
        for i, t in enumerate(titles)
    ])
    return build_corpus_model(frame, ingest_mode="legacy_single_file", source_file="f")


def _wide_corpus(n: int = 20):
    """Enough rows for the distinct-text ratio to fall below the threshold.

    Every row shares the same interface metadata, so that ratio is 1/n -- the same
    shape the real catalog has at 4000 rows. Titles differ in their *first* word so
    that field names, and therefore the schema view, stay discriminating: the point
    is to isolate one degenerate view, not to degrade the whole corpus.
    """
    return _corpus([f"Metric{i:03d} Detail Report" for i in range(n)])


def _cfg(tmp_path, **overrides):
    base = {
        "ingest_mode": "legacy_single_file",
        "corpus_granularity": "report_row",
        "bundle": {"root": str(tmp_path / "bundle")},
    }
    base.update(overrides)
    return from_mapping(base)


def _build(tmp_path, corpus=None, cfg=None, **kwargs):
    corpus = corpus or _corpus()
    cfg = cfg or _cfg(tmp_path)
    return build_bundle(
        cfg, corpus,
        encoder=kwargs.pop("encoder", FakeEncoder()),
        sparse_encoder=kwargs.pop("sparse_encoder", FakeSparseEncoder()),
        **kwargs,
    )


# --- build ------------------------------------------------------------------


def test_build_produces_a_ready_bundle(tmp_path):
    manifest = _build(tmp_path)
    manifest.assert_ready()
    assert manifest.instance_count == 3
    assert manifest.family_count == 3
    for name in ("views.identity", "views.schema", "splade", "prototypes"):
        assert manifest.components[name].status is ComponentStatus.READY


def test_bundle_id_is_deterministic_for_the_same_corpus_and_config(tmp_path):
    corpus, cfg = _corpus(), _cfg(tmp_path)
    assert bundle_id(corpus, cfg) == bundle_id(_corpus(), cfg)
    assert bundle_id(corpus, cfg) != bundle_id(
        _corpus([*TITLES[:2], "Something Else"]), cfg
    )


def test_manifest_records_every_model_that_can_change_a_result(tmp_path):
    manifest = _build(tmp_path)
    assert manifest.components["views.identity"].model_id == "fake-encoder"
    assert manifest.components["views.identity"].model_revision == "fake-rev"
    assert manifest.components["splade"].model_id == "fake-splade"
    # The cross-encoder stores no index but still affects results, so it is named.
    assert manifest.components["cross_encoder"].model_id == DEFAULT.rerank.checkpoint
    assert manifest.components["cross_encoder"].model_revision == DEFAULT.rerank.revision


def test_rebuild_reuses_unchanged_components(tmp_path):
    cfg = _cfg(tmp_path)
    _build(tmp_path, cfg=cfg)
    encoder = FakeEncoder()
    _build(tmp_path, cfg=cfg, encoder=encoder)
    # Signatures matched, so nothing was re-encoded at all.
    assert encoder.document_calls == []


def test_a_changed_corpus_produces_a_different_bundle(tmp_path):
    cfg = _cfg(tmp_path)
    first = _build(tmp_path, cfg=cfg)
    second = _build(tmp_path, corpus=_corpus([*TITLES[:2], "Renamed Report"]), cfg=cfg)
    assert first.corpus_content_hash != second.corpus_content_hash
    assert first.bundle_version != second.bundle_version


def test_only_builds_the_requested_component(tmp_path):
    cfg = _cfg(tmp_path)
    manifest = _build(tmp_path, cfg=cfg, only=["views.identity"])
    assert manifest.components["views.identity"].status is ComponentStatus.READY
    assert "views.schema" not in manifest.components
    assert "splade" not in manifest.components


# --- degenerate views --------------------------------------------------------


def test_a_near_constant_view_is_recorded_degenerate_rather_than_served(tmp_path):
    """Identical interface metadata across every row carries no signal.

    Serving it would spend shortlist slots and cross-encoder passes on an
    arbitrary but stable top-k for every query.
    """
    manifest = _build(tmp_path, corpus=_wide_corpus())
    interface = manifest.components["views.interface"]
    assert interface.status is ComponentStatus.DEGENERATE_LOW_ENTROPY
    assert interface.fallback == "generator_disabled"
    assert interface.detail["distinct_text_ratio"] < 0.10
    # Degenerate is a degraded state, not a blocking one.
    manifest.assert_ready()


def test_a_discriminating_view_is_not_flagged(tmp_path):
    manifest = _build(tmp_path)
    assert manifest.components["views.schema"].detail["distinct_text_ratio"] == 1.0
    assert manifest.components["views.schema"].status is ComponentStatus.READY


def test_empty_view_rows_are_counted(tmp_path):
    """Blank source text must be reported, not silently embedded as a point."""
    corpus = _corpus()
    # area_where_used and description both blank -> an empty purpose view.
    blank = corpus.views["R0004"]
    assert blank[ViewType.PURPOSE].text
    manifest = _build(tmp_path)
    assert manifest.components["views.purpose"].detail["empty_rows"] == 0


# --- disabled and absent components ------------------------------------------


def test_late_interaction_ships_disabled_with_its_reason_recorded(tmp_path):
    manifest = _build(tmp_path)
    record = manifest.components["late_interaction"]
    assert record.status is ComponentStatus.BUILT_DISABLED
    assert record.fallback == "generator_not_constructed"
    assert "pending latency" in record.detail["reason"]
    # The checkpoint it *would* use is still pinned and recorded.
    assert record.detail["revision"] == DEFAULT.retrieval.late_interaction.revision


def test_late_interaction_builds_when_an_encoder_is_supplied(tmp_path):
    manifest = _build(tmp_path, token_encoder=FakeTokenEncoder())
    assert manifest.components["late_interaction"].status is ComponentStatus.READY


def test_learned_components_ship_absent_with_named_fallbacks(tmp_path):
    manifest = _build(tmp_path)
    assert manifest.components["fusion_model"].status is ComponentStatus.ABSENT
    assert manifest.components["fusion_model"].fallback == "rrf"
    assert manifest.components["decision_model"].status is ComponentStatus.ABSENT
    assert manifest.components["decision_model"].fallback == "deterministic_three_way_policy"
    assert manifest.components["decision_model"].detail["calibrated"] is False


def test_disabling_splade_records_it_rather_than_dropping_it(tmp_path):
    cfg = _cfg(tmp_path, retrieval={"splade": {"enabled": False}},
               bundle={"root": str(tmp_path / "bundle"),
                       "required": ["views.identity", "views.schema", "cross_encoder"]})
    manifest = _build(tmp_path, cfg=cfg)
    assert manifest.components["splade"].status is ComponentStatus.BUILT_DISABLED
    assert manifest.components["splade"].fallback == "generator_not_constructed"


def test_every_built_manifest_satisfies_the_fallback_invariant(tmp_path):
    """No component may be non-ready without saying what happens instead."""
    _build(tmp_path).validate_structure()


# --- load -------------------------------------------------------------------


def test_load_returns_the_ready_indexes(tmp_path):
    cfg = _cfg(tmp_path)
    _build(tmp_path, corpus=_wide_corpus(), cfg=cfg)
    bundle = load_bundle(cfg, _wide_corpus())

    bundle.assert_ready()
    assert set(bundle.dense) == {ViewType.IDENTITY, ViewType.PURPOSE, ViewType.SCHEMA}
    # The degenerate view is not loaded, so no generator can be built from it.
    assert ViewType.INTERFACE not in bundle.dense
    assert bundle.splade is not None
    assert bundle.prototypes is not None
    assert bundle.late_interaction is None


def test_load_surfaces_active_fallbacks(tmp_path):
    cfg = _cfg(tmp_path)
    _build(tmp_path, corpus=_wide_corpus(), cfg=cfg)
    bundle = load_bundle(cfg, _wide_corpus())
    assert "fusion_model:rrf" in bundle.active_fallbacks
    assert "views.interface:generator_disabled" in bundle.active_fallbacks


def test_loading_without_a_build_names_the_command(tmp_path):
    with pytest.raises(FileNotFoundError, match="reportfinder-bundle build"):
        load_bundle(_cfg(tmp_path), _corpus())


def test_a_required_component_missing_blocks_readiness(tmp_path):
    cfg = _cfg(tmp_path, bundle={"root": str(tmp_path / "bundle"),
                                 "required": ["views.identity", "late_interaction"]})
    manifest = _build(tmp_path, cfg=cfg)
    with pytest.raises(BundleNotReady) as excinfo:
        manifest.assert_ready()
    assert "late_interaction" in excinfo.value.blocking


# --- build-time versus runtime configuration ---------------------------------
#
# `bundle_id` keys on corpus content plus *index* config only, deliberately: a
# shortlist depth change must not re-encode 4,000 documents. The cost is that one
# bundle id can be served under materially different retrieval behaviour. These
# pin the disclosure that makes that visible.


def test_a_matching_runtime_config_reports_no_drift(tmp_path):
    cfg = _cfg(tmp_path)
    _build(tmp_path, corpus=_wide_corpus(), cfg=cfg)
    bundle = load_bundle(cfg, _wide_corpus())

    assert bundle.runtime_config_hash
    assert bundle.runtime_config_hash == bundle.manifest.config_hash
    assert bundle.config_drift is False


def test_a_serving_only_change_keeps_the_bundle_but_reports_drift(tmp_path):
    """The exact case the bundle id cannot see: same vectors, different serving."""
    cfg = _cfg(tmp_path)
    _build(tmp_path, corpus=_wide_corpus(), cfg=cfg)

    tweaked = cfg.with_path_overrides({
        "shortlist.standard_rerank_depth": 121,
        "shortlist.high_risk_rerank_depth": 201,
    })
    assert bundle_id(_wide_corpus(), tweaked) == bundle_id(_wide_corpus(), cfg), (
        "a shortlist depth must not invalidate encoded documents"
    )

    bundle = load_bundle(tweaked, _wide_corpus())
    assert bundle.config_drift is True
    assert bundle.runtime_config_hash != bundle.manifest.config_hash
    assert bundle.manifest.config_hash, "the build-time hash is still recorded"
