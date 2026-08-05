"""Nested configuration: loading, dotted overrides, and cross-object validation.

The generator architecture adds ~70 knobs. Nesting them keeps the YAML readable,
but it introduces three ways to get things quietly wrong, each pinned here:

1. A partial nested mapping silently resetting its siblings.
2. A partial `quotas` mapping silently dropping source-exclusive quotas -- which
   would remove the guarantee the whole architecture exists to provide.
3. Validation running on a half-applied override and rejecting a valid end state.

Run: uv run pytest tests/test_config_nesting.py -v
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from reportfinder.config import (
    DEFAULT,
    Config,
    from_mapping,
    resolve_config_path,
)

CONFIG_DIR = Path(__file__).resolve().parents[1] / "configs"
# Discovered rather than enumerated. Every file in configs/ is now a Config
# document -- the ones that were not (relevance.yaml, train_*.yaml) pointed at
# workbooks absent from this tree and have been removed. Globbing means adding or
# removing a config cannot leave this list quietly stale.
CONFIG_DOCUMENTS = sorted(path.name for path in CONFIG_DIR.glob("*.yaml"))


# --- loading ----------------------------------------------------------------


@pytest.mark.parametrize("name", CONFIG_DOCUMENTS)
def test_every_config_document_loads(name):
    raw = yaml.safe_load((CONFIG_DIR / name).read_text()) or {}
    assert isinstance(from_mapping(raw), Config)


def test_flat_only_documents_still_load_unchanged():
    """Pre-existing configs use flat keys and must keep working verbatim.

    Written against an inline document rather than a named file: the claim is
    about the loader accepting flat-only keys, and tying it to one config meant
    deleting that config broke a test that was not about it.
    """
    cfg = from_mapping(yaml.safe_load("""
    ingest_mode: legacy_single_file
    corpus_granularity: report_row
    retrieval_mode: hybrid
    candidate_k: 100
    rrf_k: 60
    top_k: 10
    use_cross_encoder: false
    """))
    assert cfg.retrieval_mode == "hybrid"
    assert cfg.candidate_k == 100
    # Untouched nested objects keep their defaults.
    assert cfg.retrieval.dense.k_per_view == DEFAULT.retrieval.dense.k_per_view


def test_nested_override_leaves_siblings_alone():
    cfg = from_mapping(yaml.safe_load("""
    retrieval:
      dense: {k_per_view: 80}
    """))
    assert cfg.retrieval.dense.k_per_view == 80
    assert cfg.retrieval.splade.k == DEFAULT.retrieval.splade.k
    assert cfg.retrieval.bm25f.k == DEFAULT.retrieval.bm25f.k
    assert cfg.retrieval.rrf_constant == DEFAULT.retrieval.rrf_constant


def test_generators_config_declares_the_specified_architecture():
    cfg = from_mapping(yaml.safe_load((CONFIG_DIR / "legacy_generators.yaml").read_text()))
    assert cfg.retrieval_mode == "generators"
    assert cfg.corpus_granularity == "report_row"
    assert cfg.retrieval.bm25f.k == 100
    assert cfg.retrieval.splade.enabled and cfg.retrieval.splade.k == 100
    assert cfg.retrieval.dense.k_per_view == 50
    assert cfg.retrieval.dense.views == ("identity", "purpose", "schema", "interface")
    assert cfg.retrieval.prototypes.k == 50
    assert cfg.retrieval.rrf_constant == 60
    assert cfg.shortlist.standard_rerank_depth == 120
    assert cfg.shortlist.high_risk_rerank_depth == 200
    # Shipped fallbacks, not trained artifacts.
    assert cfg.fusion.artifact_path is None and cfg.fusion.fallback == "rrf"
    assert cfg.decision.artifact_path is None
    assert cfg.retrieval.late_interaction.enabled is False
    assert cfg.security.trust_remote_code is False


# --- quotas -----------------------------------------------------------------


def test_partial_quota_override_merges_rather_than_replaces():
    """Setting one quota must not silently zero the others.

    A file that set only `fused` would otherwise drop every source-exclusive
    quota, and the shortlist would quietly stop preserving candidates that a
    single generator found -- the exact failure this architecture removes.
    """
    cfg = from_mapping(yaml.safe_load("shortlist:\n  quotas: {fused: 40}\n"))
    quotas = cfg.shortlist.quota_map
    assert quotas["fused"] == 40
    assert quotas["bm25_exclusive"] == 15
    assert quotas["splade_exclusive"] == 15
    assert quotas["prototype_exclusive"] == 10
    assert set(quotas) == set(DEFAULT.shortlist.quota_map)


def test_dropping_a_quota_must_be_explicit():
    cfg = from_mapping(yaml.safe_load("shortlist:\n  quotas: {purpose_exclusive: 0}\n"))
    assert cfg.shortlist.quota_map["purpose_exclusive"] == 0
    assert cfg.shortlist.quota_map["schema_exclusive"] == 10


def test_unknown_quota_name_is_rejected_with_a_suggestion():
    with pytest.raises(ValueError, match="Did you mean: fused"):
        from_mapping({"shortlist": {"quotas": {"fusedd": 1}}})


# --- rejection ---------------------------------------------------------------


def test_unknown_top_level_key_is_rejected():
    with pytest.raises(ValueError, match="unknown configuration key"):
        from_mapping({"definitely_not_a_key": 1})


def test_unknown_nested_key_names_its_object_and_suggests():
    with pytest.raises(ValueError, match=r"in retrieval\.dense\. Did you mean: k_per_view"):
        from_mapping({"retrieval": {"dense": {"k_per_vew": 1}}})


# --- dotted paths ------------------------------------------------------------


def test_resolve_config_path_walks_nested_objects():
    owner, attr, value = resolve_config_path(DEFAULT, "retrieval.dense.k_per_view")
    assert attr == "k_per_view"
    assert value == 50
    assert owner is DEFAULT.retrieval.dense


def test_resolve_config_path_rejects_unknown_paths():
    with pytest.raises(ValueError, match="unknown configuration path"):
        resolve_config_path(DEFAULT, "retrieval.dense.nope")


def test_path_overrides_apply_together_not_one_at_a_time():
    """Both depths move at once.

    Applied sequentially, raising the standard depth above the current high-risk
    depth would trip the "high risk must never be narrower" invariant on an
    intermediate state, even though the end state is valid.
    """
    cfg = DEFAULT.with_path_overrides({
        "shortlist.standard_rerank_depth": 300,
        "shortlist.high_risk_rerank_depth": 400,
    })
    assert cfg.shortlist.standard_rerank_depth == 300
    assert cfg.shortlist.high_risk_rerank_depth == 400


def test_path_overrides_mix_flat_and_nested():
    cfg = DEFAULT.with_path_overrides({"top_k": 7, "retrieval.splade.k": 250})
    assert cfg.top_k == 7
    assert cfg.retrieval.splade.k == 250


# --- invariants --------------------------------------------------------------


def test_high_risk_depth_may_not_be_narrower_than_standard():
    """A vague query must receive a wider pool, never a narrower one."""
    with pytest.raises(ValueError, match="never receive a narrower pool"):
        from_mapping({"shortlist": {"high_risk_rerank_depth": 50}})


def test_exclusive_quotas_may_not_exceed_the_shortlist():
    with pytest.raises(ValueError, match="exceed standard_rerank_depth"):
        from_mapping({"shortlist": {"standard_rerank_depth": 20}})


def test_enabling_late_interaction_requires_it_to_be_a_required_component():
    """Otherwise a query could run without the index and never say so."""
    with pytest.raises(ValueError, match="bundle.required"):
        from_mapping({"retrieval": {"late_interaction": {"enabled": True}}})


def test_enabling_late_interaction_with_the_component_declared_is_accepted():
    cfg = from_mapping({
        "retrieval": {"late_interaction": {"enabled": True}},
        "bundle": {"required": ["views.identity", "late_interaction"]},
    })
    assert cfg.retrieval.late_interaction.enabled


def test_allow_all_resolver_requires_explicit_opt_in():
    """An open door must never be inheritable by accident."""
    with pytest.raises(ValueError, match="explicit opt-in"):
        from_mapping({"auth": {"allow_dev_default": False}})


def test_generators_mode_requires_row_level_granularity():
    with pytest.raises(ValueError, match="needs distinct instances"):
        from_mapping({"retrieval_mode": "generators", "corpus_granularity": "family"})


def test_calibration_without_a_decision_artifact_is_rejected():
    with pytest.raises(ValueError, match="nothing to calibrate"):
        from_mapping({"decision": {"calibration_path": "artifacts/cal.pt"}})


def test_legacy_retrieval_modes_are_still_accepted():
    for mode in ("hybrid", "legacy_weighted_logit"):
        assert from_mapping({"retrieval_mode": mode}).retrieval_mode == mode


# --- the search CLI's --config -----------------------------------------------
#
# `reportfinder-bundle` has always taken --config; `python -m reportfinder` did
# not, so a search always ran the in-code DEFAULT no matter which configuration
# its bundle was built from. These pin the loader and the override precedence.


def test_the_cli_loads_a_yaml_config_as_its_base(tmp_path):
    from reportfinder.__main__ import base_config

    path = tmp_path / "c.yaml"
    path.write_text("top_k: 3\nshortlist:\n  standard_rerank_depth: 77\n")
    cfg = base_config(path)

    assert cfg.top_k == 3
    assert cfg.shortlist.standard_rerank_depth == 77
    # A nested override must not reset its siblings.
    assert cfg.shortlist.high_risk_rerank_depth == DEFAULT.shortlist.high_risk_rerank_depth


def test_nested_ingestion_settings_survive_cli_loading(tmp_path):
    from reportfinder.__main__ import base_config

    path = tmp_path / "c.yaml"
    path.write_text(
        "ingest_mode: phase2_dual_file\n"
        "corpus_granularity: report_row\n"
        "shortlist:\n  diversify_families: false\n"
    )
    cfg = base_config(path)

    assert cfg.ingest_mode == "phase2_dual_file"
    assert cfg.shortlist.diversify_families is False


def test_no_config_still_starts_from_the_in_code_default():
    from reportfinder.__main__ import base_config

    assert base_config(None) is DEFAULT


def test_an_explicit_flag_beats_the_config_file(tmp_path):
    """`with_overrides` drops Nones, so an unpassed flag cannot reset a value the
    file set -- and a passed one must still win."""
    from reportfinder.__main__ import base_config

    path = tmp_path / "c.yaml"
    path.write_text("top_k: 3\n")
    base = base_config(path)

    assert base.with_overrides(top_k=7).top_k == 7
    assert base.with_overrides(top_k=None).top_k == 3


def test_malformed_yaml_is_reported_before_anything_loads(tmp_path, capsys):
    from reportfinder.__main__ import main

    path = tmp_path / "bad.yaml"
    path.write_text("top_k: [unclosed\n")
    assert main(["--config", str(path), "anything"]) == 2
    assert "not valid YAML" in capsys.readouterr().err


def test_a_non_mapping_config_document_is_refused(tmp_path, capsys):
    from reportfinder.__main__ import main

    path = tmp_path / "list.yaml"
    path.write_text("- a\n- b\n")
    assert main(["--config", str(path), "anything"]) == 2
    assert "must be a YAML mapping" in capsys.readouterr().err


def test_an_unknown_config_key_names_itself(tmp_path, capsys):
    from reportfinder.__main__ import main

    path = tmp_path / "typo.yaml"
    path.write_text("top_kk: 3\n")
    assert main(["--config", str(path), "anything"]) == 2
    assert "top_kk" in capsys.readouterr().err


def test_a_missing_config_file_is_reported_not_traced(tmp_path, capsys):
    from reportfinder.__main__ import main

    assert main(["--config", str(tmp_path / "absent.yaml"), "anything"]) == 2
    assert "cannot read --config" in capsys.readouterr().err


def test_the_phase2_config_selects_the_dual_file_estate():
    """The config exists to be runnable; these are the settings that make it so."""
    from reportfinder.config import from_mapping

    cfg = from_mapping(
        yaml.safe_load((CONFIG_DIR / "phase2_generators.yaml").read_text())
    )
    assert cfg.ingest_mode == "phase2_dual_file"
    assert cfg.corpus_granularity == "report_row"
    assert cfg.catalog_path.name == "Phase2_Report_Catalog_No_Fields.xlsx"
    assert cfg.field_dictionary_path.name == "Phase2_Field_Dictionary.xlsx"
    # No artifact fitted on the legacy estate may be pointed at this one.
    assert cfg.fusion.artifact_path is None
    assert cfg.decision.artifact_path is None


def test_the_phase2_workbooks_the_config_names_are_present():
    """Separate from the config contract above, which holds regardless.

    Both workbooks are tracked in git, so this should pass in a fresh clone --
    but it asserts a fact about the tree, not about the config, and a tree
    without them should say so rather than fail the config test."""
    from reportfinder.config import from_mapping

    cfg = from_mapping(
        yaml.safe_load((CONFIG_DIR / "phase2_generators.yaml").read_text())
    )
    for path in (cfg.catalog_path, cfg.field_dictionary_path):
        if not path.exists():
            pytest.skip(f"{path.name} absent from data/ -- blocked, not regressed")
    assert cfg.catalog_path.stat().st_size > 0
    assert cfg.field_dictionary_path.stat().st_size > 0
