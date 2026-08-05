"""Index components: exactness, incremental rebuild, persistence, readiness.

The claims worth pinning are the ones that fail silently:

* top-k must equal brute force, or "exact retrieval" is just a comment;
* the universe must be applied before top-k, or k stops meaning k;
* incremental re-embedding must not reuse a vector whose text changed;
* a required component that is stale must block rather than serve.

Run: uv run pytest tests/test_index_components.py -v
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from reportfinder.auth import AuthorizedUniverse
from reportfinder.config import DEFAULT
from reportfinder.corpus import ViewType, build_corpus_model
from reportfinder.index import (
    BundleManifest,
    BundleNotReady,
    ComponentRecord,
    ComponentStatus,
    DenseViewIndex,
    FamilyPrototypeIndex,
    LateInteractionIndex,
    LateInteractionUnavailable,
    Requirement,
    SpladeIndex,
    component_signature,
    config_hash,
    maxsim,
    seed_prototypes_from_catalog,
)

from .fakes import FakeEncoder, FakeSparseEncoder, FakeTokenEncoder

TITLES = [
    "Attrition and Replacement Lag Analysis",
    "Headcount by Supervisory Organization",
    "Payroll Earnings Detail",
    "Learning Course Completion",
    "Worker Location Roster",
]


def _corpus(titles: list[str] | None = None):
    titles = titles or TITLES
    frame = pd.DataFrame([
        {
            "report_key": f"R{4 + i:04d}", "title_key": t.casefold(), "source_row": 4 + i,
            "title": t, "description": "", "category": "Worker Data",
            "data_source": "All Workers", "report_type": "Advanced",
            "prompts": ["Effective Date"], "fields": [f"{t.split()[0]} Field"],
            "tags": "", "area_where_used": "", "worklet": "", "chart_type": "",
            "landing_page": "", "worklet_landing_pages": "", "shared": "Yes",
        }
        for i, t in enumerate(titles)
    ])
    return build_corpus_model(frame, ingest_mode="legacy_single_file", source_file="f")


def _dense(corpus, view=ViewType.IDENTITY, encoder=None, previous=None):
    encoder = encoder or FakeEncoder()
    return DenseViewIndex.build(
        view_type=view,
        instance_ids=corpus.instance_ids,
        texts=corpus.view_texts(view),
        hashes=corpus.view_hashes(view),
        encoder=encoder,
        previous=previous,
    )


# --- dense: exactness --------------------------------------------------------


def test_topk_matches_brute_force():
    """`torch.topk` over the score vector must agree with a plain argsort."""
    corpus = _corpus()
    index = _dense(corpus)
    query = FakeEncoder().encode_queries(["attrition and replacement"])[0]

    got = index.search(query, k=3)
    scores = index.scores(query)
    expected = np.argsort(-scores, kind="stable")[:3]

    assert [p for p, _ in got] == expected.tolist()
    for position, score in got:
        assert score == pytest.approx(float(scores[position]), abs=1e-6)


def test_scores_are_cosines_of_normalized_vectors():
    index = _dense(_corpus())
    query = FakeEncoder().encode_queries(["headcount"])[0]
    scores = index.scores(query)
    assert np.all(scores <= 1.0 + 1e-6) and np.all(scores >= -1.0 - 1e-6)


def test_universe_is_applied_before_topk_so_k_means_k():
    """Filtering after top-k would return fewer than k authorized results."""
    corpus = _corpus()
    index = _dense(corpus)
    query = FakeEncoder().encode_queries(["attrition replacement lag"])[0]

    unrestricted = index.search(query, k=2)
    best = unrestricted[0][0]

    mask = np.ones(len(corpus), dtype=bool)
    mask[best] = False
    universe = AuthorizedUniverse(mask=mask, resolver="t", acl_source="t")

    restricted = index.search(query, k=2, universe=universe)
    assert len(restricted) == 2, "k must still yield k authorized results"
    assert best not in [p for p, _ in restricted]


def test_empty_universe_returns_nothing():
    corpus = _corpus()
    index = _dense(corpus)
    universe = AuthorizedUniverse(
        mask=np.zeros(len(corpus), dtype=bool), resolver="t", acl_source="t"
    )
    query = FakeEncoder().encode_queries(["headcount"])[0]
    assert index.search(query, k=5, universe=universe) == []


def test_empty_view_text_scores_zero_rather_than_pointing_somewhere():
    corpus = _corpus()
    index = _dense(corpus, view=ViewType.PURPOSE)  # descriptions are all blank here
    query = FakeEncoder().encode_queries(["anything at all"])[0]
    assert np.allclose(index.scores(query), 0.0)


# --- dense: incremental rebuild ---------------------------------------------


def test_unchanged_rows_are_reused_not_reencoded():
    corpus = _corpus()
    first = _dense(corpus)
    assert first.stats.reencoded_rows == len(corpus)

    encoder = FakeEncoder()
    second = _dense(corpus, encoder=encoder, previous=first)
    assert second.stats.reencoded_rows == 0
    assert second.stats.reused_rows == len(corpus)
    # Not merely called with an empty batch -- not called at all.
    assert encoder.document_calls == []
    np.testing.assert_allclose(second.vectors, first.vectors)


def test_only_changed_rows_are_reencoded():
    first = _dense(_corpus())
    changed = _corpus([*TITLES[:2], "Payroll Earnings Summary", *TITLES[3:]])

    encoder = FakeEncoder()
    second = _dense(changed, encoder=encoder, previous=first)
    assert second.stats.reencoded_rows == 1
    assert second.stats.reused_rows == 4
    assert encoder.document_calls[0] == [changed.views["R0006"][ViewType.IDENTITY].text]


def test_a_changed_row_never_reuses_its_stale_vector():
    """Reuse is keyed on (instance, content hash), not instance alone."""
    first = _dense(_corpus())
    changed = _corpus([*TITLES[:2], "Completely Different Report", *TITLES[3:]])
    second = _dense(changed, previous=first)
    assert not np.allclose(second.vectors[2], first.vectors[2])


def test_changing_the_checkpoint_invalidates_every_row():
    first = _dense(_corpus())
    other = FakeEncoder(name="other-model", revision="v2")
    second = _dense(_corpus(), encoder=other, previous=first)
    assert second.stats.reencoded_rows == len(second.instance_ids)


# --- persistence -------------------------------------------------------------


def test_dense_index_round_trips(tmp_path):
    index = _dense(_corpus())
    index.save(tmp_path / "identity")
    loaded = DenseViewIndex.load(tmp_path / "identity")

    np.testing.assert_allclose(loaded.vectors, index.vectors)
    assert loaded.instance_ids == index.instance_ids
    assert loaded.hashes == index.hashes
    assert loaded.view_type is index.view_type
    assert loaded.model_id == index.model_id and loaded.revision == index.revision


def test_dense_index_rejects_a_foreign_schema_version(tmp_path):
    index = _dense(_corpus())
    index.save(tmp_path / "identity")
    meta = tmp_path / "identity" / "meta.json"
    meta.write_text(meta.read_text().replace('"schema_version": "1"', '"schema_version": "99"'))
    with pytest.raises(ValueError, match="schema version"):
        DenseViewIndex.load(tmp_path / "identity")


# --- splade ------------------------------------------------------------------


def _splade(corpus):
    return SpladeIndex.build(
        instance_ids=corpus.instance_ids,
        texts=corpus.view_texts(ViewType.IDENTITY),
        hashes=corpus.view_hashes(ViewType.IDENTITY),
        encoder=FakeSparseEncoder(),
    )


def test_splade_ranks_by_sparse_dot_product():
    corpus = _corpus()
    index = _splade(corpus)
    query = FakeSparseEncoder().encode(["attrition replacement"])
    hits = index.search(query, k=3)
    assert hits, "expected sparse matches"
    assert hits[0][0] == 0
    assert [h[1] for h in hits] == sorted((h[1] for h in hits), reverse=True)


def test_splade_zero_score_documents_are_excluded_not_ranked():
    """A zero sparse score means no shared term -- no evidence, not weak evidence."""
    corpus = _corpus()
    index = _splade(corpus)
    query = FakeSparseEncoder().encode(["zzzz nonexistent vocabulary"])
    assert index.search(query, k=5) == []


def test_splade_respects_the_universe():
    corpus = _corpus()
    index = _splade(corpus)
    mask = np.zeros(len(corpus), dtype=bool)
    mask[1] = True
    universe = AuthorizedUniverse(mask=mask, resolver="t", acl_source="t")
    hits = index.search(
        FakeSparseEncoder().encode(["attrition replacement"]), k=5, universe=universe
    )
    assert all(p == 1 for p, _ in hits)


def test_splade_rejects_a_query_from_a_different_vocabulary():
    index = _splade(_corpus())
    other = FakeSparseEncoder(vocab_size=64).encode(["attrition"])
    with pytest.raises(ValueError, match="different checkpoint"):
        index.search(other, k=3)


def test_splade_round_trips(tmp_path):
    index = _splade(_corpus())
    index.save(tmp_path / "splade")
    loaded = SpladeIndex.load(tmp_path / "splade")
    assert loaded.instance_ids == index.instance_ids
    assert (loaded.postings != index.postings).nnz == 0


# --- prototypes --------------------------------------------------------------


def test_seed_prototypes_carry_honest_provenance():
    corpus = _corpus()
    prototypes = seed_prototypes_from_catalog(corpus)
    assert len(prototypes) == len(corpus.families)
    assert all(p.source.value == "catalog_seed" for p in prototypes)
    assert all(p.validation_status.value == "unreviewed" for p in prototypes)
    # Never usable as catalog evidence.
    assert all(not p.is_authoritative_catalog_text for p in prototypes)


def test_prototype_index_returns_families_ranked_by_best_match():
    corpus = _corpus()
    index = FamilyPrototypeIndex.build(seed_prototypes_from_catalog(corpus), FakeEncoder())
    query = FakeEncoder().encode_queries(["attrition replacement lag"])[0]
    families = index.search_families(query, k=3)
    assert families
    assert families[0][0] == "attrition and replacement lag analysis"
    assert [s for _, s in families] == sorted((s for _, s in families), reverse=True)


def test_a_family_with_more_prototypes_does_not_win_by_volume():
    """Best-match, not sum: otherwise verbose families dominate every query."""
    from reportfinder.index.prototypes import (
        PrototypeSource,
        QueryPrototype,
        ValidationStatus,
    )

    def proto(pid, family, text):
        return QueryPrototype(pid, family, text, PrototypeSource.CATALOG_SEED,
                              ValidationStatus.UNREVIEWED, "r", "c")

    index = FamilyPrototypeIndex.build([
        proto("p1", "chatty", "headcount"),
        proto("p2", "chatty", "headcount"),
        proto("p3", "chatty", "headcount"),
        proto("p4", "precise", "headcount by supervisory organization"),
    ], FakeEncoder())

    query = FakeEncoder().encode_queries(["headcount by supervisory organization"])[0]
    assert index.search_families(query, k=2)[0][0] == "precise"


def test_rejected_prototypes_are_not_retrieved():
    from reportfinder.index.prototypes import (
        PrototypeSource,
        QueryPrototype,
        ValidationStatus,
    )

    index = FamilyPrototypeIndex.build([
        QueryPrototype("p1", "bad", "headcount", PrototypeSource.GENERATED,
                       ValidationStatus.REJECTED, "r", "c"),
    ], FakeEncoder())
    assert index.search_families(FakeEncoder().encode_queries(["headcount"])[0], k=5) == []


def test_prototype_index_round_trips(tmp_path):
    corpus = _corpus()
    index = FamilyPrototypeIndex.build(seed_prototypes_from_catalog(corpus), FakeEncoder())
    index.save(tmp_path / "prototypes")
    loaded = FamilyPrototypeIndex.load(tmp_path / "prototypes")
    assert [p.prototype_id for p in loaded.prototypes] == [
        p.prototype_id for p in index.prototypes
    ]
    assert loaded.prototypes[0].source is index.prototypes[0].source


# --- late interaction --------------------------------------------------------


def test_maxsim_sums_the_best_document_token_per_query_token():
    query, qmask = FakeTokenEncoder().encode_queries(["headcount organization"])
    docs, dmask = FakeTokenEncoder().encode_documents([
        "headcount organization", "payroll earnings",
    ])
    scores = maxsim(query[0], qmask[0], docs, dmask)
    assert scores[0] > scores[1]
    # Two query tokens, each capped at cosine 1.
    assert scores[0] <= 2.0 + 1e-5


def test_maxsim_ignores_padding_on_both_sides():
    encoder = FakeTokenEncoder(max_tokens=8)
    query, qmask = encoder.encode_queries(["headcount"])
    docs, dmask = encoder.encode_documents(["headcount"])
    assert qmask[0].sum() == 1 and dmask[0].sum() == 1
    # One real query token means the score cannot exceed one cosine.
    assert maxsim(query[0], qmask[0], docs, dmask)[0] <= 1.0 + 1e-5


def test_late_interaction_index_round_trips(tmp_path):
    corpus = _corpus()
    index = LateInteractionIndex.build(
        instance_ids=corpus.instance_ids,
        texts=corpus.view_texts(ViewType.IDENTITY),
        encoder=FakeTokenEncoder(),
    )
    index.save(tmp_path / "li")
    loaded = LateInteractionIndex.load(tmp_path / "li")
    np.testing.assert_allclose(loaded.doc_vectors, index.doc_vectors)
    assert loaded.instance_ids == index.instance_ids


def test_missing_late_interaction_index_says_so_rather_than_substituting(tmp_path):
    """Silence here would let another retriever masquerade as late interaction."""
    with pytest.raises(LateInteractionUnavailable, match="reportfinder-bundle build"):
        LateInteractionIndex.load(tmp_path / "absent")


# --- bundle manifest ---------------------------------------------------------


def _manifest(**components) -> BundleManifest:
    return BundleManifest(
        bundle_version="b-test", catalog_version="cat1",
        corpus_content_hash="hash1", ingest_mode="legacy_single_file",
        corpus_granularity="report_row", source_files=[], instance_count=5,
        family_count=5, config_hash="cfg1", code_version="test",
        created_at="2026-08-04T00:00:00Z", components=components,
    )


def _record(name, requirement, status, fallback=None):
    return ComponentRecord(name=name, requirement=requirement, status=status,
                           fallback=fallback)


def test_required_missing_component_blocks_with_the_exact_build_command():
    manifest = _manifest(**{
        "views.identity": _record("views.identity", Requirement.REQUIRED,
                                  ComponentStatus.ABSENT, fallback="blocks_readiness"),
    })
    with pytest.raises(BundleNotReady) as excinfo:
        manifest.assert_ready()
    assert excinfo.value.blocking == ["views.identity"]
    assert "reportfinder-bundle build --component views.identity" in str(excinfo.value)


def test_optional_missing_component_is_degraded_and_discloses_its_fallback():
    manifest = _manifest(**{
        "views.identity": _record("views.identity", Requirement.REQUIRED,
                                  ComponentStatus.READY),
        "late_interaction": _record("late_interaction", Requirement.OPTIONAL,
                                    ComponentStatus.BUILT_DISABLED,
                                    fallback="generator_not_constructed"),
    })
    manifest.assert_ready()
    readiness = manifest.readiness()
    assert readiness["ready"] is True
    assert readiness["degraded"] == ["late_interaction"]
    assert readiness["active_fallbacks"] == [
        "late_interaction:generator_not_constructed"
    ]


def test_a_non_ready_component_without_a_fallback_is_a_structural_error():
    """Forces every new component to decide what happens when it is absent."""
    manifest = _manifest(**{
        "prototypes": _record("prototypes", Requirement.OPTIONAL, ComponentStatus.ABSENT),
    })
    with pytest.raises(ValueError, match="declare no fallback"):
        manifest.validate_structure()


def test_a_corpus_change_marks_components_stale_and_blocks():
    """A stale index returns plausible results, so it must never be served."""
    manifest = _manifest(**{
        "views.identity": _record("views.identity", Requirement.REQUIRED,
                                  ComponentStatus.READY),
        "prototypes": _record("prototypes", Requirement.OPTIONAL, ComponentStatus.READY),
    })
    manifest.mark_stale_components("a-different-corpus-hash")
    assert manifest.components["views.identity"].status is ComponentStatus.STALE
    assert manifest.components["prototypes"].fallback == "component_disabled"
    with pytest.raises(BundleNotReady):
        manifest.assert_ready()


def test_an_unchanged_corpus_leaves_components_ready():
    manifest = _manifest(**{
        "views.identity": _record("views.identity", Requirement.REQUIRED,
                                  ComponentStatus.READY),
    })
    manifest.mark_stale_components("hash1")
    manifest.assert_ready()


def test_manifest_round_trips(tmp_path):
    manifest = _manifest(**{
        "splade": _record("splade", Requirement.REQUIRED, ComponentStatus.READY),
        "fusion_model": _record("fusion_model", Requirement.OPTIONAL,
                                ComponentStatus.ABSENT, fallback="rrf"),
    })
    manifest.save(tmp_path / "manifest.json")
    loaded = BundleManifest.load(tmp_path / "manifest.json")
    assert loaded.corpus_content_hash == "hash1"
    assert loaded.components["fusion_model"].fallback == "rrf"
    assert loaded.components["splade"].status is ComponentStatus.READY


def test_component_signatures_are_scoped_so_one_model_change_is_local():
    common = {"schema_version": "1", "corpus_content_hash": "h", "params": {}}
    dense = component_signature(model_id="bge", model_revision="r1", **common)
    dense_new = component_signature(model_id="bge", model_revision="r2", **common)
    splade = component_signature(model_id="splade", model_revision="s1", **common)
    assert dense != dense_new
    assert splade == component_signature(model_id="splade", model_revision="s1", **common)


def test_config_hash_tracks_retrieval_settings_but_not_display_settings():
    base = config_hash(DEFAULT)
    assert config_hash(DEFAULT.with_path_overrides({"retrieval.dense.k_per_view": 80})) != base
    assert config_hash(DEFAULT.with_path_overrides({"shortlist.standard_rerank_depth": 99})) != base
    # `top_k` changes how many results are shown, not what was retrieved.
    assert config_hash(DEFAULT.with_overrides(top_k=42)) == base


# --- tie determinism ---------------------------------------------------------


def test_dense_search_breaks_ties_deterministically():
    """Exact ties are ordinary here -- an empty view row is stored as a zero
    vector and scores 0.0 against every query -- and `torch.topk` gives no
    guarantee about which of several tied rows it returns. That made the identity
    of a retrieved candidate depend on a kernel."""
    vectors = np.zeros((6, 3), dtype=np.float32)
    vectors[:, 0] = 1.0  # every row identical => every score identical
    index = DenseViewIndex(
        view_type=ViewType.IDENTITY,
        instance_ids=tuple(f"R{i:04d}" for i in range(6)),
        hashes=tuple("h" for _ in range(6)),
        vectors=vectors,
        model_id="fake",
        revision="v1",
    )
    query = np.array([1.0, 0.0, 0.0], dtype=np.float32)

    first = index.search(query, k=3)
    assert [p for p, _ in first] == [0, 1, 2], "ties must resolve to row order"
    for _ in range(4):
        assert index.search(query, k=3) == first, "repeated calls must agree"


def test_dense_search_still_applies_the_universe_before_the_cut():
    """Stable sorting must not have reordered the mask-then-rank invariant: an
    unauthorized row cannot consume one of the k slots."""
    from reportfinder.auth.universe import AuthorizedUniverse

    vectors = np.zeros((4, 2), dtype=np.float32)
    vectors[:, 0] = [0.9, 0.8, 0.7, 0.6]
    index = DenseViewIndex(
        view_type=ViewType.IDENTITY,
        instance_ids=("R0", "R1", "R2", "R3"),
        hashes=("h",) * 4,
        vectors=vectors,
        model_id="fake",
        revision="v1",
    )
    universe = AuthorizedUniverse(
        mask=np.array([True, False, True, True]),
        resolver="test",
        acl_source="test",
    )
    hits = index.search(np.array([1.0, 0.0], dtype=np.float32), k=2, universe=universe)
    assert [p for p, _ in hits] == [0, 2], "the masked row must not take a slot"


def test_risk_coverage_is_stable_under_ties():
    from reportfinder.evaluation.metrics import risk_coverage

    scores = [0.5] * 6
    correct = [1, 0, 1, 0, 1, 0]
    first = risk_coverage(scores, correct)
    for _ in range(4):
        assert risk_coverage(scores, correct) == first


def test_discordant_pairs_ignores_ties_but_catches_inversions():
    """The parity gate used to diff two sorted index lists, so two backends
    breaking a genuine tie differently counted as an ordering disagreement."""
    from reportfinder.evaluation.metrics import discordant_pairs

    assert discordant_pairs([1.0, 1.0, 2.0], [1.0, 1.0, 2.0]) == 0
    # Same values, tie resolved "differently" -- not a disagreement.
    assert discordant_pairs([5.0, 5.0], [5.0, 5.0]) == 0
    # One side ties, the other does not: still no contradiction.
    assert discordant_pairs([1.0, 1.0], [1.0, 2.0]) == 0
    # A real inversion is caught.
    assert discordant_pairs([1.0, 2.0], [2.0, 1.0]) == 1


def test_query_encoding_survives_a_batch_larger_than_its_cache():
    """The cache is an optimization, so what `encode_queries` returns must not
    depend on which entries survived eviction.

    A batch bigger than `query_cache_size` evicted its own earliest entries on
    the way in, and the result was then reassembled by reading them back out --
    raising KeyError. A first-time prototype build does exactly this: 4,299
    prototypes against a 256-entry cache.
    """
    from reportfinder.index.encoders import SentenceTransformerEncoder

    class _Fake(SentenceTransformerEncoder):
        """The real caching encoder with only the model call replaced."""

        def __init__(self):
            super().__init__("fake", "v1", query_cache_size=4)
            self._dim = 2

        def _encode(self, texts, prefix=""):
            return np.array([[float(len(t)), 1.0] for t in texts], dtype=np.float32)

    encoder = _Fake()
    texts = [f"query number {i}" for i in range(20)]
    vectors = encoder.encode_queries(texts)

    assert vectors.shape == (20, 2)
    assert [v[0] for v in vectors] == [float(len(t)) for t in texts]
    # Repeat calls still agree, now served partly from the (small) cache.
    assert np.array_equal(encoder.encode_queries(texts), vectors)
