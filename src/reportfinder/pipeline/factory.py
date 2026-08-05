"""Construction of the real pipeline. The only place models are loaded.

Tests build `SearchPipeline` directly with fakes and never import this module, which
is what keeps the unit suite offline and fast. Everything here is the production
wiring: real checkpoints, real indexes, real readiness enforcement.

Readiness is checked eagerly, at construction. A required component that is missing
or stale raises here rather than at the first query, so a bad deploy fails on
startup instead of silently serving a stale index.
"""

from __future__ import annotations

from ..auth import build_resolver
from ..corpus import ViewType, build_corpus_model
from ..generators import (
    Bm25fGenerator,
    DenseViewGenerator,
    PrototypeGenerator,
    SpladeGenerator,
)
from ..index.build import load_bundle
from ..index.encoders import SentenceTransformerEncoder
from ..index.splade import SpladeTorchEncoder
from ..query.intent import IntentParser
from ..retrieval.field_index import FieldIndex
from .decide import build_decision_head
from .fuse import build_fuser
from .onnx_scorer import build_scorer
from .orchestrator import SearchPipeline


def build_corpus_for(cfg, frame):
    source = (
        cfg.data_path if cfg.ingest_mode == "legacy_single_file" else cfg.catalog_path
    )
    return build_corpus_model(
        frame, ingest_mode=cfg.ingest_mode, source_file=str(source)
    )


def build_pipeline(cfg, frame, *, corpus=None, bundle=None) -> SearchPipeline:
    """Assemble the production pipeline for a built bundle."""
    corpus = corpus or build_corpus_for(cfg, frame)
    bundle = bundle or load_bundle(cfg, corpus)
    # Fail fast: a stale or missing required index must not reach a query.
    bundle.assert_ready()

    dense_encoder = SentenceTransformerEncoder(
        cfg.retrieval.dense.checkpoint,
        cfg.retrieval.dense.revision,
        max_length=cfg.retrieval.dense.max_length,
        batch_size=cfg.retrieval.dense.batch_size,
        trust_remote_code=cfg.security.trust_remote_code,
    )

    generators: list = []
    if cfg.retrieval.bm25f.enabled:
        generators.append(Bm25fGenerator.from_frame(frame, cfg, corpus))

    for view_type in ViewType:
        index = bundle.dense.get(view_type)
        # A degenerate or unbuilt view is simply not constructed; the bundle
        # already records why, and the shortlist redistributes its quota.
        if index is not None:
            generators.append(DenseViewGenerator(index, dense_encoder, corpus))

    if bundle.splade is not None:
        sparse_encoder = SpladeTorchEncoder(
            cfg.retrieval.splade.checkpoint,
            cfg.retrieval.splade.revision,
            max_length=cfg.retrieval.splade.max_length,
            batch_size=cfg.retrieval.splade.batch_size,
            min_term_weight=cfg.retrieval.splade.min_term_weight,
            trust_remote_code=cfg.security.trust_remote_code,
        )
        generators.append(SpladeGenerator(bundle.splade, sparse_encoder, corpus))

    if bundle.prototypes is not None:
        generators.append(
            PrototypeGenerator(bundle.prototypes, dense_encoder, corpus)
        )

    # Late interaction: implemented end to end (index format, MaxSim scoring,
    # generator, persistence, tests) but not served. Building the index needs a
    # production ColBERT token encoder, which is not implemented here, so enabling
    # it raises rather than quietly falling through to another dense search.
    if cfg.retrieval.late_interaction.enabled:
        raise NotImplementedError(
            "late-interaction serving requires a token encoder for "
            f"{cfg.retrieval.late_interaction.checkpoint}. The index format, MaxSim "
            "scoring, generator and persistence are implemented and tested; the "
            "production token encoder is not. Keep "
            "retrieval.late_interaction.enabled=false until one exists."
        )
    # Passed as None rather than as the disabled stand-in so the risk policy does
    # not treat it as an available widening option. The disabled state is recorded
    # in the bundle manifest and surfaces in telemetry as an active fallback.
    late_interaction = None

    # One place chooses the backend, so a config naming one cannot get the other.
    reranker = build_scorer(cfg)

    return SearchPipeline(
        cfg=cfg,
        corpus=corpus,
        generators=generators,
        resolver=build_resolver(cfg),
        intent_parser=IntentParser(frame, cfg),
        reranker=reranker,
        fuser=build_fuser(cfg),
        decision_head=build_decision_head(cfg),
        field_index=FieldIndex(frame),
        late_interaction_generator=late_interaction,
        bundle=bundle,
    )
