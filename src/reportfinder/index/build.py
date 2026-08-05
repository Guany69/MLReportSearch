"""Building and loading a bundle.

Index construction is a deliberate, separate step -- never something `load_or_build`
or a Streamlit start does implicitly. Encoding 4000 documents with SPLADE takes
around 100 seconds on this hardware; folding that into process start would make
every test run and every app restart pay it.

Components are built independently and signed independently, so changing the dense
checkpoint rebuilds four view indexes and leaves the SPLADE postings alone.
"""

from __future__ import annotations

import datetime as _datetime
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

from .. import __version__
from ..corpus import ALL_VIEW_TYPES, CorpusModel, ViewType
from .bundle import (
    BundleManifest,
    ComponentRecord,
    ComponentStatus,
    Requirement,
    component_signature,
    config_hash,
    index_config_hash,
)
from .dense_views import INDEX_SCHEMA_VERSION, DenseViewIndex
from .encoders import SentenceTransformerEncoder, TextEncoder
from .late_interaction import LateInteractionIndex
from .prototypes import (
    PROTOTYPE_SCHEMA_VERSION,
    FamilyPrototypeIndex,
    seed_prototypes_from_catalog,
)
from .splade import SPLADE_SCHEMA_VERSION, SpladeIndex, SpladeTorchEncoder

# Component names, also used as directory names and manifest keys.
DENSE_COMPONENTS = tuple(f"views.{v.value}" for v in ALL_VIEW_TYPES)
ALL_COMPONENTS = (*DENSE_COMPONENTS, "splade", "prototypes", "late_interaction")


def _now() -> str:
    return _datetime.datetime.now(_datetime.UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def bundle_id(corpus: CorpusModel, cfg) -> str:
    """Deterministic identity for a set of built indexes.

    Keyed on the corpus content and the *index* configuration only. Serving knobs
    -- shortlist depths, risk thresholds, fusion weights -- are recorded in the
    manifest but deliberately excluded here, so tuning one does not invalidate
    4000 encoded documents.
    """
    return f"b-{corpus.content_hash}-{index_config_hash(cfg)[:8]}"


@dataclass
class LoadedBundle:
    """A bundle's manifest plus whatever components loaded successfully."""

    manifest: BundleManifest
    root: Path
    dense: dict[ViewType, DenseViewIndex] = field(default_factory=dict)
    splade: SpladeIndex | None = None
    prototypes: FamilyPrototypeIndex | None = None
    late_interaction: LateInteractionIndex | None = None
    # `config_hash` of the configuration this bundle was *loaded* with, against
    # the manifest's record of the one it was *built* with.
    runtime_config_hash: str = ""

    def assert_ready(self) -> None:
        self.manifest.assert_ready()

    @property
    def config_drift(self) -> bool:
        """Whether serving configuration has moved since the bundle was built.

        `bundle_id` intentionally keys on corpus content plus *index* config only,
        so tuning a shortlist depth or a risk threshold does not invalidate 4,000
        encoded documents. The cost of that is that the same bundle id can be
        served under materially different retrieval behaviour, with nothing
        saying so. This is that something. It is a disclosure, not an error:
        the stored vectors remain valid, but a result recorded under one runtime
        hash cannot be assumed reproducible under another.
        """
        stored = self.manifest.config_hash
        return bool(stored and self.runtime_config_hash and stored != self.runtime_config_hash)

    @property
    def active_fallbacks(self) -> list[str]:
        return self.manifest.active_fallbacks()


def _requirement(name: str, cfg) -> Requirement:
    return (
        Requirement.REQUIRED
        if name in cfg.bundle.required
        else Requirement.OPTIONAL
    )


def _dense_encoder(cfg) -> TextEncoder:
    return SentenceTransformerEncoder(
        cfg.retrieval.dense.checkpoint,
        cfg.retrieval.dense.revision,
        max_length=cfg.retrieval.dense.max_length,
        batch_size=cfg.retrieval.dense.batch_size,
        trust_remote_code=cfg.security.trust_remote_code,
    )


def _artifact_provenance(artifact_path) -> dict[str, object]:
    """What a served learned artifact was approved on.

    Read from the `.json` sidecar rather than the `.pt` so the manifest build
    stays free of torch. Absent or unreadable yields nothing rather than raising:
    the manifest records state, and a missing sidecar is itself a state worth
    seeing rather than a build failure.
    """
    if artifact_path is None:
        return {}
    sidecar = Path(artifact_path).with_suffix(".json")
    if not sidecar.exists():
        return {"provenance": "no sidecar next to the artifact"}
    try:
        payload = json.loads(sidecar.read_text())
    except json.JSONDecodeError:
        return {"provenance": "sidecar is not readable JSON"}
    approval = payload.get("approval") or {}
    return {
        "weights_sha256": payload.get("weights_sha256"),
        "approved": bool(payload.get("approved", False)),
        "approval_basis": approval.get("basis"),
        "label_source": payload.get("training_label_source"),
        # Stated here so a reader of the manifest cannot mistake an approved
        # model for a human-validated one.
        "human_validated": bool(payload.get("human_validated", False)),
        "primary_metric": approval.get("primary_metric"),
    }


def build_bundle(
    cfg,
    corpus: CorpusModel,
    *,
    encoder: TextEncoder | None = None,
    sparse_encoder=None,
    token_encoder=None,
    only: Iterable[str] | None = None,
    verbose: bool = False,
) -> BundleManifest:
    """Build (or incrementally refresh) the configured components."""
    root = Path(cfg.bundle.root) / bundle_id(corpus, cfg)
    root.mkdir(parents=True, exist_ok=True)
    selected = set(only) if only else set(ALL_COMPONENTS)

    manifest_path = root / "manifest.json"
    previous = BundleManifest.load(manifest_path) if manifest_path.exists() else None
    components: dict[str, ComponentRecord] = dict(previous.components) if previous else {}

    def log(message: str) -> None:
        if verbose:
            print(message)

    # -- dense views -------------------------------------------------------
    dense_cfg = cfg.retrieval.dense
    lazy_encoder = encoder
    for view_type in ALL_VIEW_TYPES:
        name = f"views.{view_type.value}"
        if name not in selected:
            continue
        if view_type.value not in dense_cfg.views:
            components[name] = ComponentRecord(
                name=name, requirement=_requirement(name, cfg),
                status=ComponentStatus.BUILT_DISABLED,
                fallback="generator_not_constructed",
                detail={"reason": "view not listed in retrieval.dense.views"},
            )
            continue

        if lazy_encoder is None:
            lazy_encoder = _dense_encoder(cfg)

        signature = component_signature(
            schema_version=INDEX_SCHEMA_VERSION,
            corpus_content_hash=corpus.content_hash,
            model_id=lazy_encoder.name, model_revision=lazy_encoder.revision,
            params={"view": view_type.value, "max_length": dense_cfg.max_length},
        )
        directory = root / "views" / view_type.value
        existing = components.get(name)
        if (
            existing is not None
            and existing.signature == signature
            and existing.is_ready
            and (directory / "meta.json").exists()
        ):
            log(f"  {name}: up to date")
            continue

        stale = (
            DenseViewIndex.load(directory)
            if (directory / "meta.json").exists()
            else None
        )
        log(f"  {name}: encoding")
        index = DenseViewIndex.build(
            view_type=view_type,
            instance_ids=corpus.instance_ids,
            texts=corpus.view_texts(view_type),
            hashes=corpus.view_hashes(view_type),
            encoder=lazy_encoder,
            previous=stale,
        )
        index.save(directory)

        stats = index.stats
        degenerate = (
            stats is not None
            and stats.distinct_text_ratio < dense_cfg.min_distinct_text_ratio
        )
        components[name] = ComponentRecord(
            name=name, requirement=_requirement(name, cfg),
            # A near-constant view returns the same arbitrary top-k for every
            # query and would consume shortlist slots that could hold real
            # candidates. Recorded, not silently served.
            status=(
                ComponentStatus.DEGENERATE_LOW_ENTROPY if degenerate
                else ComponentStatus.READY
            ),
            signature=signature, model_id=index.model_id,
            model_revision=index.revision, rows=len(index),
            fallback="generator_disabled" if degenerate else None,
            detail=dict(stats.__dict__) if stats else {},
        )
        if degenerate and stats is not None:
            log(
                f"  {name}: degenerate (distinct_text_ratio="
                f"{stats.distinct_text_ratio}); generator will not be constructed"
            )

    # -- splade ------------------------------------------------------------
    if "splade" in selected:
        splade_cfg = cfg.retrieval.splade
        if not splade_cfg.enabled:
            components["splade"] = ComponentRecord(
                name="splade", requirement=_requirement("splade", cfg),
                status=ComponentStatus.BUILT_DISABLED,
                fallback="generator_not_constructed",
                detail={"reason": "retrieval.splade.enabled=false"},
            )
        else:
            sparse = sparse_encoder or SpladeTorchEncoder(
                splade_cfg.checkpoint, splade_cfg.revision,
                max_length=splade_cfg.max_length, batch_size=splade_cfg.batch_size,
                min_term_weight=splade_cfg.min_term_weight,
                trust_remote_code=cfg.security.trust_remote_code,
            )
            signature = component_signature(
                schema_version=SPLADE_SCHEMA_VERSION,
                corpus_content_hash=corpus.content_hash,
                model_id=sparse.name, model_revision=sparse.revision,
                params={"max_length": splade_cfg.max_length,
                        "min_term_weight": splade_cfg.min_term_weight},
            )
            existing = components.get("splade")
            directory = root / "splade"
            if (
                existing is not None and existing.signature == signature
                and existing.is_ready and (directory / "meta.json").exists()
            ):
                log("  splade: up to date")
            else:
                log("  splade: encoding (this is the slow one)")
                # SPLADE reads the whole report, not one view: its value is term
                # expansion over everything the report says.
                texts = [
                    " ".join(
                        corpus.views[i][v].text for v in ALL_VIEW_TYPES
                    ).strip()
                    for i in corpus.instance_ids
                ]
                splade_index = SpladeIndex.build(
                    instance_ids=corpus.instance_ids, texts=texts,
                    hashes=[corpus.views[i][ViewType.IDENTITY].content_hash
                            for i in corpus.instance_ids],
                    encoder=sparse,
                )
                splade_index.save(directory)
                components["splade"] = ComponentRecord(
                    name="splade", requirement=_requirement("splade", cfg),
                    status=ComponentStatus.READY, signature=signature,
                    model_id=splade_index.model_id,
                    model_revision=splade_index.revision,
                    rows=len(index),
                    detail=dict(index.stats.__dict__) if index.stats else {},
                )

    # -- prototypes --------------------------------------------------------
    if "prototypes" in selected:
        proto_cfg = cfg.retrieval.prototypes
        if not proto_cfg.enabled:
            components["prototypes"] = ComponentRecord(
                name="prototypes", requirement=_requirement("prototypes", cfg),
                status=ComponentStatus.BUILT_DISABLED,
                fallback="generator_not_constructed",
                detail={"reason": "retrieval.prototypes.enabled=false"},
            )
        else:
            if lazy_encoder is None:
                lazy_encoder = encoder or _dense_encoder(cfg)
            prototypes = seed_prototypes_from_catalog(corpus)
            signature = component_signature(
                schema_version=PROTOTYPE_SCHEMA_VERSION,
                corpus_content_hash=corpus.content_hash,
                model_id=lazy_encoder.name, model_revision=lazy_encoder.revision,
                params={"prototype_count": len(prototypes)},
            )
            existing = components.get("prototypes")
            directory = root / "prototypes"
            if (
                existing is not None and existing.signature == signature
                and existing.is_ready and (directory / "meta.json").exists()
            ):
                log("  prototypes: up to date")
            else:
                log(f"  prototypes: encoding {len(prototypes)} seeds")
                prototype_index = FamilyPrototypeIndex.build(prototypes, lazy_encoder)
                prototype_index.save(directory)
                components["prototypes"] = ComponentRecord(
                    name="prototypes", requirement=_requirement("prototypes", cfg),
                    status=ComponentStatus.READY, signature=signature,
                    model_id=prototype_index.model_id,
                    model_revision=prototype_index.revision,
                    rows=len(prototype_index),
                    # Provenance travels with the component: these are catalog-derived
                    # seeds, not observed user language, and nothing may present them
                    # as production behaviour.
                    detail={
                        "sources": sorted({p.source.value for p in prototypes}),
                        "validation_statuses": sorted(
                            {p.validation_status.value for p in prototypes}
                        ),
                        "families": len({p.family_id for p in prototypes}),
                    },
                )

    # -- late interaction --------------------------------------------------
    if "late_interaction" in selected:
        li_cfg = cfg.retrieval.late_interaction
        if not li_cfg.enabled and token_encoder is None:
            components["late_interaction"] = ComponentRecord(
                name="late_interaction",
                requirement=_requirement("late_interaction", cfg),
                status=ComponentStatus.BUILT_DISABLED,
                fallback="generator_not_constructed",
                detail={"reason": li_cfg.disabled_reason,
                        "checkpoint": li_cfg.checkpoint,
                        "revision": li_cfg.revision},
            )
        elif token_encoder is not None:
            directory = root / "late_interaction"
            texts = [
                " ".join(corpus.views[i][v].text for v in ALL_VIEW_TYPES).strip()
                for i in corpus.instance_ids
            ]
            late_index = LateInteractionIndex.build(
                instance_ids=corpus.instance_ids, texts=texts, encoder=token_encoder,
            )
            late_index.save(directory)
            components["late_interaction"] = ComponentRecord(
                name="late_interaction",
                requirement=_requirement("late_interaction", cfg),
                status=ComponentStatus.READY,
                # `late_index`, not `index` -- `index` is the leftover loop
                # variable from the dense-view loop above, so this recorded the
                # last dense view's model and row count, and raised NameError
                # under `--component late_interaction` where that loop never ran.
                model_id=late_index.model_id,
                model_revision=late_index.revision,
                rows=len(late_index),
            )
        else:
            components["late_interaction"] = ComponentRecord(
                name="late_interaction",
                requirement=_requirement("late_interaction", cfg),
                status=ComponentStatus.ABSENT,
                fallback="generator_not_constructed",
                detail={"reason": "no token encoder configured"},
            )

    # The cross-encoder is a scorer, not a stored index; it is recorded so the
    # manifest names every model that can affect a result. `backend` matters
    # because the two runtimes are only interchangeable where the parity gate
    # passed, and that is host-specific.
    components["cross_encoder"] = ComponentRecord(
        name="cross_encoder", requirement=_requirement("cross_encoder", cfg),
        status=ComponentStatus.READY if cfg.rerank.enabled else ComponentStatus.BUILT_DISABLED,
        model_id=cfg.rerank.checkpoint, model_revision=cfg.rerank.revision,
        fallback=None if cfg.rerank.enabled else "fusion_ordering_without_rerank",
        detail={"max_length": cfg.rerank.max_length,
                "backend": cfg.rerank.backend},
    )
    # Learned components ship absent, with their fallbacks named. When one does
    # serve, the manifest records *what it was approved on* -- not just that it
    # exists. A bundle that says "fusion_model: READY" and nothing else invites
    # the reader to assume the model was validated against something real.
    components["fusion_model"] = ComponentRecord(
        name="fusion_model", requirement=Requirement.OPTIONAL,
        status=ComponentStatus.READY if cfg.fusion.artifact_path else ComponentStatus.ABSENT,
        fallback=None if cfg.fusion.artifact_path else cfg.fusion.fallback,
        detail={"architecture": cfg.fusion.architecture,
                **_artifact_provenance(cfg.fusion.artifact_path)},
    )
    components["decision_model"] = ComponentRecord(
        name="decision_model", requirement=Requirement.OPTIONAL,
        status=ComponentStatus.READY if cfg.decision.artifact_path else ComponentStatus.ABSENT,
        fallback=None if cfg.decision.artifact_path else cfg.decision.fallback,
        detail={"architecture": cfg.decision.architecture,
                "calibrated": bool(cfg.decision.calibration_path),
                **_artifact_provenance(cfg.decision.artifact_path)},
    )

    source = Path(corpus.source_file)
    manifest = BundleManifest(
        bundle_version=bundle_id(corpus, cfg),
        catalog_version=corpus.catalog_version,
        corpus_content_hash=corpus.content_hash,
        ingest_mode=corpus.ingest_mode,
        corpus_granularity="report_row",
        source_files=[{
            "path": corpus.source_file,
            "sha256_16": corpus.catalog_version,
            "size": source.stat().st_size if source.exists() else None,
        }],
        instance_count=len(corpus),
        family_count=len(corpus.families),
        config_hash=config_hash(cfg),
        code_version=__version__,
        created_at=_now(),
        components=components,
    )
    manifest.validate_structure()
    manifest.save(manifest_path)
    return manifest


def load_bundle(cfg, corpus: CorpusModel) -> LoadedBundle:
    """Load a built bundle, marking anything built against a different corpus stale."""
    root = Path(cfg.bundle.root) / bundle_id(corpus, cfg)
    manifest_path = root / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"No bundle at {root}. Build one with:\n"
            f"  uv run reportfinder-bundle build --config <config.yaml>"
        )

    manifest = BundleManifest.load(manifest_path)
    manifest.mark_stale_components(corpus.content_hash)

    loaded = LoadedBundle(
        manifest=manifest, root=root, runtime_config_hash=config_hash(cfg)
    )
    for view_type in ALL_VIEW_TYPES:
        record = manifest.components.get(f"views.{view_type.value}")
        if record is not None and record.is_ready:
            loaded.dense[view_type] = DenseViewIndex.load(root / "views" / view_type.value)

    if (record := manifest.components.get("splade")) is not None and record.is_ready:
        loaded.splade = SpladeIndex.load(root / "splade")
    if (record := manifest.components.get("prototypes")) is not None and record.is_ready:
        loaded.prototypes = FamilyPrototypeIndex.load(root / "prototypes")
    if (record := manifest.components.get("late_interaction")) is not None and record.is_ready:
        loaded.late_interaction = LateInteractionIndex.load(root / "late_interaction")
    return loaded
