"""End-to-end: run the pipeline once per query, then fit, evaluate, and report.

The shared pass is the point. Fusion needs per-candidate features, the decision
head needs per-query features, and evaluating either against its fallback needs
the fallback's actual output. All three come out of the same search, so one pass
replaces what would otherwise be three -- and at the measured per-query pipeline
cost that is the difference between minutes and hours.

The cache makes retraining cheap. The expensive part is the pass; the fits take
seconds, so a threshold change or a loss tweak re-uses features instead of
re-earning them.
"""

from __future__ import annotations

import json
from pathlib import Path

from ..auth import DEVELOPMENT_PRINCIPAL
from ..config import DEFAULT, Config, from_mapping
from ..model import ReportFinder
from ..pipeline.orchestrator import SearchPipeline
from ..represent import load_or_build
from .datasets import cluster_near_duplicates, load_labelled_queries
from .oof import assign_folds
from .passes import cache_key, load_passes, run_feature_pass, save_passes

SPLITS = ("train", "calibration", "validation")
CACHE_ROOT = Path("artifacts/feature_cache")


def _load_config(config_path: Path | None):
    if config_path is None:
        return DEFAULT
    import yaml

    return from_mapping(yaml.safe_load(Path(config_path).read_text()) or {})


def _archetypes(relevance_root: Path) -> dict[str, str]:
    """scenario id -> archetype, for per-slice reporting."""
    import pandas as pd

    path = Path(relevance_root) / "raw" / "scenarios_v2.csv"
    if not path.exists():
        return {}
    frame = pd.read_csv(path)
    if "Archetype" not in frame.columns:
        return {}
    return dict(zip(frame["Scenario ID"], frame["Archetype"], strict=True))


def _dataset_manifest(relevance_root: Path) -> dict:
    path = Path(relevance_root) / "dataset_manifest.json"
    return json.loads(path.read_text()) if path.exists() else {}


def gather_passes(
    *,
    config_path: Path | None,
    relevance_root: Path,
    splits=SPLITS,
    max_queries: int | None = None,
    cache_root: Path = CACHE_ROOT,
    verbose: bool = False,
) -> tuple[dict[str, list], SearchPipeline, Config]:
    """Cached feature passes for each split, running only what is missing."""
    cfg = _load_config(config_path)
    if cfg.fusion.artifact_path is not None or cfg.decision.artifact_path is not None:
        raise ValueError(
            "feature generation must run on the fallbacks. This config points at "
            "trained artifacts, so the features would describe those artifacts "
            "rather than retrieval -- a feedback loop, not training."
        )

    finder = ReportFinder(load_or_build(cfg, verbose=False), cfg)
    pipeline = finder.pipeline
    manifest = _dataset_manifest(relevance_root)
    archetypes = _archetypes(relevance_root)
    bundle_version = (
        pipeline.bundle.manifest.bundle_version if pipeline.bundle else ""
    )

    out: dict[str, list] = {}
    for split in splits:
        key = cache_key(
            bundle_version=bundle_version,
            dataset_version=str(manifest.get("version", "unknown")),
            split=split,
        )
        directory = Path(cache_root) / key
        cached = load_passes(directory, key=key)
        if cached is not None and max_queries is None:
            if verbose:
                print(f"  {split}: {len(cached)} passes from cache")
            out[split] = cached
            continue

        labelled = load_labelled_queries(relevance_root, split=split)
        if max_queries is not None:
            labelled = labelled[:max_queries]

        # Folds group near-duplicate queries, so a paraphrase cannot sit in the
        # fold used to generate features for its twin.
        clusters = cluster_near_duplicates([q.query for q in labelled])
        assignment = assign_folds(
            [q.scenario_id for q in labelled], clusters, n_splits=5,
        )
        folds = dict(
            zip(
                [q.scenario_id for q in labelled],
                assignment.folds.tolist(),
                strict=True,
            )
        )

        def progress(done, total, split=split):
            if verbose:
                print(f"  {split}: {done}/{total} queries", flush=True)

        passes = run_feature_pass(
            pipeline, labelled, principal=DEVELOPMENT_PRINCIPAL,
            archetypes=archetypes, folds=folds,
            progress=progress if verbose else None,
        )
        if max_queries is None:
            save_passes(passes, directory, key=key)
        out[split] = passes

    return out, pipeline, cfg


def train_fusion_end_to_end(
    *,
    config_path: Path | None,
    relevance_root: Path,
    out: Path,
    eval_out: Path,
    epochs: int = 20,
    max_queries: int | None = None,
    cache_root: Path = CACHE_ROOT,
    verbose: bool = False,
) -> tuple[dict, Path]:
    from .evaluate import evaluate_fusion, slice_metrics, write_eval_report
    from .nets import save_model, weights_sha256
    from .train_fusion import fit_fusion

    passes, pipeline, cfg = gather_passes(
        config_path=config_path, relevance_root=relevance_root,
        max_queries=max_queries, cache_root=cache_root, verbose=verbose,
    )
    model, metadata = fit_fusion(
        passes["train"], relevance_root=relevance_root, epochs=epochs,
        verbose=verbose,
    )

    def family_of(instance_id: str) -> str:
        return pipeline.corpus.instance(instance_id).family_id

    metrics = evaluate_fusion(model, passes["validation"], family_of=family_of)
    save_model(model, out, metadata)

    import torch

    digest = weights_sha256(
        torch.load(out, map_location="cpu", weights_only=False)["state_dict"]
    )
    report_path = write_eval_report(
        "fusion", out, metrics,
        weights_sha256=digest, feature_hash=metadata["feature_hash"],
        dataset_root=relevance_root,
        dataset_manifest=_dataset_manifest(relevance_root),
        split="validation", query_count=len(passes["validation"]),
        slices=slice_metrics(
            passes["validation"],
            lambda subset: evaluate_fusion(model, subset, family_of=family_of),
        ),
        config_path=str(config_path or ""),
        bundle_version=(
            pipeline.bundle.manifest.bundle_version if pipeline.bundle else ""
        ),
        out=eval_out,
    )
    return metadata, report_path


def train_decision_end_to_end(
    *,
    config_path: Path | None,
    relevance_root: Path,
    out: Path,
    eval_out: Path,
    epochs: int = 200,
    max_queries: int | None = None,
    cache_root: Path = CACHE_ROOT,
    verbose: bool = False,
) -> tuple[dict, Path]:
    from .evaluate import write_eval_report
    from .nets import save_model, weights_sha256
    from .train_decision import load_answerability_labels, train_decision

    passes, pipeline, cfg = gather_passes(
        config_path=config_path, relevance_root=relevance_root,
        max_queries=max_queries, cache_root=cache_root, verbose=verbose,
    )
    labels = load_answerability_labels(relevance_root)
    model, metadata, metrics = train_decision(
        passes, labels, relevance_root=relevance_root, epochs=epochs,
        verbose=verbose,
    )
    save_model(model, out, metadata)

    import torch

    digest = weights_sha256(
        torch.load(out, map_location="cpu", weights_only=False)["state_dict"]
    )
    report_path = write_eval_report(
        "decision", out, metrics,
        weights_sha256=digest, feature_hash=metadata["feature_hash"],
        dataset_root=relevance_root,
        dataset_manifest=_dataset_manifest(relevance_root),
        split="validation",
        query_count=metadata["validation_split_size"],
        config_path=str(config_path or ""),
        bundle_version=(
            pipeline.bundle.manifest.bundle_version if pipeline.bundle else ""
        ),
        out=eval_out,
    )
    return metadata, report_path
