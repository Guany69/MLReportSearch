"""Train the listwise fusion model.

Runnable, and honest about what it produces. With only synthetic labels available,
the artifact is written with `approved: false`, which `load_fusion_model` refuses
to serve. That is deliberate: the training path should be exercisable end to end
long before anyone is entitled to put its output in front of a user.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from ..pipeline.fuse import FUSION_FEATURE_HASH, FUSION_FEATURE_NAMES, build_features
from .datasets import (
    build_listwise_batches,
    cluster_near_duplicates,
    label_provenance,
    load_labelled_queries,
)
from .losses import listnet_loss
from .nets import FUSION_ARCHITECTURE, FusionMLP
from .oof import assign_folds, generate_out_of_fold_features


@dataclass
class TrainingReport:
    epochs: int
    final_loss: float
    query_count: int
    candidate_count: int
    feature_dim: int

    def as_dict(self) -> dict[str, object]:
        return {
            "epochs": self.epochs,
            "final_loss": round(self.final_loss, 6),
            "query_count": self.query_count,
            "candidate_count": self.candidate_count,
            "feature_dim": self.feature_dim,
        }


def train_fusion(
    pipeline,
    relevance_root: Path,
    *,
    split: str = "train",
    epochs: int = 20,
    learning_rate: float = 1e-3,
    seed: int = 20260720,
    folds: int = 5,
    max_queries: int | None = None,
    principal=None,
    verbose: bool = False,
) -> tuple[FusionMLP, dict]:
    """Generate out-of-fold features, then fit the listwise model."""
    from ..auth import DEVELOPMENT_PRINCIPAL

    torch.manual_seed(seed)
    np.random.seed(seed)

    labelled = load_labelled_queries(relevance_root, split)
    if max_queries is not None:
        labelled = labelled[:max_queries]
    if not labelled:
        raise ValueError(f"no labelled queries in split {split!r}")

    clusters = cluster_near_duplicates([q.query for q in labelled])
    assignment = assign_folds([q.scenario_id for q in labelled], clusters, folds, seed)

    examples = generate_out_of_fold_features(
        pipeline, labelled, assignment,
        principal=principal or DEVELOPMENT_PRINCIPAL,
        build_features=build_features,
        progress=(lambda s, f: print(f"  fold {f}: {s}")) if verbose else None,
    )
    if not examples:
        raise ValueError(
            "out-of-fold feature generation produced nothing; the pipeline "
            "returned no candidates for any labelled query"
        )

    batches = build_listwise_batches(examples)
    features = torch.from_numpy(np.stack([b.features for b in batches]))
    grades = torch.from_numpy(np.stack([b.grades for b in batches]))
    mask = torch.from_numpy(np.stack([b.mask for b in batches]))

    model = FusionMLP(input_dim=features.shape[-1])
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    model.train()
    loss_value = float("nan")
    for epoch in range(epochs):
        optimizer.zero_grad()
        scores = model(features)
        loss = listnet_loss(scores, grades, mask)
        loss.backward()
        optimizer.step()
        loss_value = float(loss.item())
        if verbose:
            print(f"epoch {epoch + 1}/{epochs} loss={loss_value:.5f}")
    model.eval()

    report = TrainingReport(
        epochs=epochs, final_loss=loss_value, query_count=len(batches),
        candidate_count=int(mask.sum().item()), feature_dim=features.shape[-1],
    )
    metadata = {
        "architecture": FUSION_ARCHITECTURE,
        "feature_hash": FUSION_FEATURE_HASH,
        "feature_names": list(FUSION_FEATURE_NAMES),
        "split": split,
        "seed": seed,
        "folds": folds,
        "training": report.as_dict(),
        **label_provenance(relevance_root),
    }
    return model, metadata


def fit_fusion(
    passes,
    *,
    relevance_root: Path,
    epochs: int = 20,
    learning_rate: float = 1e-3,
    seed: int = 20260720,
    verbose: bool = False,
) -> tuple[FusionMLP, dict]:
    """Fit the listwise model on cached feature passes.

    Takes passes rather than a pipeline: the features already came from real
    searches (`training/passes.py`), so this is pure fitting. The previous
    version regenerated features inline, which meant every retrain paid the full
    pipeline cost again and evaluation could not reuse anything.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    examples = [
        (p.scenario_id, p.instance_ids, p.fusion_features, p.grades)
        for p in passes if len(p.instance_ids)
    ]
    if not examples:
        raise ValueError(
            "no candidates in any pass; the pipeline returned nothing for every "
            "labelled query, which is a retrieval failure rather than a training one"
        )

    batches = build_listwise_batches(examples)
    features = torch.from_numpy(np.stack([b.features for b in batches]))
    grades = torch.from_numpy(np.stack([b.grades for b in batches]))
    mask = torch.from_numpy(np.stack([b.mask for b in batches]))

    model = FusionMLP(input_dim=features.shape[-1])
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    model.train()
    loss_value = float("nan")
    for epoch in range(epochs):
        optimizer.zero_grad()
        loss = listnet_loss(model(features), grades, mask)
        loss.backward()
        optimizer.step()
        loss_value = float(loss.item())
        if verbose and (epoch + 1) % 5 == 0:
            print(f"epoch {epoch + 1}/{epochs} loss={loss_value:.5f}")
    model.eval()

    report = TrainingReport(
        epochs=epochs, final_loss=loss_value, query_count=len(batches),
        candidate_count=int(mask.sum().item()), feature_dim=features.shape[-1],
    )
    return model, {
        "architecture": FUSION_ARCHITECTURE,
        "feature_hash": FUSION_FEATURE_HASH,
        "feature_names": list(FUSION_FEATURE_NAMES),
        "feature_source": "serving_pipeline_pass",
        "splits": {"trained_on": "train", "evaluated_on": "validation"},
        "seed": seed,
        "training": report.as_dict(),
        **label_provenance(relevance_root),
    }


def main(argv=None) -> int:
    import argparse

    from .runner import train_fusion_end_to_end

    parser = argparse.ArgumentParser(prog="reportfinder-train fusion")
    parser.add_argument("--config", default="configs/legacy_generators.yaml")
    parser.add_argument("--relevance-root", default="data/relevance_v2")
    parser.add_argument("--out", default="artifacts/models/fusion.pt")
    parser.add_argument("--eval-out", default="artifacts/evals/fusion_eval.json")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--max-queries", type=int, default=None)
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args(argv)

    metadata, report_path = train_fusion_end_to_end(
        config_path=Path(args.config),
        relevance_root=Path(args.relevance_root),
        out=Path(args.out),
        eval_out=Path(args.eval_out),
        epochs=args.epochs,
        max_queries=args.max_queries,
        verbose=args.verbose,
    )
    print(json.dumps(metadata, indent=2, sort_keys=True))
    print(
        f"\nWrote {args.out} (approved=false) and {report_path}.\n"
        "It will NOT be served until approved:\n"
        f"  uv run reportfinder-train approve {args.out} "
        f"--evaluation {report_path}\n"
        "which requires it to beat the RRF fallback on validation."
    )
    return 0
