"""Train and calibrate the three-class decision head.

Rewritten to fix three defects that made the previous artifact unusable and its
metrics untrustworthy:

* **It trained on the sealed test split.** `SplitGuard` was never invoked and the
  calibration slice was a random cut of *all* scenarios, so test data reached both
  fitting and calibration. Every number it produced was optimistic by an unknown
  amount.
* **It trained on the wrong feature space.** It read 14 numeric columns from a
  precomputed parquet while stamping the artifact with the 20-name *serving*
  feature hash. The loader's hash check therefore passed and the model would have
  raised a shape error on the first live request.
* **It calibrated and evaluated on the same slice.** An ECE measured where the
  temperature was fitted describes the fit, not the calibration.

Now: features come from real pipeline runs (`passes.py`), the split guard is
enforced with operation names it actually knows, temperature is fitted on
`calibration`, and every reported metric is computed on `validation` -- a split
used for neither.

Accuracy is deliberately not the headline. The classes are roughly 75/11/14, so a
head that always says RETURN_RESULTS scores 0.75 while never abstaining, which is
the one behaviour it exists to provide.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from ..relevance.splits import SplitGuard
from .datasets import _load_splits, label_provenance
from .features import DECISION_FEATURE_HASH, DECISION_FEATURE_NAMES
from .losses import balanced_class_weights, class_weighted_cross_entropy
from .nets import DECISION_ARCHITECTURE, DecisionHead, save_model
from .oof import fit_temperature

CLASSES = DecisionHead.CLASSES

# The bundle encodes 2 = Answerable, 1 = needs clarification, 0 = no answer, which
# is the reverse of this codebase's class order. Verified against the seed
# judgements: every scenario labelled 2 has a positive judgement and every scenario
# labelled 0 has none. Getting this backwards trains a head that abstains on
# answerable queries and answers unanswerable ones, and nothing in the loss curve
# would show it.
BUNDLE_LABEL_TO_CLASS = {2: 0, 1: 1, 0: 2}

# Columns computed *from the labels*. Retained as documentation of what must never
# be trained on, even though features now come from the pipeline rather than from
# a parquet that offered them.
LABEL_DERIVED_COLUMNS = (
    "top_candidate_is_seed_positive",
    "seed_positive_present_in_pool",
    "is_no_answer_family",
    "is_conflicting_family",
    "is_missing_context_family",
)


def load_answerability_labels(relevance_root: Path) -> dict[str, str]:
    """scenario id -> decision class name.

    Labels only. The features come from running the pipeline, which is the whole
    correction: a label file cannot ship features for a feature space it has
    never seen, and the previous version's did not.
    """
    import pandas as pd

    root = Path(relevance_root)
    path = root / "processed" / "answerability_labels.parquet"
    if not path.exists():  # the v1 bundle's filename
        path = root / "processed" / "answerability_features.parquet"
    frame = pd.read_parquet(path)

    if "answerability_label" not in frame.columns:
        raise ValueError(f"no answerability_label column in {path}")
    id_column = next(
        (c for c in ("scenario_id", "Scenario ID") if c in frame.columns), None
    )
    if id_column is None:
        raise ValueError(
            f"{path} has no scenario id column, so labels cannot be joined to "
            "pipeline runs"
        )

    unknown = set(frame["answerability_label"].tolist()) - set(BUNDLE_LABEL_TO_CLASS)
    if unknown:
        raise ValueError(
            f"unexpected answerability_label values {sorted(unknown)} in {path}; "
            "refusing to guess how they map to decisions"
        )
    return {
        str(row[id_column]):
            CLASSES[BUNDLE_LABEL_TO_CLASS[int(row["answerability_label"])]]
        for _, row in frame.iterrows()
    }


def per_class_recall(predicted: np.ndarray, labels: np.ndarray) -> dict[str, float]:
    """The metric that matters. Accuracy hides a head that never abstains."""
    out: dict[str, float] = {}
    for index, name in enumerate(CLASSES):
        actual = labels == index
        out[name] = (
            float((predicted[actual] == index).mean()) if actual.any() else 0.0
        )
    return out


def train_decision(
    passes: dict[str, list],
    labels: dict[str, str],
    *,
    relevance_root: Path,
    epochs: int = 200,
    learning_rate: float = 1e-3,
    seed: int = 20260720,
    verbose: bool = False,
) -> tuple[DecisionHead, dict, dict]:
    """Fit on `train`, calibrate on `calibration`, report on `validation`."""
    from .evaluate import evaluate_decision

    torch.manual_seed(seed)

    splits = _load_splits(Path(relevance_root))
    guard = SplitGuard(splits)
    # Named operations, so the guard can actually fire. The previous call site
    # used an operation name outside DEVELOPMENT_OPERATIONS and never could.
    guard.assert_allowed([p.scenario_id for p in passes["train"]], "ranker_training")
    guard.assert_allowed([p.scenario_id for p in passes["calibration"]], "calibration")
    guard.assert_allowed(
        [p.scenario_id for p in passes["validation"]], "model_comparison"
    )

    def matrix(split: str):
        subset = [p for p in passes[split] if p.scenario_id in labels]
        if not subset:
            raise ValueError(f"no labelled passes in the {split!r} split")
        x = np.vstack([p.decision_features for p in subset]).astype(np.float32)
        y = np.array(
            [CLASSES.index(labels[p.scenario_id]) for p in subset], dtype=np.int64
        )
        return subset, x, y

    _, train_x, train_y = matrix("train")
    _, calib_x, calib_y = matrix("calibration")
    validation_subset, _, _ = matrix("validation")

    if train_x.shape[1] != len(DECISION_FEATURE_NAMES):
        raise ValueError(
            f"features are {train_x.shape[1]}-wide but serving computes "
            f"{len(DECISION_FEATURE_NAMES)}; the stamped feature hash would be a "
            "lie and the model would raise on the first live request"
        )

    x = torch.from_numpy(train_x)
    y = torch.from_numpy(train_y)
    weights = balanced_class_weights(y)

    model = DecisionHead(input_dim=train_x.shape[1])
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    model.train()
    loss_value = float("nan")
    for epoch in range(epochs):
        optimizer.zero_grad()
        loss = class_weighted_cross_entropy(model(x), y, weights)
        loss.backward()
        optimizer.step()
        loss_value = float(loss.item())
        if verbose and (epoch + 1) % 50 == 0:
            print(f"epoch {epoch + 1}/{epochs} loss={loss_value:.5f}")
    model.eval()

    # Temperature on `calibration` only: fitted on training data it would measure
    # memorisation, and fitted on validation it would contaminate the split
    # approval is decided on.
    with torch.inference_mode():
        calib_logits = model(torch.from_numpy(calib_x)).numpy()
    temperature = fit_temperature(calib_logits, calib_y)

    metrics = evaluate_decision(model, temperature, validation_subset, labels)

    metadata = {
        "architecture": DECISION_ARCHITECTURE,
        "feature_hash": DECISION_FEATURE_HASH,
        "feature_names": list(DECISION_FEATURE_NAMES),
        "feature_source": "serving_pipeline_pass",
        "splits": {
            "trained_on": "train", "calibrated_on": "calibration",
            "evaluated_on": "validation",
        },
        "excluded_label_derived_columns": list(LABEL_DERIVED_COLUMNS),
        "bundle_label_mapping": {
            str(k): CLASSES[v] for k, v in BUNDLE_LABEL_TO_CLASS.items()
        },
        "seed": seed,
        "epochs": epochs,
        "final_loss": round(loss_value, 6),
        "temperature": temperature,
        "train_size": int(len(train_y)),
        "calibration_split_size": int(len(calib_y)),
        "validation_split_size": int(len(validation_subset)),
        "expected_calibration_error": metrics["candidate"]["ece"],
        "per_class_recall": metrics["candidate"]["per_class_recall"],
        "class_weights": weights.tolist(),
        "label_distribution": {
            name: int((train_y == i).sum()) for i, name in enumerate(CLASSES)
        },
        **label_provenance(relevance_root),
    }
    return model, metadata, metrics


def main(argv=None) -> int:
    import argparse

    from .runner import train_decision_end_to_end

    parser = argparse.ArgumentParser(prog="reportfinder-train decision")
    parser.add_argument("--config", default="configs/legacy_generators.yaml")
    parser.add_argument("--relevance-root", default="data/relevance_v2")
    parser.add_argument("--out", default="artifacts/models/decision.pt")
    parser.add_argument("--eval-out", default="artifacts/evals/decision_eval.json")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--max-queries", type=int, default=None)
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args(argv)

    metadata, report_path = train_decision_end_to_end(
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
        "which requires it to beat the deterministic policy on validation."
    )
    return 0


__all__ = [
    "BUNDLE_LABEL_TO_CLASS",
    "CLASSES",
    "LABEL_DERIVED_COLUMNS",
    "load_answerability_labels",
    "main",
    "per_class_recall",
    "save_model",
    "train_decision",
]
