"""Sweep the two decision thresholds offline, over cached pipeline passes.

    uv run python scripts/sweep_decision_thresholds.py \
        --relevance-root data/relevance_v2 --out artifacts/threshold_sweep.json

`weak_rerank_logit` (-9.0) and `clarify_margin` (1.0) were set by inspecting six
probe queries. The docs call the first "the single most consequential
uncalibrated number in the system", and the policy abstains on ~40% of queries.

Both are post-hoc functions of numbers each pipeline run already produces, so a
grid costs one pass plus arithmetic rather than one pass per grid point. The
sweep runs on the *calibration* split -- tuning on validation would spend the
split approval is decided on, and the sealed test split is refused outright.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from reportfinder.pipeline.decide import replay_fallback_policy
from reportfinder.relevance.splits import SplitGuard
from reportfinder.training.datasets import _load_splits
from reportfinder.training.runner import gather_passes
from reportfinder.training.train_decision import load_answerability_labels

CLASSES = ("RETURN_RESULTS", "ASK_CLARIFICATION", "NO_CONFIDENT_MATCH")


def macro_per_class_recall(predicted, actual) -> tuple[float, dict]:
    """Macro, not accuracy: the classes are ~75/11/14, so always answering
    scores 0.75 while never abstaining -- the one behaviour being tuned."""
    per_class = {}
    for name in CLASSES:
        support = sum(1 for a in actual if a == name)
        hits = sum(1 for p, a in zip(predicted, actual, strict=True)
                   if a == name and p == name)
        per_class[name] = hits / support if support else None
    measured = [v for v in per_class.values() if v is not None]
    return (float(np.mean(measured)) if measured else 0.0), per_class


def evaluate_grid(passes, labels, *, weak_values, margin_values) -> list[dict]:
    subset = [p for p in passes if p.scenario_id in labels]
    actual = [labels[p.scenario_id] for p in subset]

    rows = []
    for weak in weak_values:
        for margin in margin_values:
            predicted = [
                replay_fallback_policy(
                    ce_top=p.ce_top, ce_second=p.ce_second,
                    fusion_top=p.fusion_top, fusion_second=p.fusion_second,
                    risk_band=p.risk_band,
                    has_distinguishing_facet=p.has_distinguishing_facet,
                    weak_rerank_logit=weak, clarify_margin=margin,
                    has_families=bool(p.families_ranked),
                    # Gated decisions are catalog facts and no threshold moves
                    # them. Replaying without this scored a policy the pipeline
                    # does not run.
                    hard_gate_decision=p.hard_gate_decision,
                )
                for p in subset
            ]
            score, per_class = macro_per_class_recall(predicted, actual)
            rows.append({
                "weak_rerank_logit": round(float(weak), 2),
                "clarify_margin": round(float(margin), 2),
                "macro_per_class_recall": round(score, 4),
                "per_class_recall": {
                    k: (round(v, 4) if v is not None else None)
                    for k, v in per_class.items()
                },
                "abstain_rate": round(
                    sum(1 for p in predicted if p == "NO_CONFIDENT_MATCH")
                    / len(predicted), 4
                ) if predicted else 0.0,
            })
    return rows


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="sweep_decision_thresholds")
    parser.add_argument("--config", type=Path,
                        default=Path("configs/legacy_generators.yaml"))
    parser.add_argument("--relevance-root", type=Path,
                        default=Path("data/relevance_v2"))
    parser.add_argument("--out", type=Path,
                        default=Path("artifacts/threshold_sweep.json"))
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args(argv)

    passes, _pipeline, _cfg = gather_passes(
        config_path=args.config, relevance_root=args.relevance_root,
        splits=("calibration", "validation"), verbose=args.verbose,
    )
    labels = load_answerability_labels(args.relevance_root)

    # Tuning is a development operation; the sealed split must never feed it.
    guard = SplitGuard(_load_splits(args.relevance_root))
    for split in ("calibration", "validation"):
        guard.assert_allowed(
            [p.scenario_id for p in passes[split]], "threshold_tuning"
        )

    weak_values = np.arange(-12.0, -1.9, 0.5)
    margin_values = np.arange(0.0, 3.01, 0.25)

    grid = evaluate_grid(
        passes["calibration"], labels,
        weak_values=weak_values, margin_values=margin_values,
    )
    best = max(grid, key=lambda row: row["macro_per_class_recall"])

    # Confirm on validation. A point chosen on calibration that does not hold up
    # is a point fitted to noise.
    confirmation = evaluate_grid(
        passes["validation"], labels,
        weak_values=[best["weak_rerank_logit"]],
        margin_values=[best["clarify_margin"]],
    )[0]
    shipped = evaluate_grid(
        passes["validation"], labels, weak_values=[-9.0], margin_values=[1.0],
    )[0]

    report = {
        "grid_points": len(grid),
        "tuned_on": "calibration",
        "confirmed_on": "validation",
        "recommended": best,
        "recommended_on_validation": confirmation,
        "shipped_defaults_on_validation": shipped,
        "delta_macro_recall": round(
            confirmation["macro_per_class_recall"]
            - shipped["macro_per_class_recall"], 4
        ),
        "label_basis": (
            "synthetic v2; measures agreement with the label generator, not "
            "whether abstaining was the right call for a real user"
        ),
        "grid": grid,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")

    print(json.dumps({k: v for k, v in report.items() if k != "grid"}, indent=2,
                     sort_keys=True))
    print(f"\n-> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
