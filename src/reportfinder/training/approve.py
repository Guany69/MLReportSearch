"""The approval gate: what has to be true before an artifact may serve.

Loading refuses any artifact not marked `approved`. That flag used to be settable
only by hand-editing the `.pt`, which is a gate in name only -- nothing checked
that the edit was earned. This makes it earned:

1. the evaluation report must describe *this* model, matched on the weights
   digest, which survives the metadata rewrite approval performs;
2. it must be a validation-split report -- never the split the model was fitted
   on, and never the sealed test split;
3. the model must **beat the fallback it would replace** on the primary metric.
   A learned model that ties RRF is not an improvement, it is a dependency;
4. and it must not go backwards on anything else that was measured. A positive
   primary delta on its own is not evidence -- a model can edge the headline by
   noise while regressing something the headline does not capture. That is a
   trade-off, and a trade-off is a person's call, not an automatic approval.

What this gate deliberately does *not* claim: that the model is good. The labels
are synthetic, so approval is stamped `synthetic_v2_evaluation` and
`human_validated: false` is preserved. It licenses serving; it does not license a
production-accuracy claim.
"""

from __future__ import annotations

import argparse
import getpass
import hashlib
import json
import re
from datetime import UTC, datetime
from pathlib import Path

APPROVAL_BASIS = "synthetic_v2_evaluation"

# Approval must be earned on a split the model never saw. `train` is what it was
# fitted on and `calibration` is where its temperature came from; `test` is
# sealed for a final release measurement and must not be spent here.
ALLOWED_SPLIT = "validation"


class ApprovalRefused(RuntimeError):
    """The artifact did not earn approval. The message says which check failed."""


def _load(path: Path):
    import torch

    payload = torch.load(Path(path), map_location="cpu", weights_only=False)
    return payload, payload.get("metadata", {})


# Metrics that describe the run rather than the quality of the ranking. Comparing
# these between arms is meaningless -- both arms see the same queries.
_DESCRIPTIVE_METRICS = frozenset({
    "queries", "sibling_queries_measured", "fallback_name",
    "per_class_recall", "confusion_matrix", "predicted_distribution",
    "abstain_rate",  # a rate to be judged, not maximised
})

# Numeric noise, not a regression: a metric rounded to 4 places can differ in the
# last digit for reasons no one should be asked to explain.
_REGRESSION_TOLERANCE = 1e-4

# Which way is better. This exists because the gate previously compared every
# metric as if larger were better, which silently inverts the judgement on every
# error, loss and latency measure -- a model with twice the calibration error
# would have read as an improvement.
_LOWER_IS_BETTER = frozenset({
    "latency", "ece", "nll", "brier", "loss", "risk", "error", "regret",
})
_HIGHER_IS_BETTER = frozenset({
    "recall", "precision", "f1", "accuracy", "auc", "auroc", "mrr", "map",
    "ndcg", "dcg", "hit", "hits", "coverage", "specificity", "sensitivity",
    "agreement", "throughput", "qps",
})
_LOWER_IS_BETTER_SUFFIXES = ("_ms", "_loss", "_error", "_seconds", "_s")


def _pattern_direction(name: str) -> str | None:
    """Direction inferred from the metric's name, or None when it is not obvious.

    None is not a default -- it is a refusal trigger. Guessing "higher" for an
    unrecognised name is exactly the bug this replaces.
    """
    lowered = name.strip().lower()
    if lowered.endswith(_LOWER_IS_BETTER_SUFFIXES):
        return "lower"
    # Split on anything that is not alphanumeric, and drop @k tails: the metric
    # `family_ndcg@5` is a `ndcg`.
    tokens = {token for token in re.split(r"[^a-z0-9]+", lowered.split("@")[0]) if token}
    lower_hit, higher_hit = tokens & _LOWER_IS_BETTER, tokens & _HIGHER_IS_BETTER
    if lower_hit and not higher_hit:
        return "lower"
    if higher_hit and not lower_hit:
        return "higher"
    # Both or neither: ambiguous by name. `error_recall` deserves a declaration,
    # not a coin flip.
    return None


def _comparable(candidate: dict, fallback: dict) -> list[str]:
    """Metric names present and numeric in both arms."""
    out = []
    for name, value in candidate.items():
        if name in _DESCRIPTIVE_METRICS or name not in fallback:
            continue
        other = fallback[name]
        if isinstance(value, bool) or isinstance(other, bool):
            continue
        if isinstance(value, (int, float)) and isinstance(other, (int, float)):
            out.append(name)
    return out


def _resolve_directions(report: dict) -> dict[str, str]:
    """A direction for every comparable metric, aggregate and per-slice.

    Declared directions in the report win; name patterns fill the rest; anything
    still unresolved refuses the approval. A metric nobody can say the sign of is
    a metric the gate cannot use, and pretending otherwise is how a regression
    gets approved as an improvement.
    """
    metrics = report.get("metrics") or {}
    names = set(_comparable(metrics.get("candidate") or {}, metrics.get("fallback") or {}))
    for arms in (report.get("slices") or {}).values():
        if isinstance(arms, dict):
            names |= set(_comparable(arms.get("candidate") or {}, arms.get("fallback") or {}))

    declared_raw = report.get("metric_directions") or {}
    declared: dict[str, str] = {}
    for name, value in declared_raw.items():
        normalized = str(value).strip().lower()
        if normalized not in ("higher", "lower"):
            raise ApprovalRefused(
                f"metric_directions[{name!r}] is {value!r}; it must be exactly "
                "'higher' or 'lower'. An unreadable declaration is worse than "
                "none, because it looks like the question was answered."
            )
        declared[name] = normalized

    directions, unresolved = {}, []
    for name in sorted(names):
        direction = declared.get(name) or _pattern_direction(name)
        if direction is None:
            unresolved.append(name)
        else:
            directions[name] = direction
    if unresolved:
        raise ApprovalRefused(
            f"cannot tell which direction is better for {unresolved}. Declare them "
            "in the evaluation report's `metric_directions` as 'higher' or "
            "'lower'. Assuming higher-is-better would silently invert the "
            "judgement on any error or latency measure."
        )
    return directions


def _regressions(
    candidate: dict, fallback: dict, directions: dict[str, str],
) -> list[tuple[str, float, float]]:
    """Metrics where the candidate is worse than the fallback, in its own direction."""
    out = []
    for name in sorted(_comparable(candidate, fallback)):
        value, other = float(candidate[name]), float(fallback[name])
        worse = (
            value < other - _REGRESSION_TOLERANCE if directions[name] == "higher"
            else value > other + _REGRESSION_TOLERANCE
        )
        if worse:
            out.append((name, value, other))
    return out


def _slice_regressions(
    report: dict, directions: dict[str, str],
) -> list[tuple[str, float, float]]:
    """Per-slice regressions, named `slice/metric`.

    Aggregate metrics average a regression away. The fusion model's headline gain
    is +0.0031 nDCG while it loses on the one archetype the label set was built to
    measure -- a gate that reads only the aggregate cannot see that.
    """
    out = []
    for slice_name, arms in sorted((report.get("slices") or {}).items()):
        if not isinstance(arms, dict):
            continue
        for name, value, other in _regressions(
            arms.get("candidate") or {}, arms.get("fallback") or {}, directions
        ):
            out.append((f"{slice_name}/{name}", value, other))
    return out


def approve(
    artifact: Path,
    evaluation: Path,
    *,
    min_delta: float = 0.0,
    allow_regressions: bool = False,
    approved_by: str | None = None,
) -> dict:
    """Stamp an artifact approved, or refuse and say why."""
    import torch

    from .nets import weights_sha256

    artifact, evaluation = Path(artifact), Path(evaluation)
    if not artifact.exists():
        raise ApprovalRefused(f"artifact not found: {artifact}")
    if not evaluation.exists():
        raise ApprovalRefused(f"evaluation report not found: {evaluation}")

    payload, metadata = _load(artifact)
    report = json.loads(evaluation.read_text())

    # 1. the report describes this model
    digest = weights_sha256(payload["state_dict"])
    described = (report.get("artifact") or {}).get("weights_sha256")
    if described != digest:
        raise ApprovalRefused(
            f"the report describes weights {described!r} but the artifact is "
            f"{digest!r}. An evaluation of a different model is not evidence "
            "about this one."
        )
    if (report.get("artifact") or {}).get("feature_hash") != metadata.get("feature_hash"):
        raise ApprovalRefused(
            "the report and the artifact disagree on the feature schema; the "
            "metrics would have been computed from different columns"
        )

    # Matched against the architecture constants, not by substring. The names are
    # `pytorch_mlp_64_32_1` and `pytorch_mlp_32_16_3` -- neither contains the word
    # "fusion" or "decision", so a substring test silently classified every
    # artifact as a fusion model and refused every real decision approval.
    from .nets import DECISION_ARCHITECTURE, FUSION_ARCHITECTURE

    architecture = str(metadata.get("architecture", ""))
    expected_kind = {
        FUSION_ARCHITECTURE: "fusion", DECISION_ARCHITECTURE: "decision",
    }.get(architecture)
    if expected_kind is None:
        raise ApprovalRefused(
            f"unrecognised architecture {architecture!r}; approval cannot tell "
            "which fallback this model would replace"
        )
    if report.get("kind") != expected_kind:
        raise ApprovalRefused(
            f"report kind {report.get('kind')!r} does not match artifact "
            f"architecture {architecture!r} (expected {expected_kind!r})"
        )

    # 2. earned on a split the model never saw
    if report.get("split") != ALLOWED_SPLIT:
        raise ApprovalRefused(
            f"approval requires a {ALLOWED_SPLIT!r} report, got "
            f"{report.get('split')!r}. Fitting and judging on the same split "
            "measures memorisation; the test split is sealed for release."
        )
    if not report.get("query_count"):
        raise ApprovalRefused("the report evaluated zero queries")

    # 3. beats the fallback it would replace
    primary = report.get("primary_metric") or {}
    delta = primary.get("delta")
    if delta is None:
        raise ApprovalRefused("the report states no primary-metric delta")
    if float(delta) <= min_delta:
        raise ApprovalRefused(
            f"{primary.get('name')} = {primary.get('candidate')} against fallback "
            f"{primary.get('fallback')} (delta {delta:+.4f}, required > "
            f"{min_delta}). The fallback is simpler, already serving, and has no "
            "artifact to keep in sync -- it wins ties."
        )

    # 4. and does not go backwards on anything else
    #
    # A positive primary delta on its own is not evidence: a model can edge the
    # headline metric by noise while regressing something the headline does not
    # capture. That is a trade-off, and a trade-off is a decision for a person --
    # so it is refused by default rather than approved automatically.
    #
    # Directions are resolved unconditionally, including under
    # --allow-regressions: that flag permits a *known* trade-off, and a metric
    # whose sign nobody can state is not a known anything.
    directions = _resolve_directions(report)
    metrics = report.get("metrics") or {}
    if not allow_regressions:
        regressions = _regressions(
            metrics.get("candidate") or {}, metrics.get("fallback") or {}, directions,
        ) + _slice_regressions(report, directions)
        if regressions:
            listed = "; ".join(
                f"{name} {candidate} vs fallback {fallback}"
                for name, candidate, fallback in regressions
            )
            raise ApprovalRefused(
                f"the primary metric improved by {delta:+.4f} but other measured "
                f"results went backwards: {listed}. Approving would trade a "
                "measured regression for an unmeasured gain. Re-run with "
                "--allow-regressions only if that trade is deliberate."
            )

    metadata["approved"] = True
    metadata["approval"] = {
        "basis": APPROVAL_BASIS,
        "eval_report_path": str(evaluation),
        "eval_report_sha256": hashlib.sha256(evaluation.read_bytes()).hexdigest()[:32],
        "primary_metric": primary,
        "min_delta": min_delta,
        "allowed_regressions": allow_regressions,
        # What the gate actually judged, so a later reader can tell whether a
        # metric was compared, skipped as descriptive, or absent from one arm.
        "kind": expected_kind,
        "architecture": architecture,
        "split": report.get("split"),
        "query_count": report.get("query_count"),
        "metric_directions": directions,
        "compared_metrics": sorted(
            _comparable(metrics.get("candidate") or {}, metrics.get("fallback") or {})
        ),
        "slices_compared": sorted(report.get("slices") or {}),
        "approved_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "approved_by": approved_by or _current_user(),
    }
    # Never inferred from approval. Approval says a model beat its fallback on
    # synthetic labels; it says nothing about a human having looked at anything.
    metadata["human_validated"] = bool(metadata.get("human_validated", False))

    payload["metadata"] = metadata
    torch.save(payload, artifact)
    artifact.with_suffix(".json").write_text(
        json.dumps({"input_dim": payload["input_dim"], **metadata},
                   indent=2, sort_keys=True) + "\n"
    )
    return metadata


def _current_user() -> str:
    try:
        return getpass.getuser()
    except Exception:  # noqa: BLE001 - provenance, never worth failing approval
        return "unknown"


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="reportfinder-train approve")
    parser.add_argument("artifact", type=Path)
    parser.add_argument("--evaluation", type=Path, required=True)
    parser.add_argument(
        "--min-delta", type=float, default=0.0,
        help="How far the model must beat its fallback. Ties go to the fallback.",
    )
    parser.add_argument(
        "--allow-regressions", action="store_true",
        help="Approve even if a non-primary metric went backwards. That is a "
             "trade-off, so it has to be asked for explicitly.",
    )
    args = parser.parse_args(argv)

    try:
        metadata = approve(
            args.artifact, args.evaluation, min_delta=args.min_delta,
            allow_regressions=args.allow_regressions,
        )
    except ApprovalRefused as error:
        print(f"REFUSED: {error}")
        return 1

    primary = metadata["approval"]["primary_metric"]
    print(
        f"APPROVED {args.artifact}\n"
        f"  basis            {metadata['approval']['basis']}\n"
        f"  {primary['name']:<16} {primary['candidate']} vs fallback "
        f"{primary['fallback']} ({primary['delta']:+.4f})\n"
        f"  human_validated  {metadata['human_validated']}  "
        "<- still false; synthetic labels do not become human ones"
    )
    return 0
