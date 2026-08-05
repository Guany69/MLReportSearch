"""One pipeline run per query, yielding every feature set at once.

Three things were wrong with generating features the old way, and all three are
fixed by the same change -- running the *real* pipeline once and keeping what it
produced:

1. **Abstained queries produced nothing.** The old generator iterated
   `outcome.families`, which is empty on NO_CONFIDENT_MATCH. ~40% of queries take
   that branch, so ~40% of the training set silently vanished -- and the ones that
   vanished were exactly the hard ones a decision head most needs to learn from.
2. **The plan was a stand-in.** A `_PlanView` shim returned `{}` for stated
   facets, so the `query_specificity` fusion feature was always 0 in training and
   real at serving. That is textbook train/serve skew, invisible in every metric.
3. **The decision head was trained on the wrong columns entirely** -- 14 numeric
   columns from a precomputed parquet, while being stamped with the 20-name
   serving feature hash. The hash gate therefore passed and the model would have
   raised a shape error at request time.

Features are cached per split. The pass is the expensive part (one full search per
query); the fits that consume it take seconds, so caching turns re-training into
a fast loop instead of a two-hour one.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from ..auth import Principal, SearchRequest
from ..pipeline.decide import was_hard_gated
from ..pipeline.fuse import FUSION_FEATURE_HASH, FUSION_FEATURE_NAMES, build_features
from .features import DECISION_FEATURE_HASH, decision_features

log = logging.getLogger(__name__)

# Bumped when the *storage layout* changes, not just the feature schema. v1
# stored one npz member per query; v2 stores one concatenated matrix plus row
# offsets; v3 adds hard_gate_decision, whose absence would make a threshold
# replay silently score a policy the pipeline does not run. Both hold the same numbers, so nothing but the version distinguishes
# them -- a cache written by v1 and read by v2 raises a KeyError rather than
# silently returning wrong features, but only because this is in the key.
CACHE_SCHEMA_VERSION = "3"


@dataclass
class QueryPass:
    """What one query's pipeline run produced, in a trainable shape."""

    scenario_id: str
    query: str
    archetype: str
    fold: int

    instance_ids: list[str]
    fusion_features: np.ndarray          # [n_candidates, 40]
    decision_features: np.ndarray        # [20]
    grades: dict[str, int]

    # The deterministic fallbacks' own output, recorded during the same run.
    # Comparing a learned model against them later therefore costs nothing and
    # cannot drift from what the fallback would really have done.
    rrf_ranking: list[str] = field(default_factory=list)
    fallback_decision: str = ""
    families_ranked: list[str] = field(default_factory=list)
    selected_by_family: dict[str, str] = field(default_factory=dict)

    # Enough of the decision policy's inputs to replay it offline over a
    # threshold grid, instead of re-running the pipeline per grid point.
    ce_top: float | None = None
    ce_second: float | None = None
    fusion_top: float = 0.0
    fusion_second: float = 0.0
    risk_band: str = "LOW"
    has_distinguishing_facet: bool = False
    # Set when a hard gate decided this query, so a threshold replay can honour
    # gates it cannot recompute. `None` means the thresholds actually decided.
    hard_gate_decision: str | None = None

    @property
    def positives(self) -> set[str]:
        return {r for r, g in self.grades.items() if g > 0}


def _fallback_inputs(state) -> dict:
    """The values `decide()`'s fallback branch reads, so it can be replayed."""
    families = state.families
    if not families:
        return {"ce_top": None, "ce_second": None, "fusion_top": 0.0,
                "fusion_second": 0.0, "has_distinguishing_facet": False}

    top = families[0]
    second = families[1] if len(families) > 1 else None
    return {
        "ce_top": top.selected.cross_encoder_score,
        "ce_second": second.selected.cross_encoder_score if second else None,
        "fusion_top": float(top.score),
        "fusion_second": float(second.score) if second else 0.0,
        # Whether a clarification could be grounded at all -- the fallback asks
        # only when the leading families actually differ on something.
        "has_distinguishing_facet": _distinguishing_facet_exists(families, state),
    }


def _distinguishing_facet_exists(families, state) -> bool:
    from ..pipeline.decide import _candidate_difference

    corpus = getattr(state, "corpus", None)
    if corpus is None:
        return False
    try:
        return _candidate_difference(families, corpus) is not None
    except Exception:  # noqa: BLE001 - a replay input, never worth failing a pass
        return False


def run_feature_pass(
    pipeline,
    labelled,
    *,
    principal: Principal,
    archetypes: dict[str, str],
    folds: dict[str, int] | None = None,
    progress=None,
) -> list[QueryPass]:
    """Run every labelled query through the real pipeline once.

    Asserts the pipeline is on its fallbacks. Features describing a pipeline that
    already contains a trained fuser would teach the next fuser to agree with the
    last one, which is a feedback loop, not learning.
    """
    from ..pipeline.fuse import RrfFuser

    if not isinstance(pipeline.fuser, RrfFuser):
        raise ValueError(
            "feature generation requires the RRF fallback fuser; features from a "
            "pipeline that already has a trained fuser describe that fuser, not "
            "retrieval. Clear fusion.artifact_path for the pass."
        )
    if pipeline.decision_head is not None:
        raise ValueError(
            "feature generation requires the deterministic decision fallback; "
            "clear decision.artifact_path for the pass."
        )

    folds = folds or {}
    out: list[QueryPass] = []
    for index, item in enumerate(labelled):
        if progress is not None and index % 25 == 0:
            progress(index, len(labelled))

        state = pipeline.run_traced(
            SearchRequest(raw_query=item.query, principal=principal)
        )
        state.corpus = pipeline.corpus  # for the replay inputs

        instance_ids, rows = [], []
        for family in state.families:
            for member in family.instances:
                instance = pipeline.corpus.instance(member.instance_id)
                instance_ids.append(member.instance_id)
                rows.append(build_features(
                    member.record, instance,
                    pipeline.corpus.families[member.family_id],
                    state.rerank, state.plan,
                ))

        fusion_matrix = (
            np.vstack(rows).astype(np.float32) if rows
            else np.zeros((0, len(FUSION_FEATURE_NAMES)), dtype=np.float32)
        )
        decision_vector = decision_features(
            families=state.families, plan=state.plan, risk=state.risk,
            union=state.union, rerank=state.rerank, fusion=state.fusion,
        )

        out.append(QueryPass(
            scenario_id=item.scenario_id,
            query=item.query,
            archetype=archetypes.get(item.scenario_id, "unknown"),
            fold=folds.get(item.scenario_id, 0),
            instance_ids=instance_ids,
            fusion_features=fusion_matrix,
            decision_features=decision_vector,
            grades=dict(item.grades),
            rrf_ranking=list(instance_ids),
            fallback_decision=state.outcome.decision.value,
            families_ranked=[f.family_id for f in state.families],
            selected_by_family={
                f.family_id: f.selected.instance_id for f in state.families
            },
            risk_band=state.risk.risk_band if state.risk else "LOW",
            hard_gate_decision=(
                state.outcome.decision.value
                if was_hard_gated(state.decision_result) else None
            ),
            **_fallback_inputs(state),
        ))
    if progress is not None:
        progress(len(labelled), len(labelled))
    return out


# --- caching -----------------------------------------------------------------


def cache_key(*, bundle_version: str, dataset_version: str, split: str) -> str:
    """Everything that changes what a pass produces.

    Both feature hashes are in the key: a feature-schema change makes every
    cached row wrong in a way that no shape check would catch, because the width
    is unchanged when a name is merely redefined.
    """
    return "-".join([
        CACHE_SCHEMA_VERSION, bundle_version or "nobundle", dataset_version,
        split, FUSION_FEATURE_HASH, DECISION_FEATURE_HASH,
    ])


def save_passes(passes: list[QueryPass], directory: Path, *, key: str) -> None:
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)

    # One concatenated matrix plus row offsets, rather than one npz member per
    # query. The per-query features are ragged (a query's candidate count varies),
    # and this keeps the file to three arrays regardless of how many queries there
    # are -- which matters at ~1,500 of them.
    width = len(FUSION_FEATURE_NAMES)
    fusion = (
        np.vstack([p.fusion_features for p in passes if len(p.fusion_features)])
        if any(len(p.fusion_features) for p in passes)
        else np.zeros((0, width), dtype=np.float32)
    )
    offsets = np.cumsum(
        [0, *[len(p.fusion_features) for p in passes]], dtype=np.int64
    )
    decision = (
        np.vstack([p.decision_features for p in passes]) if passes
        else np.zeros((0, 0), dtype=np.float32)
    )
    np.savez_compressed(
        directory / "features.npz",
        fusion=fusion, fusion_offsets=offsets, decision=decision,
    )
    (directory / "passes.json").write_text(json.dumps({
        "key": key,
        "passes": [
            {
                "scenario_id": p.scenario_id, "query": p.query,
                "archetype": p.archetype, "fold": p.fold,
                "instance_ids": p.instance_ids, "grades": p.grades,
                "rrf_ranking": p.rrf_ranking,
                "fallback_decision": p.fallback_decision,
                "families_ranked": p.families_ranked,
                "selected_by_family": p.selected_by_family,
                "ce_top": p.ce_top, "ce_second": p.ce_second,
                "fusion_top": p.fusion_top, "fusion_second": p.fusion_second,
                "risk_band": p.risk_band,
                "has_distinguishing_facet": p.has_distinguishing_facet,
                "hard_gate_decision": p.hard_gate_decision,
            }
            for p in passes
        ],
    }, indent=2, sort_keys=True) + "\n")


def load_passes(directory: Path, *, key: str) -> list[QueryPass] | None:
    """Cached passes, or None when the cache does not describe this build."""
    directory = Path(directory)
    meta_path = directory / "passes.json"
    if not meta_path.exists():
        return None

    meta = json.loads(meta_path.read_text())
    if meta.get("key") != key:
        log.info(
            "feature_cache_stale",
            extra={"expected": key, "found": meta.get("key")},
        )
        return None

    arrays = np.load(directory / "features.npz")
    decision, fusion = arrays["decision"], arrays["fusion"]
    offsets = arrays["fusion_offsets"]
    out = []
    for index, row in enumerate(meta["passes"]):
        start, end = int(offsets[index]), int(offsets[index + 1])
        out.append(QueryPass(
            scenario_id=row["scenario_id"], query=row["query"],
            archetype=row["archetype"], fold=row["fold"],
            instance_ids=row["instance_ids"],
            fusion_features=fusion[start:end],
            decision_features=decision[index],
            grades=row["grades"], rrf_ranking=row["rrf_ranking"],
            fallback_decision=row["fallback_decision"],
            families_ranked=row["families_ranked"],
            selected_by_family=row["selected_by_family"],
            ce_top=row["ce_top"], ce_second=row["ce_second"],
            fusion_top=row["fusion_top"], fusion_second=row["fusion_second"],
            risk_band=row["risk_band"],
            has_distinguishing_facet=row["has_distinguishing_facet"],
            hard_gate_decision=row.get("hard_gate_decision"),
        ))
    return out
