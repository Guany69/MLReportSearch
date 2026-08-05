"""Out-of-fold feature generation, and calibration.

Fusion features describe how the *pipeline* behaved on a query: which generators
found the candidate, at what rank, what the reranker said. If those features were
produced by a pipeline already tuned on the same query, the fusion model learns
from an optimistic view of retrieval it will never see in production.

So features are generated fold-wise: for each fold, the pipeline runs with the
fallback fuser active (never a partially-trained one), and only the held-out
queries' features are kept.

Temperature scaling for the decision head lives here too, on a split used for
nothing else. Fitting it on the training or validation data would produce a
temperature that flatters the model rather than calibrating it.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class FoldAssignment:
    """Which fold each query belongs to, grouped so duplicates travel together."""

    scenario_ids: list[str]
    folds: np.ndarray

    def held_out(self, fold: int) -> list[str]:
        return [s for s, f in zip(self.scenario_ids, self.folds, strict=True) if f == fold]


def assign_folds(scenario_ids: list[str], clusters: list[int], n_splits: int = 5,
                 seed: int = 20260720) -> FoldAssignment:
    """Group-aware folds: near-duplicate queries share a fold.

    Grouping on the duplicate cluster rather than the scenario id is what stops a
    paraphrase of a held-out query from sitting in the fold used to fit features
    for it.
    """
    from sklearn.model_selection import GroupKFold

    if not scenario_ids:
        return FoldAssignment([], np.zeros(0, dtype=int))

    indices = np.arange(len(scenario_ids))
    folds = np.zeros(len(scenario_ids), dtype=int)
    splitter = GroupKFold(n_splits=min(n_splits, len(set(clusters))))
    for fold, (_, held_out) in enumerate(
        splitter.split(indices, groups=np.asarray(clusters))
    ):
        folds[held_out] = fold
    return FoldAssignment(list(scenario_ids), folds)


def generate_out_of_fold_features(
    pipeline,
    labelled,
    assignment: FoldAssignment,
    *,
    principal,
    build_features,
    progress=None,
):
    """Run the pipeline per fold and collect held-out features.

    The pipeline keeps its fallback fuser throughout: features must describe
    retrieval as it behaves *without* the model being trained, or the model is
    learning to correct its own earlier output.
    """
    from ..auth import SearchRequest

    by_scenario = {q.scenario_id: q for q in labelled}
    examples = []

    for fold in sorted(set(assignment.folds.tolist())):
        for scenario_id in assignment.held_out(fold):
            query = by_scenario.get(scenario_id)
            if query is None or not query.query.strip():
                continue
            outcome = pipeline.run(SearchRequest(query.query, principal))
            telemetry = outcome.telemetry
            if telemetry is None:
                continue

            instance_ids, rows = [], []
            for family in outcome.families:
                for instance in family.instances:
                    instance_ids.append(instance.instance_id)
                    rows.append(
                        build_features(
                            instance.record,
                            pipeline.corpus.instance(instance.instance_id),
                            pipeline.corpus.family_of(instance.instance_id),
                            _RerankView(outcome),
                            _PlanView(query.query),
                        )
                    )
            if not rows:
                continue
            examples.append(
                (scenario_id, instance_ids, np.vstack(rows), query.grades)
            )
            if progress is not None:
                progress(scenario_id, fold)
    return examples


class _RerankView:
    """Adapts a finished outcome back to the rerank interface `build_features` wants."""

    def __init__(self, outcome) -> None:
        self.scores = {
            i.instance_id: i.cross_encoder_score
            for f in outcome.families for i in f.instances
            if i.cross_encoder_score is not None
        }
        ordered = sorted(self.scores.items(), key=lambda kv: -kv[1])
        self.ranks = {k: r for r, (k, _) in enumerate(ordered, start=1)}


class _PlanView:
    def __init__(self, raw_query: str) -> None:
        self.raw_query = raw_query

    def stated_facets(self):
        return {}


def fit_temperature(logits: np.ndarray, labels: np.ndarray, *, max_iter: int = 200) -> float:
    """Temperature scaling on a dedicated calibration split.

    One parameter, fitted by minimising NLL. Only after this is a softmax over the
    decision head's logits describable as a probability -- before it, the numbers
    are squashed scores that merely look like one.
    """
    import torch

    x = torch.from_numpy(np.asarray(logits, dtype=np.float32))
    y = torch.from_numpy(np.asarray(labels, dtype=np.int64))
    log_temperature = torch.zeros(1, requires_grad=True)
    optimizer = torch.optim.LBFGS([log_temperature], lr=0.1, max_iter=max_iter)

    def closure():
        optimizer.zero_grad()
        loss = torch.nn.functional.cross_entropy(x / log_temperature.exp(), y)
        loss.backward()
        return loss

    optimizer.step(closure)
    return float(log_temperature.exp().item())


def expected_calibration_error(probabilities: np.ndarray, labels: np.ndarray,
                               bins: int = 10) -> float:
    confidence = probabilities.max(axis=1)
    predicted = probabilities.argmax(axis=1)
    correct = (predicted == labels).astype(float)

    edges = np.linspace(0.0, 1.0, bins + 1)
    error = 0.0
    for low, high in zip(edges[:-1], edges[1:], strict=True):
        in_bin = (confidence > low) & (confidence <= high)
        if in_bin.sum() == 0:
            continue
        error += in_bin.mean() * abs(correct[in_bin].mean() - confidence[in_bin].mean())
    return float(error)
