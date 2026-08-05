"""Final scoring: learned listwise fusion, or an explicit RRF fallback.

No trained fusion artifact ships with this repository, so the RRF fallback is what
actually runs. That is stated in the manifest, in telemetry, and in the API
response. Three things this module refuses to do:

* initialize random weights and call the result a trained model;
* describe any of these scores as a probability -- they are ordering scores, and a
  softmax over them would look like confidence without being calibrated;
* mix raw generator scores on their native scales. The fallback fuses **ranks**,
  including the cross-encoder's rank, so a BM25F score and a cosine never meet.

The feature builder is shared between serving and training so the learned model, if
one is ever trained, sees exactly what serving can produce. Missing generators
arrive as (value, present) pairs rather than an imputed 0, because a model given 0
learns "this generator ranked it last" instead of "this generator did not see it".
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import numpy as np

# Feature layout. Versioned and hashed like `ranking/features.py` so a trained
# artifact can be rejected when the feature space moves underneath it.
FUSION_FEATURE_SCHEMA_VERSION = "1"

GENERATOR_ORDER = (
    "bm25f", "splade", "dense_identity", "dense_purpose", "dense_schema",
    "dense_interface", "prototype", "late_interaction",
)


def _generator_feature_names() -> list[str]:
    names: list[str] = []
    for generator in GENERATOR_ORDER:
        # Rank and score always travel with an explicit presence flag.
        names.append(f"{generator}_reciprocal_rank")
        names.append(f"{generator}_score")
        names.append(f"{generator}_present")
    return names


FUSION_FEATURE_NAMES = tuple([
    *_generator_feature_names(),
    "cross_encoder_score",
    "cross_encoder_reciprocal_rank",
    "cross_encoder_present",
    "fused_reciprocal_rank",
    "generator_count",
    "query_variant_count",
    "source_exclusive",
    "exact_title_overlap",
    "alias_overlap",
    "field_overlap",
    "prompt_overlap",
    "query_length",
    "query_specificity",
    "metadata_completeness",
    "lifecycle_unknown",
    "family_instance_count",
])


def fusion_feature_hash() -> str:
    import hashlib

    payload = "|".join([FUSION_FEATURE_SCHEMA_VERSION, *FUSION_FEATURE_NAMES])
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


FUSION_FEATURE_HASH = fusion_feature_hash()


def _tokens(text: str) -> set[str]:
    import re

    return set(re.findall(r"[a-z0-9]+", (text or "").casefold()))


def _overlap(a: set[str], b: set[str]) -> float:
    return len(a & b) / len(a) if a else 0.0


def build_features(record, instance, family, rerank, plan) -> np.ndarray:
    """Serving-time features for one candidate.

    Everything here is available at serving time. Nothing derived from gold labels,
    relevance judgements, future user actions or training-only annotations may enter
    -- a feature that is not computable during a live request is a train/serve skew
    waiting to happen.
    """
    values: list[float] = []
    for generator in GENERATOR_ORDER:
        rank = record.ranks.get(generator)
        score = record.scores.get(generator)
        present = 1.0 if rank else 0.0
        values.append(1.0 / rank if rank else 0.0)
        values.append(float(score) if score is not None else 0.0)
        values.append(present)

    ce_score = rerank.scores.get(record.instance_id)
    ce_rank = rerank.ranks.get(record.instance_id)
    values.append(float(ce_score) if ce_score is not None else 0.0)
    values.append(1.0 / ce_rank if ce_rank else 0.0)
    values.append(1.0 if ce_rank else 0.0)

    values.append(1.0 / record.fused_rank if record.fused_rank else 0.0)
    values.append(float(record.generator_count))
    values.append(float(len(record.query_variants)))
    values.append(1.0 if record.source_exclusive else 0.0)

    query_tokens = _tokens(plan.raw_query)
    values.append(_overlap(query_tokens, _tokens(instance.title)))
    values.append(_overlap(query_tokens, _tokens(" ".join(family.aliases))))
    values.append(_overlap(query_tokens, _tokens(" ".join(instance.fields))))
    values.append(_overlap(query_tokens, _tokens(" ".join(instance.prompts))))

    values.append(float(len(query_tokens)))
    # Specificity proxy: longer, rarer-looking queries state more.
    values.append(float(len(plan.stated_facets())))

    filled = sum(
        1.0 for value in (
            instance.description, instance.category, instance.data_source,
            instance.report_type, instance.tags, instance.area_where_used,
        ) if value
    )
    values.append(filled / 6.0)
    values.append(1.0 if instance.lifecycle_state.value == "UNKNOWN" else 0.0)
    values.append(float(len(family.instance_ids)))

    return np.asarray(values, dtype=np.float32)


@dataclass
class FusionResult:
    """Final instance ordering, and which fuser produced it."""

    scores: dict[str, float]
    ranks: dict[str, int]
    method: str
    fallback_active: bool
    # Never a probability. Named to make that hard to forget.
    score_kind: str = "unnormalized_ranking_score"

    def telemetry(self) -> dict[str, object]:
        return {
            "fusion_method": self.method,
            "fusion_fallback_active": self.fallback_active,
            "fusion_score_kind": self.score_kind,
        }


@runtime_checkable
class Fuser(Protocol):
    name: str
    fallback_active: bool

    def fuse(self, shortlist, union, rerank, corpus, plan) -> FusionResult: ...


class RrfFuser:
    """The declared fallback: the reranker orders, RRF breaks ties.

    Why not simply add the cross-encoder into the RRF sum as another ranked source:
    rank fusion deliberately discards score magnitude, so a candidate the reranker
    scored far higher than the next one still only gains a single rank. At k=60
    that is worth 1/61 - 1/62 ~= 0.0003, while two extra retrievers agreeing are
    worth 0.033 -- a hundred times more. Under a pure rank sum the only stage that
    actually reads the query and the report together is reliably outvoted by
    retrievers that never did, and no weight fixes it robustly because the gap it
    has to overcome is a magnitude gap compressed into a rank gap.

    So the fallback orders by cross-encoder rank and uses the RRF sum to break ties
    and to order anything the reranker did not score. That is still purely
    rank-based -- no logit is ever compared to an RRF score -- and it keeps
    retrieval agreement as the tie-breaker rather than the decider.

    This affects *ordering only*. Every candidate here already survived retrieval,
    the union and the shortlist, so the reranker is not gating recall; the learned
    fuser is what will eventually combine these signals properly.
    """

    name = "rrf"
    fallback_active = True

    def __init__(
        self,
        *,
        constant: int = 60,
        cross_encoder_priority: bool = True,
        cross_encoder_weight: float = 4.0,
    ) -> None:
        self.constant = constant
        self.cross_encoder_priority = cross_encoder_priority
        # Only used when `cross_encoder_priority` is False, i.e. the pure weighted
        # rank-sum ablation. Retained so that variant can be measured.
        self.cross_encoder_weight = cross_encoder_weight

    def fuse(self, shortlist, union, rerank, corpus, plan) -> FusionResult:
        rrf: dict[str, float] = {}
        for entry in shortlist:
            record = entry.record
            rrf[record.instance_id] = sum(
                1.0 / (self.constant + rank) for rank in record.ranks.values() if rank
            )

        if not self.cross_encoder_priority or not rerank.ran:
            scores = dict(rrf)
            if rerank.ran:
                for instance_id, ce_rank in rerank.ranks.items():
                    if instance_id in scores:
                        scores[instance_id] += self.cross_encoder_weight / (
                            self.constant + ce_rank
                        )
            ordered = sorted(scores.items(), key=lambda item: (-item[1], item[0]))
            return FusionResult(
                scores=scores,
                ranks={i: r for r, (i, _) in enumerate(ordered, start=1)},
                method="rrf_rank_sum",
                fallback_active=True,
            )

        # Reranked candidates first, in reranker order; then anything it did not
        # score, in RRF order. Sorting on ranks throughout keeps this scale-safe.
        unscored_rank = len(shortlist) + 1
        ordered_ids = sorted(
            rrf,
            key=lambda i: (rerank.ranks.get(i, unscored_rank), -rrf[i], i),
        )
        # The published score is a descending ordering score, not a probability and
        # not the reranker's logit.
        scores = {
            instance_id: float(len(ordered_ids) - position)
            for position, instance_id in enumerate(ordered_ids)
        }
        return FusionResult(
            scores=scores,
            ranks={i: r for r, i in enumerate(ordered_ids, start=1)},
            method="rrf_with_cross_encoder_priority",
            fallback_active=True,
        )


class NeuralFuser:
    """Listwise MLP fusion. Only constructed when a trained artifact exists."""

    name = "neural"
    fallback_active = False

    def __init__(self, model, *, feature_hash: str) -> None:
        if feature_hash != FUSION_FEATURE_HASH:
            raise ValueError(
                "fusion artifact was trained against a different feature space "
                f"({feature_hash} != {FUSION_FEATURE_HASH}); refusing to serve it"
            )
        self.model = model

    def fuse(self, shortlist, union, rerank, corpus, plan) -> FusionResult:
        import torch

        entries = list(shortlist)
        if not entries:
            return FusionResult({}, {}, "neural", False)

        features = np.vstack([
            build_features(
                entry.record,
                corpus.instance(entry.instance_id),
                corpus.family_of(entry.instance_id),
                rerank, plan,
            )
            for entry in entries
        ])
        with torch.inference_mode():
            raw = self.model(torch.from_numpy(features)).reshape(-1).numpy()

        scores = {
            entry.instance_id: float(value) for entry, value in zip(entries, raw, strict=True)
        }
        ordered = sorted(scores.items(), key=lambda item: (-item[1], item[0]))
        return FusionResult(
            scores=scores,
            ranks={i: r for r, (i, _) in enumerate(ordered, start=1)},
            method="neural_mlp_64_32_1",
            fallback_active=False,
        )


def build_fuser(cfg) -> Fuser:
    """Load the trained fuser if one is configured, otherwise the RRF fallback."""
    if cfg.fusion.artifact_path is None:
        return RrfFuser(constant=cfg.retrieval.rrf_constant)

    from ..training.nets import load_fusion_model

    model, metadata = load_fusion_model(cfg.fusion.artifact_path)
    return NeuralFuser(model, feature_hash=metadata["feature_hash"])
