"""Candidate fusion methods. Scores returned here are ranking scores, not probabilities."""

from __future__ import annotations

from collections import defaultdict


def reciprocal_rank_fusion(rankings: dict[str, list[tuple[int, float]]], k: int = 60) -> dict[int, float]:
    fused: dict[int, float] = defaultdict(float)
    for results in rankings.values():
        for rank, (doc_id, _) in enumerate(results, 1):
            fused[doc_id] += 1.0 / (k + rank)
    return dict(fused)
