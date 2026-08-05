from __future__ import annotations

import math

import numpy as np


def ranking_metrics(ranked: list[str], relevant: dict[str, int]) -> dict[str, float]:
    binary = [int(doc in relevant) for doc in ranked]
    gains = [relevant.get(doc, 0) for doc in ranked]
    out = {}
    for k in (1, 5, 10):
        out[f"hit@{k}"] = float(any(binary[:k]))
    for k in (10, 20, 50): out[f"recall@{k}"] = sum(binary[:k]) / max(1, len(relevant))
    out["mrr@10"] = next((1/i for i, hit in enumerate(binary[:10], 1) if hit), 0.0)
    for k in (5, 10):
        dcg = sum((2**g-1)/math.log2(i+2) for i, g in enumerate(gains[:k]))
        ideal = sum((2**g-1)/math.log2(i+2) for i, g in enumerate(sorted(relevant.values(), reverse=True)[:k]))
        out[f"ndcg@{k}"] = dcg/ideal if ideal else 0.0
    out["p@5"] = sum(binary[:5])/5
    precisions = [sum(binary[:i])/i for i in range(1, len(binary)+1) if binary[i-1]]
    out["map"] = sum(precisions)/max(1, len(relevant))
    return out

def brier(scores, labels): return float(np.mean((np.asarray(scores)-np.asarray(labels))**2))
def binary_nll(scores, labels):
    p=np.clip(np.asarray(scores,dtype=float),1e-7,1-1e-7); y=np.asarray(labels,dtype=float)
    return float(np.mean(-(y*np.log(p)+(1-y)*np.log(1-p))))
def risk_coverage(scores, correct):
    # Stable: the curve is a prefix mean, so tie order changes the reported risk at
    # every coverage point a tie straddles -- and an uncalibrated policy score ties
    # constantly.
    order=np.argsort(-np.asarray(scores), kind="stable"); hits=np.asarray(correct,dtype=float)[order]
    return [{"coverage":float(i/len(hits)),"risk":float(1-hits[:i].mean())} for i in range(1,len(hits)+1)] if len(hits) else []

def discordant_pairs(reference, correct) -> int:
    """Pairs two scorings order strictly *oppositely*.

    An ordering comparison that sorts both sides and diffs the index lists counts
    a tie broken two different ways as a disagreement, which it is not: neither
    scorer expressed a preference. This compares sign of difference per pair, so
    only a genuine inversion counts.
    """
    a, b = np.asarray(reference, dtype=float), np.asarray(correct, dtype=float)
    if a.shape != b.shape:
        raise ValueError(f"orderings differ in length: {a.shape} vs {b.shape}")
    left, right = np.sign(a[:, None] - a[None, :]), np.sign(b[:, None] - b[None, :])
    return int((np.triu(left * right, k=1) < 0).sum())
def expected_calibration_error(scores, labels, bins=10):
    scores, labels = np.asarray(scores), np.asarray(labels)
    total = max(1, len(scores)); value = 0.0
    for low in np.linspace(0, 1, bins, endpoint=False):
        mask = (scores >= low) & (scores < low + 1/bins)
        if mask.any(): value += mask.sum()/total * abs(scores[mask].mean()-labels[mask].mean())
    return float(value)
