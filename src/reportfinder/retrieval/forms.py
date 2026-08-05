"""Weighted query forms and bounded aggregation on one shared score scale."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Literal, Sequence

import numpy as np


@dataclass(frozen=True)
class QueryForm:
    text: str
    weight: float
    kind: Literal["original", "expanded", "subquery"]
    key: str


@dataclass(frozen=True)
class ChannelResult:
    combined: np.ndarray
    per_form: np.ndarray
    winning_form: np.ndarray


def form_key(text: str) -> str:
    return " ".join(re.findall(r"[a-z0-9]+", text.casefold()))


def build_forms(original: str, expanded: str, subqueries: Sequence[str], *,
                expansion_weight: float = .85, subquery_weight: float = .75) -> tuple[QueryForm, ...]:
    candidates = [QueryForm(original, 1.0, "original", form_key(original))]
    if expanded and form_key(expanded) != form_key(original):
        candidates.append(QueryForm(expanded, expansion_weight, "expanded", form_key(expanded)))
    candidates.extend(
        QueryForm(text, subquery_weight, "subquery", form_key(text))
        for text in subqueries if form_key(text)
    )
    seen: dict[str, QueryForm] = {}
    for form in candidates:
        previous = seen.get(form.key)
        if previous is None or form.weight > previous.weight:
            seen[form.key] = form
    return tuple(sorted(seen.values(), key=lambda f: (-f.weight, f.kind, f.key)))


def score_forms(scorer, forms: Sequence[QueryForm]) -> np.ndarray:
    scores = np.stack([np.asarray(scorer(form.text), dtype=np.float64) for form in forms])
    scores = np.where(np.isfinite(scores) & (scores > 0), scores, 0.0)
    maximum = float(scores.max(initial=0.0))
    return scores / maximum if maximum > 0 else scores


def aggregate_top_two(scores: np.ndarray, weights: Sequence[float],
                      lam: float = .30) -> ChannelResult:
    values = scores * np.asarray(weights, dtype=np.float64)[:, None]
    if values.shape[0] == 1:
        first = values[0]
        second = np.zeros_like(first)
    else:
        partitioned = np.partition(values, -2, axis=0)
        first, second = partitioned[-1], partitioned[-2]
    return ChannelResult(
        combined=(first + lam * second) / (1.0 + lam),
        per_form=scores,
        winning_form=values.argmax(axis=0),
    )
