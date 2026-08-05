"""Normalized exact-field inverted index with corpus-IDF weighting."""

from __future__ import annotations

import math
import re
from collections import defaultdict
from dataclasses import dataclass
from typing import Sequence

import numpy as np


@dataclass(frozen=True)
class FieldScores:
    scores: np.ndarray
    matched: dict[int, tuple[str, ...]]
    required_mass: float
    resolved: tuple[str, ...]
    unknown: tuple[str, ...]
    required_combination_exists: bool = True


def normalize(text: str) -> str:
    return " ".join(re.findall(r"[a-z0-9]+", str(text).casefold()))


class FieldIndex:
    def __init__(self, frame):
        self.n = len(frame)
        self.postings: dict[str, list[int]] = defaultdict(list)
        for i, fields in enumerate(frame["fields"]):
            for field in set(map(normalize, fields)):
                if field:
                    self.postings[field].append(i)

    def detected(self, query: str) -> list[str]:
        padded = f" {normalize(query)} "
        return sorted((f for f in self.postings if f" {f} " in padded), key=lambda x: (-len(x), x))

    def scores(self, query: str) -> tuple[np.ndarray, list[str]]:
        found = self.detected(query)
        scores = np.zeros(self.n, dtype=np.float64)
        for field in found:
            weight = math.log(1 + self.n / len(self.postings[field]))
            scores[self.postings[field]] += weight
        return scores, found

    def scores_for_fields(self, required: Sequence[str], optional: Sequence[str] = (),
                          *, optional_weight: float = .5) -> FieldScores:
        required_normalized = tuple(dict.fromkeys(normalize(field) for field in required))
        optional_normalized = tuple(
            field for field in dict.fromkeys(normalize(field) for field in optional)
            if field not in required_normalized
        )
        resolved_required = tuple(field for field in required_normalized if field in self.postings)
        resolved_optional = tuple(field for field in optional_normalized if field in self.postings)
        unknown = tuple(dict.fromkeys(
            str(field) for field in tuple(required) + tuple(optional)
            if normalize(field) not in self.postings
        ))
        maximum_idf = math.log(1 + self.n)

        def weight(field: str) -> float:
            return math.log(1 + self.n / len(self.postings[field])) / maximum_idf

        required_mass = sum(weight(field) for field in resolved_required)
        total_mass = required_mass + optional_weight * sum(weight(field) for field in resolved_optional)
        scores = np.zeros(self.n, dtype=np.float64)
        matches: dict[int, list[str]] = defaultdict(list)
        for field in resolved_required:
            scores[self.postings[field]] += weight(field)
            for doc in self.postings[field]:
                matches[doc].append(field)
        for field in resolved_optional:
            scores[self.postings[field]] += optional_weight * weight(field)
            for doc in self.postings[field]:
                matches[doc].append(field)
        if total_mass:
            scores /= total_mass
        if required_normalized and len(resolved_required) == len(required_normalized):
            common = set(self.postings[resolved_required[0]])
            for field in resolved_required[1:]:
                common.intersection_update(self.postings[field])
            required_combination_exists = bool(common)
        else:
            required_combination_exists = not required_normalized
        return FieldScores(
            scores, {doc: tuple(values) for doc, values in matches.items()},
            required_mass, resolved_required + resolved_optional, unknown,
            required_combination_exists,
        )

    def search(self, query: str, k: int = 100) -> list[tuple[int, float]]:
        scores, _ = self.scores(query)
        order = np.argsort(-scores, kind="stable")[:k]
        return [(int(i), float(scores[i])) for i in order if scores[i] > 0]
