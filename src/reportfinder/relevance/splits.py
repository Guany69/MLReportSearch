from __future__ import annotations

import math
import re
from collections import Counter
from collections.abc import Iterable, Mapping


class SplitLeakageError(RuntimeError):
    pass


class SplitGuard:
    """Blocks sealed-test use in every model-development operation."""
    DEVELOPMENT_OPERATIONS = frozenset({"feature_fitting", "ranker_training", "hyperparameter_selection",
        "threshold_tuning", "calibration", "model_comparison", "early_stopping", "alias_mining"})
    # Operations that are *allowed* to touch the sealed split. Naming them is what
    # makes the guard falsifiable: an operation outside both sets is a typo, and a
    # typo'd operation name used to be a silent no-op guard.
    NON_DEVELOPMENT_OPERATIONS = frozenset({"final_evaluation"})
    KNOWN_OPERATIONS = DEVELOPMENT_OPERATIONS | NON_DEVELOPMENT_OPERATIONS

    # Every split that must be disjoint from every other. `calibration` is in here
    # because a temperature fitted on rows the model trained on is not calibration,
    # and it was previously the one split nothing compared.
    GUARDED_SPLITS = ("train", "validation", "test", "calibration")

    def __init__(self, splits: Mapping[str, Iterable[str]]):
        self.splits = {name: frozenset(ids) for name, ids in splits.items()}
        for index, left in enumerate(self.GUARDED_SPLITS):
            for right in self.GUARDED_SPLITS[index + 1:]:
                overlap = self.splits.get(left, frozenset()) & self.splits.get(right, frozenset())
                if overlap:
                    raise SplitLeakageError(
                        f"scenario split overlap {left}/{right}: {sorted(overlap)[:10]}"
                    )

    def assert_allowed(self, scenario_ids, operation: str) -> None:
        if operation not in self.KNOWN_OPERATIONS:
            raise ValueError(
                f"unknown split-guard operation {operation!r}; a name outside "
                f"{sorted(self.KNOWN_OPERATIONS)} guards nothing at all"
            )
        leaked = set(scenario_ids) & self.splits.get("test", frozenset())
        if leaked and operation in self.DEVELOPMENT_OPERATIONS:
            raise SplitLeakageError(f"sealed test scenarios passed to {operation}: {sorted(leaked)[:10]}")

    @staticmethod
    def assert_disjoint_positives(by_split: dict[str, set[str]], identity: str = "report") -> None:
        names = sorted(by_split)
        for i, left in enumerate(names):
            for right in names[i + 1:]:
                overlap = set(by_split[left]) & set(by_split[right])
                if overlap: raise SplitLeakageError(f"positive {identity} leakage {left}/{right}: {sorted(overlap)[:10]}")

    @staticmethod
    def trigram_cosine(left: str, right: str) -> float:
        def grams(text):
            normalized = re.sub(r"\s+", " ", text.casefold().strip())
            return Counter(normalized[i:i+3] for i in range(max(0, len(normalized)-2)))
        a, b = grams(left), grams(right)
        if not a or not b: return 1.0 if left.casefold().strip() == right.casefold().strip() else 0.0
        dot = sum(value * b.get(key, 0) for key, value in a.items())
        return dot / math.sqrt(sum(v*v for v in a.values()) * sum(v*v for v in b.values()))

    @classmethod
    def assert_no_near_duplicates(cls, queries_by_split: dict[str, dict[str, str]], threshold: float = .95) -> None:
        names = sorted(queries_by_split)
        for i, left in enumerate(names):
            for right in names[i+1:]:
                for lid, lq in queries_by_split[left].items():
                    for rid, rq in queries_by_split[right].items():
                        if cls.trigram_cosine(lq, rq) >= threshold:
                            raise SplitLeakageError(f"near-duplicate query leakage {left}:{lid}/{right}:{rid}")
