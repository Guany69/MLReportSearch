"""Conservative, dependency-free OSA Damerau-Levenshtein typo correction."""

from __future__ import annotations

from collections import defaultdict

from ..normalize import depluralize

_STOP = frozenset(
    ["about", "after", "again", "all", "also", "and", "any", "are", "but", "can", "could", "did", "does", "each", "find", "for", "from", "get", "give", "how", "into", "list", "many", "may", "me", "need", "of", "on", "or", "please", "report", "reports", "show", "that", "the", "these", "this", "to", "want", "what", "where", "which", "who", "why", "with", "would"]
)
_SAFE_WORDS = frozenset(
    ["training", "earnings", "learning", "tracking", "hiring", "starter", "starters", "recent", "prior", "former", "old", "grouped", "organization", "supervisory", "overdue", "learner", "partner"]
)
_KNOWN_CORRECTIONS = {"hom": "home"}


def osa_distance(a: str, b: str, limit: int) -> int:
    if abs(len(a) - len(b)) > limit:
        return limit + 1
    previous2 = None
    previous = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        current = [i]
        for j, cb in enumerate(b, 1):
            value = min(current[j - 1] + 1, previous[j] + 1,
                        previous[j - 1] + (ca != cb))
            if previous2 is not None and i > 1 and j > 1 and ca == b[j - 2] and a[i - 2] == cb:
                value = min(value, previous2[j - 2] + 1)
            current.append(value)
        if min(current) > limit:
            return limit + 1
        previous2, previous = previous, current
    return previous[-1]


class CorrectionLexicon:
    def __init__(self, vocabulary, safe_tokens=()) -> None:
        targets = {
            token for token, df in vocabulary.token_df.items()
            if len(token) >= 5 and df >= 3
        }
        self.vocabulary = vocabulary
        self.targets = frozenset(targets)
        self.safe_tokens = frozenset(safe_tokens) | _SAFE_WORDS
        self.buckets: dict[tuple[str, int], list[str]] = defaultdict(list)
        for target in targets:
            for length in range(len(target) - 2, len(target) + 3):
                self.buckets[(target[0], length)].append(target)

    def correct(self, token: str, max_distance: int = 2):
        value = token.casefold()
        if value in self.safe_tokens or value in self.vocabulary.canonical_tokens or value in _STOP:
            return None
        if value in _KNOWN_CORRECTIONS:
            corrected = _KNOWN_CORRECTIONS[value]
            return corrected, 1, None, None, .90, "typo"
        if len(value) < 5:
            return None
        if not value.isalpha():
            return None
        singular = depluralize(value)
        if singular in self.vocabulary.canonical_tokens:
            return singular, 0, None, None, .95, "morphological"
        bound = 1 if len(value) < 8 else max_distance
        candidates = []
        for target in self.buckets.get((value[0], len(value)), ()):
            distance = osa_distance(value, target, bound)
            if distance == 2 and (
                len(value) < 10 or self.vocabulary.token_df.get(target, 0) < 25
            ):
                continue
            if distance <= bound:
                candidates.append((distance, target))
        candidates.sort()
        if not candidates:
            return None
        best_distance, best = candidates[0]
        same_best = [candidate for distance, candidate in candidates if distance == best_distance]
        if len(same_best) != 1:
            return None
        runner = candidates[1] if len(candidates) > 1 else None
        if runner is not None and runner[0] < best_distance + 1:
            return None
        confidence = .90 if best_distance == 1 else .60
        return best, best_distance, runner[1] if runner else None, runner[0] if runner else None, confidence, "typo"
