"""Deterministic stand-ins for every model in the pipeline.

The repository had no injection seam for models: everything was constructed inside
`ReportFinder._ensure_hybrid_indexes`, so any test wanting retrieval behaviour had
to load a real checkpoint or skip. These fakes close that gap, which is what lets
the union, shortlist, rerank and decision logic be tested exhaustively and offline.

`FakeEncoder` is the important one. It is not random: it embeds text into a space
where dimensions correspond to *concepts*, so a test can state "these two phrasings
should be neighbours" and have it be true by construction. That makes the vague-query
regression test meaningful -- the target is retrieved because the fake genuinely
places it near the query, not because a hash collision happened to work out.
"""

from __future__ import annotations

import hashlib
import re
from typing import Sequence

import numpy as np
import scipy.sparse as sp

_TOKEN = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> list[str]:
    return _TOKEN.findall(text.casefold())


# Concept groups. Every term in a group maps to the same dimension, so texts that
# talk about the same idea in different words end up pointing the same way. This is
# how the fake models "semantics" without a model.
CONCEPT_GROUPS: tuple[tuple[str, ...], ...] = (
    ("attrition", "termination", "terminated", "leave", "leaving", "losing", "leaver",
     "turnover", "exit", "separations"),
    ("backfill", "replacement", "refill", "vacancy", "requisition", "openings"),
    ("time", "lag", "duration", "fill", "speed", "faster", "days"),
    ("headcount", "count", "people", "workers", "employees", "staff", "roster"),
    ("manager", "supervisor", "boss", "leader", "supervisory"),
    ("pay", "salary", "compensation", "earnings", "wages", "payroll"),
    ("organization", "org", "department", "team", "company", "division"),
    ("location", "region", "country", "site", "office"),
    ("promotion", "promoted", "advancement", "career"),
    ("learning", "training", "course", "certification"),
)

_CONCEPT_OF: dict[str, int] = {
    term: index
    for index, group in enumerate(CONCEPT_GROUPS)
    for term in group
}
_CONCEPT_DIMS = len(CONCEPT_GROUPS)
# Remaining dimensions carry a stable hash of unmatched tokens, so unrelated text
# stays roughly orthogonal instead of collapsing onto the concept axes.
_HASH_DIMS = 22
FAKE_DIM = _CONCEPT_DIMS + _HASH_DIMS


def _embed(text: str) -> np.ndarray:
    vector = np.zeros(FAKE_DIM, dtype=np.float32)
    for token in tokenize(text):
        concept = _CONCEPT_OF.get(token)
        if concept is not None:
            vector[concept] += 1.0
        else:
            digest = hashlib.sha256(token.encode("utf-8")).digest()
            vector[_CONCEPT_DIMS + digest[0] % _HASH_DIMS] += 0.5
    norm = float(np.linalg.norm(vector))
    return vector / norm if norm else vector


class FakeEncoder:
    """Deterministic concept-space encoder. No downloads, no randomness."""

    def __init__(self, name: str = "fake-encoder", revision: str = "fake-rev") -> None:
        self.name = name
        self.revision = revision
        self.dim = FAKE_DIM
        self.document_calls: list[list[str]] = []
        self.query_calls: list[list[str]] = []

    def _encode(self, texts: Sequence[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        return np.vstack([_embed(t) for t in texts]).astype(np.float32)

    def encode_documents(self, texts: Sequence[str]) -> np.ndarray:
        self.document_calls.append(list(texts))
        return self._encode(texts)

    def encode_queries(self, texts: Sequence[str]) -> np.ndarray:
        self.query_calls.append(list(texts))
        return self._encode(texts)


class FakeSparseEncoder:
    """Token-to-term-weight encoder with a small deterministic expansion.

    Mirrors what SPLADE does structurally -- a term appears in the vector even when
    it is absent from the text -- without needing the model.
    """

    def __init__(self, name: str = "fake-splade", revision: str = "fake-rev",
                 vocab_size: int = 512) -> None:
        self.name = name
        self.revision = revision
        self.vocab_size = vocab_size

    def _term_id(self, token: str) -> int:
        return int(hashlib.sha256(token.encode("utf-8")).hexdigest()[:8], 16) % self.vocab_size

    def encode(self, texts: Sequence[str]) -> sp.csr_matrix:
        rows, cols, data = [], [], []
        for row, text in enumerate(texts):
            weights: dict[int, float] = {}
            for token in tokenize(text):
                weights[self._term_id(token)] = weights.get(self._term_id(token), 0.0) + 1.0
                # Learned-style expansion: co-members of a concept group get weight
                # even when the token itself never appears.
                concept = _CONCEPT_OF.get(token)
                if concept is not None:
                    for sibling in CONCEPT_GROUPS[concept]:
                        term = self._term_id(sibling)
                        weights[term] = max(weights.get(term, 0.0), 0.4)
            for term, weight in weights.items():
                rows.append(row)
                cols.append(term)
                data.append(weight)
        return sp.csr_matrix(
            (data, (rows, cols)), shape=(len(texts), self.vocab_size), dtype=np.float32
        )


class FakeTokenEncoder:
    """Per-token vectors for late-interaction tests."""

    def __init__(self, name: str = "fake-colbert", revision: str = "fake-rev",
                 max_tokens: int = 12) -> None:
        self.name = name
        self.revision = revision
        self.dim = FAKE_DIM
        self.max_tokens = max_tokens

    def _encode(self, texts: Sequence[str]) -> tuple[np.ndarray, np.ndarray]:
        vectors = np.zeros((len(texts), self.max_tokens, self.dim), dtype=np.float32)
        mask = np.zeros((len(texts), self.max_tokens), dtype=bool)
        for row, text in enumerate(texts):
            for position, token in enumerate(tokenize(text)[: self.max_tokens]):
                vectors[row, position] = _embed(token)
                mask[row, position] = True
        return vectors, mask

    def encode_documents(self, texts: Sequence[str]):
        return self._encode(texts)

    def encode_queries(self, texts: Sequence[str]):
        return self._encode(texts)


class ScriptedPairScorer:
    """A cross-encoder whose scores are stated by the test.

    Records every query string it is handed, which is how the pipeline's promise
    that the *raw* query reaches the cross-encoder is asserted rather than assumed.

    Scores are keyed by instance id, but a real cross-encoder only ever sees text --
    `authoritative_text` deliberately contains no identifiers. Passing `corpus`
    lets the fake resolve text back to an id exactly, without weakening the
    production contract to smuggle ids into what the model reads.
    """

    def __init__(
        self,
        scores: dict[str, float] | None = None,
        default: float = 0.0,
        *,
        corpus=None,
        max_report_chars: int = 1200,
    ) -> None:
        self.scores = scores or {}
        self.default = default
        self.seen_queries: list[str] = []
        self.seen_texts: list[list[str]] = []
        self.pair_count = 0
        self.name = "scripted-cross-encoder"
        self.revision = "fake-rev"

        self._by_text: dict[str, str] = {}
        if corpus is not None:
            from reportfinder.corpus import authoritative_text

            self._by_text = {
                authoritative_text(instance, max_chars=max_report_chars):
                    instance.report_instance_id
                for instance in corpus.instances
            }

    def score_pairs(self, query: str, texts: Sequence[str]) -> np.ndarray:
        self.seen_queries.append(query)
        self.seen_texts.append(list(texts))
        self.pair_count += len(texts)
        return np.array(
            [self.scores.get(self._key(t), self.default) for t in texts],
            dtype=np.float32,
        )

    def _key(self, text: str) -> str:
        resolved = self._by_text.get(text)
        if resolved is not None:
            return resolved
        # Fall back to an id embedded in the text, for tests that build their own.
        match = re.search(r"\bR\d{4}\b", text)
        return match.group(0) if match else text


class ScriptedGenerator:
    """A generator that returns exactly what the test says it found."""

    def __init__(self, name: str, results: dict[str, list[tuple[str, float]]],
                 *, view_type: str | None = None, query_variant: str = "Q0") -> None:
        self.name = name
        self.results = results
        self.view_type = view_type
        self.query_variant = query_variant
        self.calls: list[str] = []

    def generate(self, plan, universe, k):
        from reportfinder.generators.base import GeneratorResult

        self.calls.append(plan.raw_query)
        hits = self.results.get(plan.raw_query, [])
        return GeneratorResult(
            generator=self.name,
            query_variant=self.query_variant,
            view_type=self.view_type,
            hits=tuple(
                (instance_id, rank, score)
                for rank, (instance_id, score) in enumerate(hits[:k], start=1)
            ),
        )


class FailingGenerator:
    """A generator that always raises, for optional-generator failure tests."""

    def __init__(self, name: str = "exploding") -> None:
        self.name = name
        self.view_type = None
        self.query_variant = "Q0"

    def generate(self, plan, universe, k):
        raise RuntimeError("generator backend unavailable")
