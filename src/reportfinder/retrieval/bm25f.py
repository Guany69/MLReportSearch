"""Small, dependency-free BM25F implementation over report zones."""

from __future__ import annotations

import math
import re
from collections import Counter, defaultdict

import numpy as np

TOKEN_RE = re.compile(r"[a-z0-9]+")


def tokenize(value: object) -> list[str]:
    if isinstance(value, (list, tuple)):
        value = " ".join(map(str, value))
    return TOKEN_RE.findall(str(value or "").casefold())


class BM25FIndex:
    """Robertson-style BM25F: weighted normalized term frequencies per zone."""

    def __init__(self, frame, weights: dict[str, float], k1: float = 1.2, b: float = .75):
        self.n = len(frame)
        self.weights = {z: w for z, w in weights.items() if w > 0}
        self.k1, self.b = k1, b
        self.docs: dict[str, list[Counter[str]]] = {}
        self.postings: dict[str, dict[int, float]] = defaultdict(dict)
        self.avg_len: dict[str, float] = {}
        document_frequency: Counter[str] = Counter()
        for zone in self.weights:
            values = frame[zone] if zone in frame else [""] * self.n
            counters = [Counter(tokenize(v)) for v in values]
            self.docs[zone] = counters
            self.avg_len[zone] = max(1.0, sum(map(lambda c: sum(c.values()), counters)) / max(1, self.n))
        for i in range(self.n):
            terms = set().union(*(self.docs[z][i] for z in self.weights))
            document_frequency.update(terms)
        self.idf = {t: math.log(1 + (self.n - df + .5) / (df + .5)) for t, df in document_frequency.items()}
        for zone, counters in self.docs.items():
            for doc_id, counter in enumerate(counters):
                norm = 1 - self.b + self.b * sum(counter.values()) / self.avg_len[zone]
                for term, count in counter.items():
                    self.postings[term][doc_id] = self.postings[term].get(doc_id, 0.0) + self.weights[zone] * count / norm

    def scores(self, query: str) -> np.ndarray:
        scores = np.zeros(self.n, dtype=np.float64)
        for term in set(tokenize(query)):
            idf = self.idf.get(term)
            if idf is None:
                continue
            for doc_id, tf in self.postings.get(term, {}).items():
                scores[doc_id] += idf * ((self.k1 + 1) * tf) / (self.k1 + tf)
        return scores

    def search(self, query: str, k: int = 100) -> list[tuple[int, float]]:
        scores = self.scores(query)
        order = np.argsort(-scores, kind="stable")[:k]
        return [(int(i), float(scores[i])) for i in order if scores[i] > 0]
