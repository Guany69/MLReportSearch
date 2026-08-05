"""Character n-gram TF-IDF retriever for misspellings and abbreviations."""

from __future__ import annotations

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer


class CharNgramIndex:
    def __init__(self, documents: list[str], ngram_range: tuple[int, int] = (3, 5)):
        self.vectorizer = TfidfVectorizer(analyzer="char_wb", ngram_range=ngram_range, min_df=1, sublinear_tf=True)
        self.matrix = self.vectorizer.fit_transform(documents)

    def scores(self, query: str) -> np.ndarray:
        return (self.matrix @ self.vectorizer.transform([query]).T).toarray().ravel()

    def search(self, query: str, k: int = 100) -> list[tuple[int, float]]:
        scores = self.scores(query)
        order = np.argsort(-scores, kind="stable")[:k]
        return [(int(i), float(scores[i])) for i in order if scores[i] > 0]
