"""Optional process-wide cross-encoder; imported only when explicitly enabled."""

import functools

import numpy as np


class CrossEncoderReranker:
    def __init__(self, model_name: str):
        from sentence_transformers import CrossEncoder
        self.model = CrossEncoder(model_name)
    def scores(self, query: str, rows):
        texts = [f"Title: {r['title']}\nFields: {', '.join(r['fields'])}\nDescription: {r.get('description', '')}" for r in rows]
        return self.model.predict([(query, text) for text in texts])


@functools.lru_cache(maxsize=2)
def get_reranker(model_name: str) -> CrossEncoderReranker:
    return CrossEncoderReranker(model_name)


def apply_rerank(scored, raw_cross_scores, weight: float, top_k: int):
    """Blend a centered cross-encoder signal and globally restore score order."""
    count = min(top_k, len(scored))
    values = np.asarray(raw_cross_scores, dtype=np.float64)
    if values.size and float(np.ptp(values)) > 0:
        values = (values - values.min()) / np.ptp(values)
    else:
        values = np.zeros_like(values)
    head = [
        (score + weight * (float(values[position]) - .5), idx, features)
        for position, (score, idx, features) in enumerate(scored[:count])
    ]
    return sorted(head + scored[count:], key=lambda item: (-item[0], item[1]))
