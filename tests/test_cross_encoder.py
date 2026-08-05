from __future__ import annotations

from reportfinder.ranking import cross_encoder


def test_reranker_is_initialized_once(monkeypatch):
    created = []

    class Stub:
        def __init__(self, name):
            created.append(name)

    cross_encoder.get_reranker.cache_clear()
    monkeypatch.setattr(cross_encoder, "CrossEncoderReranker", Stub)
    assert cross_encoder.get_reranker("stub") is cross_encoder.get_reranker("stub")
    assert created == ["stub"]
    cross_encoder.get_reranker.cache_clear()


def test_centered_weight_can_demote_head_below_untouched_tail():
    scored = [(1.00, 0, None), (.99, 1, None), (.98, 2, None)]
    reranked = cross_encoder.apply_rerank(scored, [0.0, 1.0], .10, 2)
    assert [item[1] for item in reranked] == [1, 2, 0]
    assert reranked == sorted(reranked, key=lambda item: (-item[0], item[1]))


# --- pair deduplication ------------------------------------------------------
#
# Reranking is the dominant measured stage (~85% of query latency on this host),
# and within one query the model input is `(query, text)` -- so two candidates
# whose authoritative text is byte-identical are the same forward pass run twice.
# These pin the fan-out contract. No latency figure is claimed here: the telemetry
# counters are the measurement hook, not the measurement.

import numpy as np
import pytest

from reportfinder.pipeline.rerank import score_pairs_deduplicated


class _Counting:
    """Scores by text length, and records exactly what it was asked to score."""

    name = "counting"
    revision = "v1"

    def __init__(self, scores=None):
        self.calls: list[list[str]] = []
        self._scores = scores

    def score_pairs(self, query, texts):
        self.calls.append(list(texts))
        if self._scores is not None:
            return np.asarray(self._scores, dtype=np.float32)
        return np.asarray([float(len(t)) for t in texts], dtype=np.float32)


def test_identical_texts_are_scored_once_and_fanned_back_out():
    scorer = _Counting()
    scores, model_pairs = score_pairs_deduplicated(
        scorer, "q", ["alpha", "beta", "alpha", "alpha"]
    )

    assert scorer.calls == [["alpha", "beta"]], "each distinct text once, in order"
    assert model_pairs == 2
    # ...and every candidate still gets its own score, in its own position.
    assert list(scores) == [5.0, 4.0, 5.0, 5.0]


def test_distinct_texts_are_never_merged():
    scorer = _Counting()
    _, model_pairs = score_pairs_deduplicated(scorer, "q", ["a", "bb", "ccc"])
    assert model_pairs == 3
    assert scorer.calls == [["a", "bb", "ccc"]]


def test_no_texts_never_reaches_the_model():
    scorer = _Counting()
    scores, model_pairs = score_pairs_deduplicated(scorer, "q", [])
    assert scorer.calls == [], "an empty shortlist must not load a model"
    assert scores.shape == (0,) and model_pairs == 0


def test_a_scorer_returning_the_wrong_count_raises_rather_than_misaligning():
    """Silently truncating or padding would attach one report's score to another
    -- a wrong answer that looks exactly like a right one."""
    scorer = _Counting(scores=[1.0, 2.0, 3.0])
    with pytest.raises(ValueError, match="one score per input"):
        score_pairs_deduplicated(scorer, "q", ["alpha", "beta"])


def test_deduplication_does_not_change_the_ranking(corpus_and_shortlist=None):
    """The property that makes this safe: same text implies same logit, so the
    fanned-out scores are identical to what per-candidate scoring produced."""
    scorer = _Counting()
    texts = ["alpha", "beta", "alpha", "gamma", "beta"]

    deduped, _ = score_pairs_deduplicated(scorer, "q", texts)
    naive = np.asarray([float(len(t)) for t in texts], dtype=np.float32)
    assert list(deduped) == list(naive)
