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
