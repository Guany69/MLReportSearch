"""The ONNX cross-encoder backend, and the gate that lets it serve.

The backend exists for a measured reason: torch 2.11 on arm64 macOS 13 returns
NaN in float32 for realistic cross-encoder input, so the torch path escalates to
float64 -- exact, and the dominant term in query latency. ONNX Runtime executes
the same graph with different kernels and does not reproduce the NaN.

What must not happen is a faster backend that ranks differently, so the config
selection is pinned here and the numeric comparison lives in the parity gate
(`scripts/export_cross_encoder_onnx.py --verify`), which is run before the
default is changed and re-run per host.

Run: uv run pytest tests/test_onnx_backend.py -v
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from reportfinder.config import DEFAULT, from_mapping
from reportfinder.pipeline.onnx_scorer import OnnxPairScorer, build_scorer
from reportfinder.pipeline.rerank import PairScorer, TorchCrossEncoder

ONNX_MODEL = Path("artifacts/onnx/cross_encoder.onnx")

requires_export = pytest.mark.skipif(
    not ONNX_MODEL.exists(),
    reason=(
        "no exported graph; run "
        "`uv run python scripts/export_cross_encoder_onnx.py --verify`"
    ),
)


# --- configuration -----------------------------------------------------------


def test_backend_must_be_a_known_runtime():
    with pytest.raises(ValueError, match="rerank.backend"):
        from_mapping({"rerank": {"backend": "tensorrt"}})


def test_torch_remains_the_default_backend():
    """A host that has not run the parity gate must get the runtime that is known
    to be numerically correct there, even though it is slower."""
    assert DEFAULT.rerank.backend == "torch"


def test_build_scorer_returns_the_configured_backend(tmp_path):
    torch_cfg = from_mapping({"rerank": {"backend": "torch"}})
    assert isinstance(build_scorer(torch_cfg), TorchCrossEncoder)

    onnx_cfg = from_mapping({
        "rerank": {"backend": "onnx", "onnx_path": str(tmp_path / "m.onnx")},
    })
    assert isinstance(build_scorer(onnx_cfg), OnnxPairScorer)


def test_a_disabled_reranker_builds_nothing():
    assert build_scorer(from_mapping({"rerank": {"enabled": False}})) is None


def test_both_backends_satisfy_the_scorer_protocol(tmp_path):
    """The protocol is the whole substitution contract; if either drifts from it
    the swap stops being a drop-in."""
    assert isinstance(
        OnnxPairScorer(tmp_path / "m.onnx", "checkpoint", "revision"), PairScorer
    )
    assert isinstance(TorchCrossEncoder("checkpoint", "revision"), PairScorer)


def test_a_missing_graph_names_the_command_that_makes_it(tmp_path):
    """An error that does not say what to run is a error someone has to research."""
    scorer = OnnxPairScorer(tmp_path / "absent.onnx", "checkpoint", "revision")
    with pytest.raises(FileNotFoundError, match="export_cross_encoder_onnx"):
        scorer.score_pairs("query", ["document"])


def test_no_texts_needs_no_model(tmp_path):
    """An empty shortlist must not load 91 MB to score nothing."""
    scorer = OnnxPairScorer(tmp_path / "absent.onnx", "checkpoint", "revision")
    assert scorer.score_pairs("query", []).shape == (0,)


# --- against the real exported graph -----------------------------------------


@requires_export
@pytest.mark.integration
def test_onnx_scores_are_finite_where_torch_float32_is_not():
    """The measured reason this backend exists.

    torch float32 NaNs beyond ~24 tokens on this host, which is every real
    report. If ONNX ever starts doing the same, the speedup is worthless and this
    fails rather than silently ranking on NaN.
    """
    from reportfinder.config import CROSS_ENCODER_MODEL, CROSS_ENCODER_REVISION

    document = (
        "Title: Attrition and Replacement Lag Analysis | Category: Talent | "
        "Data source: All Workers | Fields: Termination Reason; Time to Fill; "
        "Headcount; Hire Date | Prompts: Effective Date"
    )
    scorer = OnnxPairScorer(
        ONNX_MODEL, CROSS_ENCODER_MODEL, CROSS_ENCODER_REVISION,
    )
    scores = scorer.score_pairs(
        "why are we losing people faster than we can backfill", [document] * 8
    )

    assert scores.shape == (8,)
    assert np.isfinite(scores).all()
    assert scorer.unstable_batches == 0
    # Identical inputs must score identically; a batching bug shows up here.
    assert np.allclose(scores, scores[0])


@requires_export
@pytest.mark.integration
def test_onnx_and_torch_rank_a_shortlist_identically():
    """Ordering, not absolute error, is the substitution criterion: a backend
    that reorders the shortlist is a different ranker wearing the same name."""
    from reportfinder.config import CROSS_ENCODER_MODEL, CROSS_ENCODER_REVISION

    documents = [
        "Title: Terminations by Manager | Fields: Termination Reason; Manager",
        "Title: Payroll Results by Pay Group | Fields: Net Pay; Gross Pay",
        "Title: Headcount by Organization | Fields: Headcount; Supervisory Organization",
        "Title: Learning Course Completion | Fields: Learning Assignment; Due Date",
    ]
    query = "why did people leave and who was their manager"

    onnx = OnnxPairScorer(
        ONNX_MODEL, CROSS_ENCODER_MODEL, CROSS_ENCODER_REVISION,
    ).score_pairs(query, documents)
    torch_scores = TorchCrossEncoder(
        CROSS_ENCODER_MODEL, CROSS_ENCODER_REVISION,
    ).score_pairs(query, documents)

    assert list(np.argsort(-onnx)) == list(np.argsort(-torch_scores))
    assert np.abs(onnx - torch_scores).max() < 1e-2
