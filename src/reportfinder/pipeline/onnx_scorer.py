"""An ONNX cross-encoder backend, behind the existing `PairScorer` protocol.

Why this exists is a measurement, not a preference. On this host torch 2.11 on
arm64 macOS 13 returns NaN in float32 for any cross-encoder input beyond ~24
tokens -- which is every real report -- so `TorchCrossEncoder` probes at load and
escalates to float64. float64 is exact and roughly 2.3x slower, and reranking is
the dominant term in per-query latency.

The NaN is a torch/mkldnn kernel problem, not a property of the weights: the same
graph executed by a different runtime has no reason to reproduce it. ONNX Runtime
is that different runtime. Whether it actually helps is decided by the parity
gate, not by this docstring -- `scripts/export_cross_encoder_onnx.py --verify`
compares logits and induced ordering against float64 torch on real report text,
and the torch backend stays the default until that passes.

The protocol is exactly `name`, `revision`, `score_pairs`, so this is a drop-in
at one construction site in `factory.py`.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Sequence

import numpy as np

log = logging.getLogger(__name__)


class OnnxPairScorer:
    """Cross-encoder inference through ONNX Runtime.

    Deliberately mirrors `TorchCrossEncoder`'s public surface -- including the
    non-finite guard. A backend that silently emits NaN would look faster while
    scoring nothing, which is the exact failure the torch path already had to
    grow a probe for.
    """

    def __init__(
        self,
        model_path: Path,
        checkpoint: str,
        revision: str,
        *,
        max_length: int = 256,
        batch_size: int = 32,
        trust_remote_code: bool = False,
        threads: int | None = None,
    ) -> None:
        self.model_path = Path(model_path)
        self.name = checkpoint
        self.revision = revision
        self.max_length = max_length
        self.batch_size = batch_size
        self.trust_remote_code = trust_remote_code
        self.threads = threads
        self.dtype = "float32"
        # Kept so telemetry can report the same fields for either backend.
        self.dtype_escalated = False
        self.unstable_batches = 0
        self.backend = "onnx"
        self._session = None
        self._tokenizer = None
        self._inputs: tuple[str, ...] = ()

    def _load(self):
        if self._session is not None:
            return self._session, self._tokenizer

        import onnxruntime as ort
        from transformers import AutoTokenizer

        if not self.model_path.exists():
            raise FileNotFoundError(
                f"ONNX cross-encoder not found at {self.model_path}. Export it with "
                "`uv run python scripts/export_cross_encoder_onnx.py`, or set "
                "rerank.backend=torch."
            )

        options = ort.SessionOptions()
        if self.threads:
            options.intra_op_num_threads = self.threads
        self._session = ort.InferenceSession(
            str(self.model_path), options, providers=["CPUExecutionProvider"],
        )
        self._inputs = tuple(i.name for i in self._session.get_inputs())
        self._tokenizer = AutoTokenizer.from_pretrained(
            self.name, revision=self.revision,
            trust_remote_code=self.trust_remote_code,
        )
        return self._session, self._tokenizer

    def score_pairs(self, query: str, texts: Sequence[str]) -> np.ndarray:
        if not texts:
            return np.zeros(0, dtype=np.float32)
        session, tokenizer = self._load()

        scores: list[np.ndarray] = []
        for start in range(0, len(texts), self.batch_size):
            chunk = list(texts[start : start + self.batch_size])
            encoded = tokenizer(
                [(query, text) for text in chunk],
                return_tensors="np", padding=True,
                truncation=True, max_length=self.max_length,
            )
            feed = {
                name: encoded[name].astype(np.int64)
                for name in self._inputs if name in encoded
            }
            logits = session.run(None, feed)[0].reshape(-1)
            if not np.isfinite(logits).all():
                # Same failure the torch path has, counted the same way rather
                # than assumed impossible.
                self.unstable_batches += 1
                log.warning(
                    "onnx_cross_encoder_non_finite",
                    extra={"checkpoint": self.name, "batch": start},
                )
            scores.append(logits.astype(np.float32))
        return np.concatenate(scores).astype(np.float32)


def build_scorer(cfg):
    """The configured cross-encoder backend.

    One place decides, so a config that names a backend cannot silently get the
    other one.
    """
    from .rerank import TorchCrossEncoder

    rerank = cfg.rerank
    if not rerank.enabled:
        return None
    if rerank.backend == "onnx":
        return OnnxPairScorer(
            rerank.onnx_path or Path("artifacts/onnx/cross_encoder.onnx"),
            rerank.checkpoint, rerank.revision,
            max_length=rerank.max_length, batch_size=rerank.batch_size,
            trust_remote_code=cfg.security.trust_remote_code,
        )
    return TorchCrossEncoder(
        rerank.checkpoint, rerank.revision,
        max_length=rerank.max_length, batch_size=rerank.batch_size,
        trust_remote_code=cfg.security.trust_remote_code,
        dtype=rerank.dtype,
    )
