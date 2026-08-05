"""Cross-encoder reranking over the whole shortlist.

The cross-encoder is the first stage that reads the query and the report *together*,
so it is the only stage that can tell "termination reason" from "termination date"
in context. That is exactly why it must see the full shortlist rather than a
pre-truncated head: a candidate cut before this point was judged by retrievers that
never read the query and the report jointly.

Two invariants:

* it is always handed the **raw** query (Q0), never a normalized or expanded form.
  The rewrites exist to widen retrieval; the model that makes the fine-grained
  judgement should see what the user actually wrote.
* it is shown **catalog text only**. Generated query prototypes are retrieval aids;
  letting them into the reranked text would let generated language justify its own
  match.

Measured on this hardware (CPU, 4 threads, MiniLM-L6): 120 pairs in ~370 ms, 200
pairs in ~620 ms. The old path capped this at 20 to stay fast; the measurement is
what justifies raising it.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Protocol, Sequence, runtime_checkable

import numpy as np

from ..corpus import authoritative_text

log = logging.getLogger(__name__)


@runtime_checkable
class PairScorer(Protocol):
    """Scores (query, document) pairs jointly."""

    name: str
    revision: str

    def score_pairs(self, query: str, texts: Sequence[str]) -> np.ndarray: ...


@dataclass
class RerankResult:
    """Cross-encoder scores plus the rank they imply."""

    scores: dict[str, float]
    ranks: dict[str, int]
    scored_count: int
    model_id: str
    model_revision: str
    skipped_reason: str | None = None
    dtype: str = "float32"
    dtype_escalated: bool = False
    non_finite_scores: int = 0

    @property
    def ran(self) -> bool:
        return self.skipped_reason is None

    def telemetry(self) -> dict[str, object]:
        return {
            "cross_encoder_ran": self.ran,
            "cross_encoder_scored": self.scored_count,
            "cross_encoder_model": self.model_id,
            "cross_encoder_revision": self.model_revision,
            "cross_encoder_skipped_reason": self.skipped_reason,
            "cross_encoder_dtype": self.dtype,
            # True when float32 was found untrustworthy on this machine and the
            # scorer escalated to float64 to stay correct.
            "cross_encoder_dtype_escalated": self.dtype_escalated,
            "cross_encoder_non_finite_scores": self.non_finite_scores,
        }


# A pair long enough to exercise the attention kernels that misbehave. Used once
# at load to decide whether float32 is trustworthy on this machine.
_PROBE_PAIR = (
    "why are we losing people faster than we can backfill",
    "Title: Attrition and Replacement Lag Analysis | Category: Talent | "
    "Data source: All Workers | Fields: Termination Reason; Time to Fill; Headcount",
)


class TorchCrossEncoder:
    """The production scorer, on transformers directly.

    Built on `AutoModelForSequenceClassification` rather than
    `sentence_transformers.CrossEncoder` so truncation, batching and dtype are
    explicit and inspectable -- all three turned out to matter here.

    **Numerical-stability probe.** On this development machine (torch 2.11.0,
    arm64 macOS 13, mkldnn unavailable) this checkpoint returns NaN in float32 for
    any input beyond roughly 24 tokens -- which is every real report -- while
    float64 is exact. Silently returning NaN is the worst possible failure: the
    reranker appears to run, scores nothing, and every downstream margin becomes
    meaningless. So the scorer probes once at load and escalates to float64 if
    float32 is not trustworthy, recording that it did. On a healthy machine it
    stays in float32 and pays nothing.
    """

    def __init__(
        self,
        checkpoint: str,
        revision: str,
        *,
        max_length: int = 256,
        batch_size: int = 32,
        trust_remote_code: bool = False,
        dtype: str = "auto",
    ) -> None:
        self.name = checkpoint
        self.revision = revision
        self.max_length = max_length
        self.batch_size = batch_size
        self.trust_remote_code = trust_remote_code
        self.dtype_policy = dtype
        self.dtype_escalated = False
        self.unstable_batches = 0
        self._model = None
        self._tokenizer = None

    def _load(self):
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        if self._model is not None:
            return self._model, self._tokenizer

        self._tokenizer = AutoTokenizer.from_pretrained(
            self.name, revision=self.revision,
            trust_remote_code=self.trust_remote_code,
        )
        model = AutoModelForSequenceClassification.from_pretrained(
            self.name, revision=self.revision,
            trust_remote_code=self.trust_remote_code,
        ).eval()

        if self.dtype_policy == "float64":
            model = model.double()
            self.dtype_escalated = True
        elif self.dtype_policy == "auto" and not self._probe_is_finite(
            model, self._tokenizer
        ):
            log.warning(
                "cross_encoder_float32_unstable",
                extra={"checkpoint": self.name, "torch": torch.__version__},
            )
            model = model.double()
            self.dtype_escalated = True

        self._model = model
        return self._model, self._tokenizer

    @staticmethod
    def _probe_is_finite(model, tokenizer) -> bool:
        import torch

        batch = tokenizer(
            [_PROBE_PAIR], return_tensors="pt", padding=True,
            truncation=True, max_length=256,
        )
        with torch.inference_mode():
            logits = model(**batch).logits
        return bool(torch.isfinite(logits).all())

    @property
    def dtype(self) -> str:
        self._load()
        return "float64" if self.dtype_escalated else "float32"

    def score_pairs(self, query: str, texts: Sequence[str]) -> np.ndarray:
        import torch

        if not texts:
            return np.zeros(0, dtype=np.float32)
        model, tokenizer = self._load()

        scores: list[np.ndarray] = []
        for start in range(0, len(texts), self.batch_size):
            chunk = list(texts[start : start + self.batch_size])
            batch = tokenizer(
                [(query, text) for text in chunk],
                return_tensors="pt", padding=True,
                truncation=True, max_length=self.max_length,
            )
            with torch.inference_mode():
                logits = model(**batch).logits.reshape(-1)
                if not torch.isfinite(logits).all() and not self.dtype_escalated:
                    # Belt and braces: the probe passed but this batch did not.
                    # Recompute it exactly rather than emitting NaN downstream.
                    self.unstable_batches += 1
                    logits = (
                        model.double()(**batch).logits.reshape(-1)
                    )
                    self.dtype_escalated = True
            scores.append(logits.float().numpy())
        return np.concatenate(scores).astype(np.float32)


def rerank_shortlist(
    shortlist,
    corpus,
    scorer: PairScorer | None,
    *,
    raw_query: str,
    max_report_chars: int = 1200,
) -> RerankResult:
    """Score every shortlisted candidate against the raw query.

    Note there is no depth parameter. The shortlist *is* the depth -- the risk
    policy already decided how wide to go, and silently rescoring only its head
    would reintroduce the truncation this architecture removes.
    """
    if scorer is None:
        return RerankResult(
            scores={}, ranks={}, scored_count=0,
            model_id="", model_revision="",
            skipped_reason="no cross-encoder configured",
        )
    if not len(shortlist):
        return RerankResult(
            scores={}, ranks={}, scored_count=0,
            model_id=getattr(scorer, "name", ""),
            model_revision=getattr(scorer, "revision", ""),
            skipped_reason="empty shortlist",
        )

    instance_ids = shortlist.instance_ids
    texts = [
        authoritative_text(corpus.instance(instance_id), max_chars=max_report_chars)
        for instance_id in instance_ids
    ]
    raw_scores = scorer.score_pairs(raw_query, texts)

    # A non-finite score must never reach the ranker: it would silently poison
    # every margin and comparison downstream. Any that survive are dropped and
    # counted, so the candidate falls back to its retrieval ordering rather than
    # to a meaningless number.
    finite = np.isfinite(raw_scores)
    scores = {
        instance_id: float(score)
        for instance_id, score, ok in zip(instance_ids, raw_scores, finite, strict=True)
        if ok
    }
    ordered = sorted(scores.items(), key=lambda item: (-item[1], item[0]))
    ranks = {instance_id: rank for rank, (instance_id, _) in enumerate(ordered, start=1)}

    return RerankResult(
        scores=scores,
        ranks=ranks,
        scored_count=len(scores),
        model_id=getattr(scorer, "name", ""),
        model_revision=getattr(scorer, "revision", ""),
        dtype=getattr(scorer, "dtype", "float32"),
        dtype_escalated=bool(getattr(scorer, "dtype_escalated", False)),
        non_finite_scores=int((~finite).sum()),
    )
