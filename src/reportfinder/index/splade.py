"""SPLADE learned-sparse retrieval.

SPLADE sits between BM25F and a bi-encoder: it scores in vocabulary space like a
lexical retriever, but the weights are learned, so a document about "termination
reason" carries non-zero weight on "attrition" and "leaver" without anyone writing
a synonym list. That is why it is an independent generator here rather than a
substitute for either neighbour -- it finds a different kind of match than both.

It is emphatically *not* a static synonym dictionary. The expansion is produced by
a transformer at index time and query time.

Storage is a CSR matrix. Measured on this estate: ~148 non-zero terms per document,
so 4000 documents is about 5 MB. A dense 4000x30522 float32 matrix would be 488 MB,
which is the reason the postings are sparse rather than a plain array.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, Sequence, runtime_checkable

import numpy as np
import scipy.sparse as sp

SPLADE_SCHEMA_VERSION = "1"


@runtime_checkable
class SparseEncoder(Protocol):
    """Encodes text to sparse term-weight vectors over a fixed vocabulary."""

    name: str
    revision: str

    # A read-only member: the production encoder reads this from the model config
    # on first load, so it is a property rather than a settable attribute.
    @property
    def vocab_size(self) -> int: ...

    def encode(self, texts: Sequence[str]) -> sp.csr_matrix: ...


@dataclass(frozen=True)
class SpladeBuildStats:
    rows: int
    nnz_total: int
    nnz_median: float
    vocab_size: int


class SpladeTorchEncoder:
    """The production encoder: masked-LM logits pooled into term weights.

    `log(1 + relu(logits))` masked by attention, max-pooled over positions -- the
    standard SPLADE formulation. Loaded lazily and pinned, with `trust_remote_code`
    off by default.
    """

    def __init__(
        self,
        checkpoint: str,
        revision: str,
        *,
        max_length: int = 256,
        batch_size: int = 32,
        min_term_weight: float = 0.01,
        trust_remote_code: bool = False,
    ) -> None:
        self.name = checkpoint
        self.revision = revision
        self.max_length = max_length
        self.batch_size = batch_size
        self.min_term_weight = min_term_weight
        self.trust_remote_code = trust_remote_code
        self._model = None
        self._tokenizer = None

    def _load(self):
        if self._model is None:
            from transformers import AutoModelForMaskedLM, AutoTokenizer

            self._tokenizer = AutoTokenizer.from_pretrained(
                self.name, revision=self.revision,
                trust_remote_code=self.trust_remote_code,
            )
            self._model = AutoModelForMaskedLM.from_pretrained(
                self.name, revision=self.revision,
                trust_remote_code=self.trust_remote_code,
            ).eval()
        return self._model, self._tokenizer

    @property
    def vocab_size(self) -> int:
        model, _ = self._load()
        return int(model.config.vocab_size)

    def encode(self, texts: Sequence[str]) -> sp.csr_matrix:
        import torch

        model, tokenizer = self._load()
        if not texts:
            return sp.csr_matrix((0, self.vocab_size), dtype=np.float32)

        blocks = []
        for start in range(0, len(texts), self.batch_size):
            batch = list(texts[start : start + self.batch_size])
            encoded = tokenizer(
                batch, return_tensors="pt", padding=True,
                truncation=True, max_length=self.max_length,
            )
            with torch.inference_mode():
                logits = model(**encoded).logits
                weights = torch.log1p(torch.relu(logits))
                mask = encoded["attention_mask"].unsqueeze(-1)
                pooled = torch.max(weights * mask, dim=1).values
                pooled[pooled < self.min_term_weight] = 0.0
            blocks.append(sp.csr_matrix(pooled.numpy().astype(np.float32)))
        return sp.vstack(blocks, format="csr")


class SpladeIndex:
    """Sparse postings over the corpus, scored by sparse dot product."""

    def __init__(
        self,
        *,
        instance_ids: tuple[str, ...],
        hashes: tuple[str, ...],
        postings: sp.csr_matrix,
        model_id: str,
        revision: str,
        stats: SpladeBuildStats | None = None,
    ) -> None:
        if postings.shape[0] != len(instance_ids):
            raise ValueError("splade postings and instance ids disagree in length")
        self.instance_ids = instance_ids
        self.hashes = hashes
        self.postings = postings.tocsr()
        self.model_id = model_id
        self.revision = revision
        self.stats = stats

    def __len__(self) -> int:
        return len(self.instance_ids)

    def scores(self, query_vector: sp.csr_matrix) -> np.ndarray:
        """Sparse dot product of the query's expanded terms against every document."""
        if query_vector.shape[1] != self.postings.shape[1]:
            raise ValueError(
                "splade query vocabulary does not match the index vocabulary; "
                "the index was built with a different checkpoint"
            )
        return np.asarray((self.postings @ query_vector.T).todense()).ravel()

    def search(
        self, query_vector: sp.csr_matrix, k: int, *, universe=None
    ) -> list[tuple[int, float]]:
        raw = self.scores(query_vector)
        if universe is not None:
            raw = universe.restrict(raw)
        # A zero score means no shared expanded term -- genuinely no evidence, so
        # it is excluded rather than ranked arbitrarily.
        eligible = np.flatnonzero(np.isfinite(raw) & (raw > 0))
        if eligible.size == 0:
            return []
        order = eligible[np.argsort(-raw[eligible], kind="stable")][:k]
        return [(int(p), float(raw[p])) for p in order]

    @classmethod
    def build(
        cls,
        *,
        instance_ids: Sequence[str],
        texts: Sequence[str],
        hashes: Sequence[str],
        encoder: SparseEncoder,
    ) -> SpladeIndex:
        postings = encoder.encode(list(texts))
        nnz_per_row = np.diff(postings.indptr) if postings.shape[0] else np.array([0])
        return cls(
            instance_ids=tuple(instance_ids),
            hashes=tuple(hashes),
            postings=postings,
            model_id=encoder.name,
            revision=encoder.revision,
            stats=SpladeBuildStats(
                rows=postings.shape[0],
                nnz_total=int(postings.nnz),
                nnz_median=float(np.median(nnz_per_row)),
                vocab_size=int(postings.shape[1]),
            ),
        )

    def save(self, directory: Path) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        sp.save_npz(directory / "postings.npz", self.postings)
        (directory / "ids.json").write_text(json.dumps(list(self.instance_ids)))
        (directory / "hashes.json").write_text(json.dumps(list(self.hashes)))
        (directory / "meta.json").write_text(json.dumps({
            "schema_version": SPLADE_SCHEMA_VERSION,
            "model_id": self.model_id,
            "revision": self.revision,
            "rows": len(self),
            "vocab_size": int(self.postings.shape[1]),
            "stats": self.stats.__dict__ if self.stats else None,
        }, indent=2, sort_keys=True))

    @classmethod
    def load(cls, directory: Path) -> SpladeIndex:
        meta = json.loads((directory / "meta.json").read_text())
        if meta.get("schema_version") != SPLADE_SCHEMA_VERSION:
            raise ValueError(
                f"splade index at {directory} has schema version "
                f"{meta.get('schema_version')!r}; expected {SPLADE_SCHEMA_VERSION!r}"
            )
        stats = meta.get("stats")
        return cls(
            instance_ids=tuple(json.loads((directory / "ids.json").read_text())),
            hashes=tuple(json.loads((directory / "hashes.json").read_text())),
            postings=sp.load_npz(directory / "postings.npz"),
            model_id=meta["model_id"],
            revision=meta["revision"],
            stats=SpladeBuildStats(**stats) if stats else None,
        )
