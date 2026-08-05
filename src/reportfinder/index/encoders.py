"""Text encoders behind a narrow protocol.

Retrieval code depends on `TextEncoder`, never on sentence-transformers directly.
That keeps unit tests free of model downloads (they inject a deterministic fake)
while production uses the real checkpoint, and it makes the query/document
asymmetry explicit rather than a convention people have to remember.

The asymmetry matters: bge-* models are trained with an instruction prefix on the
query side only. Applying it to documents, or omitting it from queries, silently
degrades retrieval without failing anything.
"""

from __future__ import annotations

import threading
from collections import OrderedDict
from typing import Protocol, Sequence, runtime_checkable

import numpy as np

# Applied query-side only, matching how the checkpoint was trained.
QUERY_PREFIXES = {
    "BAAI/bge-small-en-v1.5": "Represent this sentence for searching relevant passages: ",
    "BAAI/bge-base-en-v1.5": "Represent this sentence for searching relevant passages: ",
    "intfloat/e5-small-v2": "query: ",
    "intfloat/e5-base-v2": "query: ",
}
DOCUMENT_PREFIXES = {
    # e5 is symmetric-prefixed: documents carry "passage: ". bge documents carry none.
    "intfloat/e5-small-v2": "passage: ",
    "intfloat/e5-base-v2": "passage: ",
}


@runtime_checkable
class TextEncoder(Protocol):
    """Encodes text to L2-normalized vectors."""

    name: str
    revision: str

    @property
    def dim(self) -> int: ...

    def encode_documents(self, texts: Sequence[str]) -> np.ndarray: ...

    def encode_queries(self, texts: Sequence[str]) -> np.ndarray: ...


def l2_normalize(matrix: np.ndarray) -> np.ndarray:
    """Row-normalize so a dot product is a cosine.

    Zero rows stay zero rather than becoming NaN: an empty view legitimately has no
    direction, and it must score 0 against every query rather than poisoning the
    matmul.
    """
    matrix = np.asarray(matrix, dtype=np.float32)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    return matrix / np.where(norms > 0, norms, 1.0)


class SentenceTransformerEncoder:
    """The production encoder. Loads lazily and pins the checkpoint revision.

    **Query encodings are cached.** Three dense view generators each search several
    query representations, and they overlap heavily -- Q0 is searched by all of
    them, Q2 by two. Without the cache one request encodes the same string three
    times. Documents are deliberately *not* cached: they are encoded once at build
    time and caching 4000 of them would just hold the corpus twice in memory.
    """

    def __init__(
        self,
        checkpoint: str,
        revision: str,
        *,
        max_length: int = 512,
        batch_size: int = 64,
        trust_remote_code: bool = False,
        query_cache_size: int = 256,
    ) -> None:
        self.name = checkpoint
        self.revision = revision
        self.max_length = max_length
        self.batch_size = batch_size
        self.trust_remote_code = trust_remote_code
        self._model = None
        self._dim: int | None = None
        self._query_cache: OrderedDict[str, np.ndarray] = OrderedDict()
        self._query_cache_size = query_cache_size
        # Generators run concurrently, so the cache is touched from several threads.
        self._cache_lock = threading.Lock()
        self.query_cache_hits = 0
        self.query_cache_misses = 0

    @property
    def model(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(
                self.name,
                revision=self.revision,
                trust_remote_code=self.trust_remote_code,
            )
            self._model.max_seq_length = self.max_length
        return self._model

    @property
    def dim(self) -> int:
        if self._dim is None:
            self._dim = int(self.model.get_sentence_embedding_dimension())
        return self._dim

    def _encode(self, texts: Sequence[str], prefix: str) -> np.ndarray:
        import torch

        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        prepared = [f"{prefix}{text}" for text in texts]
        with torch.inference_mode():
            vectors = self.model.encode(
                prepared,
                batch_size=self.batch_size,
                convert_to_numpy=True,
                normalize_embeddings=True,
                show_progress_bar=False,
            )
        return l2_normalize(np.asarray(vectors, dtype=np.float32))

    def encode_documents(self, texts: Sequence[str]) -> np.ndarray:
        return self._encode(texts, DOCUMENT_PREFIXES.get(self.name, ""))

    def encode_queries(self, texts: Sequence[str]) -> np.ndarray:
        """Encode, reusing any representation already seen.

        Only the texts not already cached are sent to the model, then the full
        result is reassembled in the caller's order -- so this is an optimization
        with no effect on what is returned.
        """
        texts = list(texts)
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)

        with self._cache_lock:
            cached = {t: self._query_cache[t] for t in texts if t in self._query_cache}
            for text in cached:
                self._query_cache.move_to_end(text)
        missing = [t for t in dict.fromkeys(texts) if t not in cached]
        self.query_cache_hits += len(texts) - sum(
            1 for t in texts if t in missing
        )
        self.query_cache_misses += len(missing)

        if missing:
            encoded = self._encode(missing, QUERY_PREFIXES.get(self.name, ""))
            with self._cache_lock:
                for text, vector in zip(missing, encoded, strict=True):
                    self._query_cache[text] = vector
                    self._query_cache.move_to_end(text)
                while len(self._query_cache) > self._query_cache_size:
                    self._query_cache.popitem(last=False)
                cached.update({t: self._query_cache[t] for t in missing})

        return np.vstack([cached[t] for t in texts]).astype(np.float32)
