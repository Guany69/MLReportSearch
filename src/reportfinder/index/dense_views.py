"""One dense index per view, with content-hash incremental re-embedding.

At 4000 instances an exact search is a single matmul plus `torch.topk` -- roughly a
millisecond. An approximate index would add a dependency, a build step and a recall
loss to solve a problem this estate does not have, so retrieval here is exact and
`test_index_dense_views.py` asserts parity with brute force.

Incremental re-embedding is the reason the corpus is split into four views at all:
each view stores the content hash of every row it embedded, so a title correction
re-encodes 1 row of the identity view instead of 4000 rows of everything.

A view whose text is near-constant is worse than useless -- it returns the same
arbitrary top-k for every query and consumes shortlist slots that would otherwise
hold real candidates. `distinct_text_ratio` is measured at build time so the bundle
can record a degenerate status instead of silently serving noise.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np

from ..corpus import ViewType
from .encoders import TextEncoder, l2_normalize

INDEX_SCHEMA_VERSION = "1"


@dataclass(frozen=True)
class BuildStats:
    """What the build actually did, for the manifest."""

    rows: int
    reencoded_rows: int
    reused_rows: int
    empty_rows: int
    distinct_text_ratio: float


class DenseViewIndex:
    """Exact dense retrieval over one view."""

    def __init__(
        self,
        *,
        view_type: ViewType,
        instance_ids: tuple[str, ...],
        hashes: tuple[str, ...],
        vectors: np.ndarray,
        model_id: str,
        revision: str,
        stats: BuildStats | None = None,
    ) -> None:
        if vectors.shape[0] != len(instance_ids):
            raise ValueError("dense index vectors and instance ids disagree in length")
        self.view_type = view_type
        self.instance_ids = instance_ids
        self.hashes = hashes
        self.vectors = np.ascontiguousarray(vectors, dtype=np.float32)
        self.model_id = model_id
        self.revision = revision
        self.stats = stats
        self._torch_vectors = None

    def __len__(self) -> int:
        return len(self.instance_ids)

    @property
    def dim(self) -> int:
        return int(self.vectors.shape[1]) if self.vectors.size else 0

    def _as_torch(self):
        import torch

        if self._torch_vectors is None:
            self._torch_vectors = torch.from_numpy(self.vectors)
        return self._torch_vectors

    def scores(self, query_vector: np.ndarray) -> np.ndarray:
        """Cosine of the query against every row. Both sides are L2-normalized."""
        import torch

        with torch.inference_mode():
            query = torch.from_numpy(
                np.ascontiguousarray(query_vector, dtype=np.float32).reshape(-1)
            )
            return (self._as_torch() @ query).numpy()

    def search(
        self,
        query_vector: np.ndarray,
        k: int,
        *,
        universe=None,
    ) -> list[tuple[int, float]]:
        """Top-k authorized positions, best first.

        The universe is applied to the score vector *before* top-k, not after, so
        an unauthorized row cannot consume a slot -- k means k authorized results.

        The sort is stable, so ties resolve to row order rather than to whatever
        order a top-k kernel happens to emit. Exact ties are ordinary here: an
        empty view row is stored as a zero vector and scores 0.0 against every
        query, so a near-degenerate view can tie dozens of rows at once.
        """
        raw = self.scores(query_vector)
        if universe is not None:
            raw = universe.restrict(raw)

        finite = np.isfinite(raw)
        if not finite.any():
            return []
        limit = min(k, int(finite.sum()))
        # -inf (unauthorized) sorts to the end of a descending order and NaN sorts
        # after everything, so the first `limit` positions are the finite ones; the
        # isfinite filter below stays as a belt.
        order = np.argsort(-raw, kind="stable")[:limit]
        return [
            (int(position), float(raw[position]))
            for position in order
            if np.isfinite(raw[position])
        ]

    # -- build / persist ---------------------------------------------------

    @classmethod
    def build(
        cls,
        *,
        view_type: ViewType,
        instance_ids: Sequence[str],
        texts: Sequence[str],
        hashes: Sequence[str],
        encoder: TextEncoder,
        previous: DenseViewIndex | None = None,
    ) -> DenseViewIndex:
        """Encode this view, reusing unchanged rows from `previous` when possible."""
        instance_ids = tuple(instance_ids)
        hashes = tuple(hashes)
        texts = list(texts)

        reusable: dict[str, int] = {}
        if (
            previous is not None
            and previous.model_id == encoder.name
            and previous.revision == encoder.revision
            and previous.view_type is view_type
        ):
            # Keyed by (instance, content hash): the same instance with different
            # text must not reuse a stale vector.
            reusable = {
                f"{i}\x1f{h}": position
                for position, (i, h) in enumerate(
                    zip(previous.instance_ids, previous.hashes, strict=True)
                )
            }

        dim = previous.dim if previous is not None and previous.dim else None
        to_encode: list[int] = []
        for position, (instance_id, digest) in enumerate(zip(instance_ids, hashes, strict=True)):
            if f"{instance_id}\x1f{digest}" not in reusable:
                to_encode.append(position)

        encoded = (
            encoder.encode_documents([texts[p] for p in to_encode])
            if to_encode
            else np.zeros((0, dim or 0), dtype=np.float32)
        )
        if encoded.size:
            dim = int(encoded.shape[1])
        if dim is None:
            dim = getattr(encoder, "dim", 0)

        vectors = np.zeros((len(instance_ids), dim), dtype=np.float32)
        for offset, position in enumerate(to_encode):
            vectors[position] = encoded[offset]
        for position, (instance_id, digest) in enumerate(zip(instance_ids, hashes, strict=True)):
            source = reusable.get(f"{instance_id}\x1f{digest}")
            if source is not None:
                vectors[position] = previous.vectors[source]  # type: ignore[union-attr]

        # An empty view has no direction and must score 0 against everything rather
        # than sitting at some arbitrary point in the space.
        empty = [position for position, text in enumerate(texts) if not text.strip()]
        for position in empty:
            vectors[position] = 0.0

        non_empty = [t for t in texts if t.strip()]
        distinct_ratio = (
            len(set(non_empty)) / len(texts) if texts else 0.0
        )

        return cls(
            view_type=view_type,
            instance_ids=instance_ids,
            hashes=hashes,
            vectors=l2_normalize(vectors),
            model_id=encoder.name,
            revision=encoder.revision,
            stats=BuildStats(
                rows=len(instance_ids),
                reencoded_rows=len(to_encode),
                reused_rows=len(instance_ids) - len(to_encode),
                empty_rows=len(empty),
                distinct_text_ratio=round(distinct_ratio, 4),
            ),
        )

    def save(self, directory: Path) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        np.save(directory / "vectors.npy", self.vectors)
        (directory / "ids.json").write_text(json.dumps(list(self.instance_ids)))
        (directory / "hashes.json").write_text(json.dumps(list(self.hashes)))
        (directory / "meta.json").write_text(json.dumps({
            "schema_version": INDEX_SCHEMA_VERSION,
            "view_type": self.view_type.value,
            "model_id": self.model_id,
            "revision": self.revision,
            "dim": self.dim,
            "rows": len(self),
            "stats": self.stats.__dict__ if self.stats else None,
        }, indent=2, sort_keys=True))

    @classmethod
    def load(cls, directory: Path) -> DenseViewIndex:
        meta = json.loads((directory / "meta.json").read_text())
        if meta.get("schema_version") != INDEX_SCHEMA_VERSION:
            raise ValueError(
                f"dense view index at {directory} has schema version "
                f"{meta.get('schema_version')!r}; expected {INDEX_SCHEMA_VERSION!r}"
            )
        stats = meta.get("stats")
        return cls(
            view_type=ViewType(meta["view_type"]),
            instance_ids=tuple(json.loads((directory / "ids.json").read_text())),
            hashes=tuple(json.loads((directory / "hashes.json").read_text())),
            vectors=np.load(directory / "vectors.npy"),
            model_id=meta["model_id"],
            revision=meta["revision"],
            stats=BuildStats(**stats) if stats else None,
        )
