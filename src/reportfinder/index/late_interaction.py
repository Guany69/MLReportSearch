"""Token-level late interaction (ColBERT-style). Implemented, shipped disabled.

Single-vector retrieval compresses a whole report into one point, which is where
short and under-specified queries lose: there is no room left to match on the one
token that mattered. Late interaction keeps a vector per token and scores with
MaxSim -- for each query token, the best-matching document token, summed. It is the
right tool for exactly the high-recall-risk queries the risk policy flags, and it
is far too expensive to run on every query.

**Status: disabled.** The implementation, artifact loader, storage format and tests
are real, but no index has been built or measured on this estate, so there is no
evidence it helps here and no latency measurement to justify its cost. It ships
`enabled: false` with the reason recorded in config and surfaced in telemetry, and
`Config.__post_init__` refuses to enable it unless the component is also declared
required -- so it cannot be switched on without an index existing.

Nothing silently substitutes for it. When it is disabled the pipeline records that
it did not run; it does not quietly fall back to another dense search and report
that late interaction happened.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, Sequence, runtime_checkable

import numpy as np

LATE_INTERACTION_SCHEMA_VERSION = "1"


class LateInteractionUnavailable(RuntimeError):
    """Raised when a late-interaction search is attempted without an index."""


@runtime_checkable
class TokenEncoder(Protocol):
    """Encodes text to per-token vectors plus a validity mask."""

    name: str
    revision: str
    dim: int

    def encode_documents(self, texts: Sequence[str]) -> tuple[np.ndarray, np.ndarray]: ...

    def encode_queries(self, texts: Sequence[str]) -> tuple[np.ndarray, np.ndarray]: ...


@dataclass(frozen=True)
class LateInteractionStatus:
    """Why the generator is or is not running. Reported, never inferred."""

    enabled: bool
    built: bool
    reason: str

    def as_dict(self) -> dict[str, object]:
        return {"enabled": self.enabled, "built": self.built, "reason": self.reason}


def maxsim(
    query_vectors: np.ndarray,
    query_mask: np.ndarray,
    doc_vectors: np.ndarray,
    doc_mask: np.ndarray,
) -> np.ndarray:
    """MaxSim: per query token, the best document token; summed over query tokens.

    Shapes: query (q, d); documents (n, t, d). Returns (n,).
    Padding is excluded on both sides -- a padded document token that happened to
    align well would otherwise inflate the score.
    """
    import torch

    with torch.inference_mode():
        q = torch.from_numpy(np.ascontiguousarray(query_vectors, dtype=np.float32))
        d = torch.from_numpy(np.ascontiguousarray(doc_vectors, dtype=np.float32))
        qm = torch.from_numpy(np.asarray(query_mask, dtype=bool))
        dm = torch.from_numpy(np.asarray(doc_mask, dtype=bool))

        # (n, t, q) similarity between every doc token and every query token.
        similarity = torch.einsum("ntd,qd->ntq", d, q)
        similarity = similarity.masked_fill(~dm.unsqueeze(-1), float("-inf"))
        best_per_query_token = similarity.max(dim=1).values  # (n, q)
        best_per_query_token = best_per_query_token.masked_fill(~qm.unsqueeze(0), 0.0)
        # A document with no real tokens yields -inf; report it as no evidence.
        best_per_query_token = torch.nan_to_num(
            best_per_query_token, neginf=0.0, posinf=0.0
        )
        return best_per_query_token.sum(dim=1).numpy()


class LateInteractionIndex:
    """Per-token document vectors with MaxSim scoring."""

    def __init__(
        self,
        *,
        instance_ids: tuple[str, ...],
        doc_vectors: np.ndarray,
        doc_mask: np.ndarray,
        model_id: str,
        revision: str,
    ) -> None:
        if doc_vectors.shape[0] != len(instance_ids):
            raise ValueError("late-interaction vectors and instance ids disagree")
        self.instance_ids = instance_ids
        self.doc_vectors = np.ascontiguousarray(doc_vectors, dtype=np.float32)
        self.doc_mask = np.asarray(doc_mask, dtype=bool)
        self.model_id = model_id
        self.revision = revision

    def __len__(self) -> int:
        return len(self.instance_ids)

    def search(
        self,
        query_vectors: np.ndarray,
        query_mask: np.ndarray,
        k: int,
        *,
        universe=None,
    ) -> list[tuple[int, float]]:
        raw = maxsim(query_vectors, query_mask, self.doc_vectors, self.doc_mask)
        if universe is not None:
            raw = universe.restrict(raw)
        eligible = np.flatnonzero(np.isfinite(raw))
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
        encoder: TokenEncoder,
    ) -> LateInteractionIndex:
        vectors, mask = encoder.encode_documents(list(texts))
        return cls(
            instance_ids=tuple(instance_ids),
            doc_vectors=vectors,
            doc_mask=mask,
            model_id=encoder.name,
            revision=encoder.revision,
        )

    def save(self, directory: Path) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            directory / "tokens.npz", vectors=self.doc_vectors, mask=self.doc_mask
        )
        (directory / "ids.json").write_text(json.dumps(list(self.instance_ids)))
        (directory / "meta.json").write_text(json.dumps({
            "schema_version": LATE_INTERACTION_SCHEMA_VERSION,
            "model_id": self.model_id,
            "revision": self.revision,
            "rows": len(self),
            "tokens_per_doc": int(self.doc_vectors.shape[1]),
            "dim": int(self.doc_vectors.shape[2]),
        }, indent=2, sort_keys=True))

    @classmethod
    def load(cls, directory: Path) -> LateInteractionIndex:
        meta_path = directory / "meta.json"
        if not meta_path.exists():
            raise LateInteractionUnavailable(
                f"no late-interaction index at {directory}. Build it with "
                "'reportfinder-bundle build --component late_interaction', or leave "
                "retrieval.late_interaction.enabled=false."
            )
        meta = json.loads(meta_path.read_text())
        if meta.get("schema_version") != LATE_INTERACTION_SCHEMA_VERSION:
            raise ValueError(
                f"late-interaction index at {directory} has schema version "
                f"{meta.get('schema_version')!r}; "
                f"expected {LATE_INTERACTION_SCHEMA_VERSION!r}"
            )
        payload = np.load(directory / "tokens.npz")
        return cls(
            instance_ids=tuple(json.loads((directory / "ids.json").read_text())),
            doc_vectors=payload["vectors"],
            doc_mask=payload["mask"],
            model_id=meta["model_id"],
            revision=meta["revision"],
        )
