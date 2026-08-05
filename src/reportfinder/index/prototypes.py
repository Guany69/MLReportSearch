"""Family query-prototypes: how people actually ask for a report.

Every other generator matches a query against text the *catalog* wrote. Report
titles are named by their authors, in system vocabulary; users ask in outcome
language. Prototypes close that gap by indexing likely *requests* per family, so
"why are we losing people" can match a family whose catalog text never says it.

The hard rule: prototype text is a retrieval aid, never evidence. It is indexed and
retrieved from, but it is never merged into report descriptions and never shown to
the cross-encoder as though the catalog said it. Generated language must not end up
justifying its own match.

Provenance is mandatory. `PrototypeSource` distinguishes a business question a
report owner wrote from a synthetic string a script produced, and
`validation_status` records whether a human ever confirmed it. On this estate only
deterministic seed prototypes exist, so nothing here should be presented as
production behaviour.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Sequence

import numpy as np

from .encoders import TextEncoder, l2_normalize

PROTOTYPE_SCHEMA_VERSION = "1"


class PrototypeSource(str, Enum):
    HUMAN_AUTHORED = "human_authored"
    REVIEWED_SEARCH = "reviewed_search"
    OWNER_BUSINESS_QUESTION = "owner_business_question"
    REVIEWED_SYNTHETIC = "reviewed_synthetic"
    GENERATED = "generated"
    # Derived deterministically from catalog text. Not user language, and not a
    # substitute for it -- a documented cold-start seed.
    CATALOG_SEED = "catalog_seed"


class ValidationStatus(str, Enum):
    VALIDATED = "validated"
    UNREVIEWED = "unreviewed"
    REJECTED = "rejected"


@dataclass(frozen=True)
class QueryPrototype:
    prototype_id: str
    family_id: str
    text: str
    source: PrototypeSource
    validation_status: ValidationStatus
    generator_revision: str
    catalog_version: str

    @property
    def is_authoritative_catalog_text(self) -> bool:
        """Always False. Prototypes never count as catalog evidence."""
        return False


def seed_prototypes_from_catalog(corpus, *, generator_revision: str = "seed-1") -> list[QueryPrototype]:
    """Deterministic cold-start prototypes, one per family.

    Built from the canonical title plus the family's most common field names, in
    request phrasing. This is a *seed*, not user language: it exists so the
    generator has something to retrieve from before any real request log or
    owner-authored question is collected, and its provenance says exactly that.
    """
    prototypes: list[QueryPrototype] = []
    for family_id, family in sorted(corpus.families.items()):
        instance = corpus.instance(family.instance_ids[0])
        fields = ", ".join(instance.fields[:4])
        text = f"show {family.canonical_title.lower()}"
        if fields:
            text = f"{text} with {fields.lower()}"
        prototypes.append(
            QueryPrototype(
                prototype_id=f"P-{family_id[:40]}",
                family_id=family_id,
                text=text,
                source=PrototypeSource.CATALOG_SEED,
                validation_status=ValidationStatus.UNREVIEWED,
                generator_revision=generator_revision,
                catalog_version=corpus.catalog_version,
            )
        )
    return prototypes


class FamilyPrototypeIndex:
    """Dense index over prototype text, returning families rather than instances."""

    def __init__(
        self,
        *,
        prototypes: tuple[QueryPrototype, ...],
        vectors: np.ndarray,
        model_id: str,
        revision: str,
    ) -> None:
        if vectors.shape[0] != len(prototypes):
            raise ValueError("prototype index vectors and prototypes disagree in length")
        self.prototypes = prototypes
        self.vectors = np.ascontiguousarray(vectors, dtype=np.float32)
        self.model_id = model_id
        self.revision = revision

    def __len__(self) -> int:
        return len(self.prototypes)

    def search_families(self, query_vector: np.ndarray, k: int) -> list[tuple[str, float]]:
        """Top-k families by their best-matching prototype.

        Best-matching rather than summed: a family with twenty prototypes must not
        outrank one with a single better match simply by having more text.
        """
        import torch

        if not len(self.prototypes):
            return []
        with torch.inference_mode():
            query = torch.from_numpy(
                np.ascontiguousarray(query_vector, dtype=np.float32).reshape(-1)
            )
            scores = (torch.from_numpy(self.vectors) @ query).numpy()

        best: dict[str, float] = {}
        for prototype, score in zip(self.prototypes, scores, strict=True):
            if prototype.validation_status is ValidationStatus.REJECTED:
                continue
            current = best.get(prototype.family_id)
            if current is None or score > current:
                best[prototype.family_id] = float(score)

        ranked = sorted(best.items(), key=lambda item: (-item[1], item[0]))
        return ranked[:k]

    @classmethod
    def build(
        cls,
        prototypes: Sequence[QueryPrototype],
        encoder: TextEncoder,
    ) -> FamilyPrototypeIndex:
        prototypes = tuple(prototypes)
        # Prototypes are queries, so they are encoded query-side -- indexing them
        # as documents would put them in the wrong space for an asymmetric model.
        vectors = (
            encoder.encode_queries([p.text for p in prototypes])
            if prototypes
            else np.zeros((0, getattr(encoder, "dim", 0)), dtype=np.float32)
        )
        return cls(
            prototypes=prototypes,
            vectors=l2_normalize(vectors),
            model_id=encoder.name,
            revision=encoder.revision,
        )

    def save(self, directory: Path) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        np.save(directory / "vectors.npy", self.vectors)
        (directory / "prototypes.json").write_text(json.dumps([
            {
                "prototype_id": p.prototype_id, "family_id": p.family_id,
                "text": p.text, "source": p.source.value,
                "validation_status": p.validation_status.value,
                "generator_revision": p.generator_revision,
                "catalog_version": p.catalog_version,
            }
            for p in self.prototypes
        ], indent=2))
        (directory / "meta.json").write_text(json.dumps({
            "schema_version": PROTOTYPE_SCHEMA_VERSION,
            "model_id": self.model_id,
            "revision": self.revision,
            "prototypes": len(self.prototypes),
            "families": len({p.family_id for p in self.prototypes}),
            "sources": sorted({p.source.value for p in self.prototypes}),
            "validation_statuses": sorted(
                {p.validation_status.value for p in self.prototypes}
            ),
        }, indent=2, sort_keys=True))

    @classmethod
    def load(cls, directory: Path) -> FamilyPrototypeIndex:
        meta = json.loads((directory / "meta.json").read_text())
        if meta.get("schema_version") != PROTOTYPE_SCHEMA_VERSION:
            raise ValueError(
                f"prototype index at {directory} has schema version "
                f"{meta.get('schema_version')!r}; expected {PROTOTYPE_SCHEMA_VERSION!r}"
            )
        raw = json.loads((directory / "prototypes.json").read_text())
        return cls(
            prototypes=tuple(
                QueryPrototype(
                    prototype_id=item["prototype_id"], family_id=item["family_id"],
                    text=item["text"], source=PrototypeSource(item["source"]),
                    validation_status=ValidationStatus(item["validation_status"]),
                    generator_revision=item["generator_revision"],
                    catalog_version=item["catalog_version"],
                )
                for item in raw
            ),
            vectors=np.load(directory / "vectors.npy"),
            model_id=meta["model_id"],
            revision=meta["revision"],
        )
