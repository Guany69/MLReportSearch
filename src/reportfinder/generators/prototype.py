"""Family query-prototype nomination.

Retrieves *families* by how people ask for them, then maps each family to its
authorized instances before the union. That mapping is where authorization has to be
re-applied: a family may be visible while some of its instances are not, and the
generator must not leak the unauthorized ones back in through the family door.

Prototype text never becomes evidence. The hit records that a prototype matched and
which one, but the cross-encoder is shown catalog text only.
"""

from __future__ import annotations

from .base import GeneratorResult, merge_variant_runs

PROTOTYPE_VARIANTS = ("Q0", "Q3", "Q5", "QC")


class PrototypeGenerator:
    """Independent nomination via user-language prototypes."""

    name = "prototype"
    view_type = None

    def __init__(self, index, encoder, corpus, *, variants=PROTOTYPE_VARIANTS) -> None:
        self.index = index
        self.encoder = encoder
        self.corpus = corpus
        self.variants = variants

    def generate(self, plan, universe, k: int) -> GeneratorResult:
        selected = plan.variants_for(self.variants)
        vectors = self.encoder.encode_queries([v.text for v in selected])

        runs = []
        evidence: dict[str, dict] = {}
        for variant, vector in zip(selected, vectors, strict=True):
            families = self.index.search_families(vector, k)
            positions: list[tuple[int, float]] = []
            for family_id, score in families:
                family = self.corpus.families.get(family_id)
                if family is None:
                    continue
                for instance_id in family.instance_ids:
                    position = self.corpus.position_of(instance_id)
                    # Family-level retrieval must not reintroduce an instance the
                    # resolver excluded.
                    if universe is not None and not universe.allows_position(position):
                        continue
                    positions.append((position, float(score)))
                    evidence.setdefault(instance_id, {
                        "matched_family": family_id,
                        "prototype_similarity": round(float(score), 4),
                        # Provenance travels with the hit so nothing downstream can
                        # mistake generated language for catalog text.
                        "prototype_is_authoritative_catalog_text": False,
                        "matched_via_variant": variant.key,
                    })
            runs.append((variant.key, positions[:k]))

        return merge_variant_runs(
            self.name, self.view_type, runs, self.corpus.instance_ids,
            match_evidence=evidence,
        )
