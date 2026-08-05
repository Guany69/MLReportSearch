"""BM25F as an independent generator.

BM25F used to be one of several channels whose ranks were fused and then cut to a
single shared depth. Here it is one nominator among many: it retrieves its own
`bm25_k` candidates, keeps its own scores, and cannot prevent any other generator
from nominating a report it missed. Nothing is required to appear in the BM25F
slate to reach the union.

It remains the strongest signal for exact-name lookups, which is why the migration
keeps it rather than replacing it -- `test_vague_query_recall.py` asserts both
directions.
"""

from __future__ import annotations

import numpy as np

from ..retrieval.bm25f import BM25FIndex, tokenize
from .base import GeneratorResult, merge_variant_runs

# BM25F reads catalog text, so alias expansion helps it; the raw query is always
# searched too and always ranks first on ties.
BM25F_VARIANTS = ("Q0", "Q1", "Q2", "Q5", "QC")

# Zone weights come from Config; the names must match frame columns.
ZONE_FIELDS = (
    ("title", "bm25_title"),
    ("description", "bm25_description"),
    ("fields", "bm25_fields"),
    ("prompts", "bm25_prompts"),
    ("category", "bm25_category"),
    ("tags", "bm25_tags"),
    ("data_source", "bm25_data_source"),
    ("area_where_used", "bm25_area_where_used"),
)


def zone_weights(cfg) -> dict[str, float]:
    return {zone: getattr(cfg, attr) for zone, attr in ZONE_FIELDS}


class Bm25fGenerator:
    """Field-aware lexical nomination."""

    name = "bm25f"
    view_type = None

    def __init__(self, index: BM25FIndex, corpus, *, variants=BM25F_VARIANTS) -> None:
        self.index = index
        self.corpus = corpus
        self.variants = variants

    @classmethod
    def from_frame(cls, frame, cfg, corpus) -> Bm25fGenerator:
        return cls(BM25FIndex(frame, zone_weights(cfg)), corpus)

    def generate(self, plan, universe, k: int) -> GeneratorResult:
        runs: list[tuple[str, list]] = []
        evidence: dict[str, dict] = {}
        for variant in plan.variants_for(self.variants):
            scores = self.index.scores(variant.text)
            if universe is not None:
                scores = universe.restrict(scores)
            # A zero BM25F score means no shared term: no lexical evidence at all,
            # so it is excluded rather than ranked arbitrarily. Other generators
            # remain free to nominate the same report.
            eligible = np.flatnonzero(np.isfinite(scores) & (scores > 0))
            if eligible.size == 0:
                runs.append((variant.key, []))
                continue
            order = eligible[np.argsort(-scores[eligible], kind="stable")][:k]
            runs.append((variant.key, [(int(p), float(scores[p])) for p in order]))

            query_terms = set(tokenize(variant.text))
            for position in order:
                instance_id = self.corpus.instance_ids[position]
                if instance_id in evidence:
                    continue
                instance = self.corpus.instances[position]
                evidence[instance_id] = {
                    "matched_title_terms": sorted(
                        query_terms & set(tokenize(instance.title))
                    ),
                    "matched_field_terms": sorted(
                        query_terms & set(tokenize(" ".join(instance.fields)))
                    ),
                    "matched_prompt_terms": sorted(
                        query_terms & set(tokenize(" ".join(instance.prompts)))
                    ),
                }

        return merge_variant_runs(
            self.name, self.view_type, runs, self.corpus.instance_ids,
            match_evidence=evidence,
        )
