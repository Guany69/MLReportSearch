"""SPLADE learned-sparse nomination.

SPLADE is searched with the raw query only, deliberately. Its whole value is that it
performs its own learned expansion in vocabulary space; feeding it the lexicon's
alias-expanded query would stack a hand-written expansion on top of a learned one
and blur what the generator actually contributes.
"""

from __future__ import annotations

from .base import GeneratorResult, merge_variant_runs

SPLADE_VARIANTS = ("Q0", "QC")


class SpladeGenerator:
    """Independent learned-sparse nomination."""

    name = "splade"
    view_type = None

    def __init__(self, index, encoder, corpus, *, variants=SPLADE_VARIANTS,
                 top_terms: int = 12) -> None:
        self.index = index
        self.encoder = encoder
        self.corpus = corpus
        self.variants = variants
        self.top_terms = top_terms

    def generate(self, plan, universe, k: int) -> GeneratorResult:
        selected = plan.variants_for(self.variants)
        runs = []
        evidence: dict[str, dict] = {}

        for variant in selected:
            query_vector = self.encoder.encode([variant.text])
            hits = self.index.search(query_vector, k, universe=universe)
            runs.append((variant.key, hits))

            expanded = self._expanded_terms(query_vector)
            for position, score in hits:
                instance_id = self.corpus.instance_ids[position]
                evidence.setdefault(instance_id, {
                    # The raw sparse dot product. Not comparable to a BM25F score or
                    # a cosine, and never presented as one.
                    "sparse_score": round(float(score), 4),
                    "query_expansion_terms": expanded,
                    "matched_via_variant": variant.key,
                })

        return merge_variant_runs(
            self.name, self.view_type, runs, self.corpus.instance_ids,
            match_evidence=evidence,
        )

    def _expanded_terms(self, query_vector) -> list[int]:
        """The heaviest query term ids, for explainability.

        Term ids rather than strings: decoding needs the tokenizer, which the fake
        encoder used in tests does not have. The pipeline records what SPLADE
        weighted, and the CLI can decode it when a real tokenizer is present.
        """
        row = query_vector.tocoo()
        ordered = sorted(zip(row.col.tolist(), row.data.tolist(), strict=True),
                         key=lambda item: -item[1])
        return [int(term) for term, _ in ordered[: self.top_terms]]
