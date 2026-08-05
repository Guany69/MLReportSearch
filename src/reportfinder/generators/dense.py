"""Dense retrieval, one generator per view.

Four independent generators rather than one over a blended document. A report whose
title says nothing useful but whose *fields* match is found by the schema generator;
one whose *purpose* matches is found by the purpose generator. Under a single
blended embedding both signals are averaged with everything else and neither is
strong enough to surface.

Each view generator also prefers the query representation aimed at it -- the purpose
view is searched with the business-intent phrasing, the schema view with field
phrasing -- while always searching the raw query as well.
"""

from __future__ import annotations

from ..corpus import ViewType
from .base import GeneratorResult, merge_variant_runs

# Which query representations each view is searched with. Q0 is implicit and always
# added by `QueryPlan.variants_for`.
VIEW_VARIANTS = {
    ViewType.IDENTITY: ("Q0", "Q1", "Q2", "Q5", "QC"),
    ViewType.PURPOSE: ("Q0", "Q3", "Q5", "QC"),
    ViewType.SCHEMA: ("Q0", "Q4", "Q2", "QC"),
    ViewType.INTERFACE: ("Q0", "Q1", "QC"),
}


class DenseViewGenerator:
    """Independent dense nomination from one view."""

    def __init__(self, index, encoder, corpus, *, variants=None) -> None:
        self.index = index
        self.encoder = encoder
        self.corpus = corpus
        self.view_type = index.view_type.value
        self.name = f"dense_{index.view_type.value}"
        self.variants = variants or VIEW_VARIANTS[index.view_type]

    def generate(self, plan, universe, k: int) -> GeneratorResult:
        selected = plan.variants_for(self.variants)
        # One encode call for every variant this view uses.
        vectors = self.encoder.encode_queries([v.text for v in selected])

        runs = []
        evidence: dict[str, dict] = {}
        for variant, vector in zip(selected, vectors, strict=True):
            hits = self.index.search(vector, k, universe=universe)
            runs.append((variant.key, hits))
            for position, score in hits:
                instance_id = self.corpus.instance_ids[position]
                evidence.setdefault(instance_id, {
                    "view": self.view_type,
                    "view_text": self.corpus.views[instance_id][
                        self.index.view_type
                    ].text[:200],
                    "cosine": round(float(score), 4),
                    "matched_via_variant": variant.key,
                })

        return merge_variant_runs(
            self.name, self.view_type, runs, self.corpus.instance_ids,
            match_evidence=evidence,
        )
