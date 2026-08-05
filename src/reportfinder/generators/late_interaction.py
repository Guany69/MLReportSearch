"""Token-level late-interaction nomination. Real, and currently inactive.

Triggered only for high-recall-risk queries, because MaxSim over per-token vectors
is far too expensive to run on every request. When the risk policy does not ask for
it, or no index exists, the generator is simply not constructed -- and the pipeline
records that it did not run.

The failure mode this guards against is substitution: quietly running another dense
search and labelling it "late interaction". `status()` reports what actually
happened, and the telemetry carries it into the response.
"""

from __future__ import annotations

from ..index.late_interaction import LateInteractionStatus
from .base import GeneratorResult, merge_variant_runs

LATE_INTERACTION_VARIANTS = ("Q0", "QC")


class LateInteractionGenerator:
    """MaxSim nomination over per-token vectors."""

    name = "late_interaction"
    view_type = None

    def __init__(self, index, encoder, corpus, *, variants=LATE_INTERACTION_VARIANTS) -> None:
        self.index = index
        self.encoder = encoder
        self.corpus = corpus
        self.variants = variants

    def status(self) -> LateInteractionStatus:
        return LateInteractionStatus(enabled=True, built=True, reason="active")

    def generate(self, plan, universe, k: int) -> GeneratorResult:
        selected = plan.variants_for(self.variants)
        vectors, masks = self.encoder.encode_queries([v.text for v in selected])

        runs = []
        evidence: dict[str, dict] = {}
        for index, variant in enumerate(selected):
            hits = self.index.search(vectors[index], masks[index], k, universe=universe)
            runs.append((variant.key, hits))
            for position, score in hits:
                instance_id = self.corpus.instance_ids[position]
                evidence.setdefault(instance_id, {
                    "maxsim_score": round(float(score), 4),
                    "matched_via_variant": variant.key,
                })

        return merge_variant_runs(
            self.name, self.view_type, runs, self.corpus.instance_ids,
            match_evidence=evidence,
        )


class DisabledLateInteractionGenerator:
    """Stands in when no index exists, and says so.

    Returning an empty result with `error` set is deliberate: an empty result with
    no error would be indistinguishable from "ran and found nothing", which is
    exactly the confusion that lets a disabled component look active.
    """

    name = "late_interaction"
    view_type = None

    def __init__(self, reason: str) -> None:
        self.reason = reason

    def status(self) -> LateInteractionStatus:
        return LateInteractionStatus(enabled=False, built=False, reason=self.reason)

    def generate(self, plan, universe, k: int) -> GeneratorResult:
        return GeneratorResult(
            generator=self.name, query_variant="Q0", view_type=None,
            hits=(), error=f"not_run: {self.reason}",
        )
