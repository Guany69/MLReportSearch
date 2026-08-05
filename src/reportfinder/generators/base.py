"""The candidate contract every generator satisfies.

The system invariant is that no retriever gates any other. Concretely that means
each generator:

* searches independently, over the full authorized universe;
* searches only authorized instances, before top-k rather than after;
* keeps its own raw score and its own rank, on its own scale;
* records which query representation and which view produced the hit;
* can nominate a report every other generator missed.

Scores from different generators are never compared directly -- a BM25F score, a
cosine and a SPLADE dot product have no common scale. Fusion is over *ranks*, and
the raw scores travel alongside as features and evidence.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable


@dataclass(frozen=True)
class CandidateHit:
    """One generator's nomination of one instance."""

    report_instance_id: str
    family_id: str
    generator: str
    query_variant: str
    view_type: str | None
    raw_score: float
    rank: int
    match_evidence: dict = field(default_factory=dict)


@dataclass(frozen=True)
class GeneratorResult:
    """Everything one generator found, in its own ranking."""

    generator: str
    query_variant: str
    view_type: str | None
    # (instance_id, rank, score) -- rank is 1-based within this generator.
    hits: tuple[tuple[str, int, float], ...] = ()
    match_evidence: dict[str, dict] = field(default_factory=dict)
    # Set when an optional generator failed. The pipeline continues without it and
    # discloses the failure rather than pretending the generator found nothing.
    error: str | None = None
    # Per-hit variant provenance, when a generator searched more than one variant.
    variant_by_instance: dict[str, str] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return self.error is None

    def variant_for(self, instance_id: str) -> str:
        return self.variant_by_instance.get(instance_id, self.query_variant)

    def to_hits(self, family_of) -> list[CandidateHit]:
        return [
            CandidateHit(
                report_instance_id=instance_id,
                family_id=family_of(instance_id),
                generator=self.generator,
                query_variant=self.variant_for(instance_id),
                view_type=self.view_type,
                raw_score=score,
                rank=rank,
                match_evidence=self.match_evidence.get(instance_id, {}),
            )
            for instance_id, rank, score in self.hits
        ]


@runtime_checkable
class CandidateGenerator(Protocol):
    """An independent source of candidates."""

    name: str
    view_type: str | None

    def generate(self, plan, universe, k: int) -> GeneratorResult: ...


def merge_variant_runs(
    generator: str,
    view_type: str | None,
    runs: list[tuple[str, list[tuple[int, float]]]],
    instance_ids,
    *,
    match_evidence: dict[str, dict] | None = None,
) -> GeneratorResult:
    """Combine one generator's runs over several query variants.

    Best rank wins, and the variant that produced it is recorded. Combining inside
    the generator keeps `generator` meaningful as a *source* for shortlist quotas,
    while `query_variant` still says which lens actually found the report.

    Variants add candidates; they never remove them. A report found only by Q4 is
    kept exactly like one found by Q0.
    """
    best: dict[str, tuple[int, float, str]] = {}
    for variant_key, hits in runs:
        for rank, (position, score) in enumerate(hits, start=1):
            instance_id = instance_ids[position]
            current = best.get(instance_id)
            if current is None or rank < current[0]:
                best[instance_id] = (rank, score, variant_key)

    ordered = sorted(best.items(), key=lambda item: (item[1][0], -item[1][1], item[0]))
    return GeneratorResult(
        generator=generator,
        query_variant=runs[0][0] if runs else "Q0",
        view_type=view_type,
        hits=tuple(
            (instance_id, rank, score)
            for rank, (instance_id, (_, score, _)) in enumerate(ordered, start=1)
        ),
        match_evidence=match_evidence or {},
        variant_by_instance={
            instance_id: variant for instance_id, (_, _, variant) in ordered
        },
    )
