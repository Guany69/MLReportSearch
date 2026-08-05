"""The source-preserving candidate union.

Merging is where recall is usually lost, because the natural implementation
flattens every generator into one score and throws the provenance away. This one
deliberately keeps everything:

* a candidate is never dropped because some *other* generator missed it;
* every generator's own rank and own raw score survive, on their own scales;
* a generator that did not retrieve a candidate is recorded as a **mask**, not as a
  zero. Zero is a score; "not retrieved" is the absence of one, and a model given a
  0 will learn that the generator voted against the candidate.

Fusion here is Reciprocal Rank Fusion over ranks only. Raw BM25F scores, cosines and
sparse dot products are never added together -- they have no common scale, and
normalizing them per query would make the top result's score depend on how good the
*second* result was.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field


@dataclass
class UnionRecord:
    """One candidate, with everything every generator said about it."""

    instance_id: str
    family_id: str
    ranks: dict[str, int] = field(default_factory=dict)
    # None means "this generator did not retrieve it" -- never 0.0.
    scores: dict[str, float | None] = field(default_factory=dict)
    variants: dict[str, str] = field(default_factory=dict)
    views: dict[str, str | None] = field(default_factory=dict)
    evidence: dict[str, dict] = field(default_factory=dict)
    fused_score: float = 0.0
    fused_rank: int = 0

    @property
    def found_by(self) -> set[str]:
        return {name for name, rank in self.ranks.items() if rank}

    @property
    def masked(self) -> set[str]:
        """Generators that ran but did not retrieve this candidate."""
        return {name for name, score in self.scores.items() if score is None}

    @property
    def generator_count(self) -> int:
        return len(self.found_by)

    @property
    def source_exclusive(self) -> bool:
        return self.generator_count == 1

    @property
    def exclusive_to(self) -> str | None:
        return next(iter(self.found_by)) if self.source_exclusive else None

    @property
    def query_variants(self) -> set[str]:
        return set(self.variants.values())

    def best_rank(self) -> int:
        return min(self.ranks.values()) if self.ranks else 10**6

    def telemetry(self) -> dict[str, object]:
        return {
            "instance_id": self.instance_id,
            "family_id": self.family_id,
            "found_by": sorted(self.found_by),
            "masked": sorted(self.masked),
            "source_exclusive": self.source_exclusive,
            "exclusive_to": self.exclusive_to,
            "query_variants": sorted(self.query_variants),
        }


@dataclass
class CandidateUnion:
    """Every candidate any generator nominated, with full provenance."""

    records: dict[str, UnionRecord] = field(default_factory=dict)
    generators_run: tuple[str, ...] = ()
    generators_failed: dict[str, str] = field(default_factory=dict)
    ordering: tuple[str, ...] = ()

    def __len__(self) -> int:
        return len(self.records)

    def __contains__(self, instance_id: str) -> bool:
        return instance_id in self.records

    def __iter__(self):
        return iter(self.ordered())

    def ordered(self) -> list[UnionRecord]:
        """Records in fused order."""
        return [self.records[i] for i in self.ordering if i in self.records]

    def exclusive_to(self, generator: str) -> list[UnionRecord]:
        """Candidates only this generator found, best fused rank first."""
        return [r for r in self.ordered() if r.exclusive_to == generator]

    def found_only_via_variant(self, variant_key: str) -> list[UnionRecord]:
        """Candidates every retrieving generator reached only through one variant.

        These are the candidates an alternate interpretation contributed. If the
        shortlist did not reserve room for them, adding the interpretation would
        have had no effect on what the ranker sees.
        """
        return [
            r for r in self.ordered()
            if r.query_variants == {variant_key}
        ]

    def counts_by_generator(self) -> dict[str, int]:
        counts: dict[str, int] = {name: 0 for name in self.generators_run}
        for record in self.records.values():
            for name in record.found_by:
                counts[name] = counts.get(name, 0) + 1
        return counts

    def source_exclusive_counts(self) -> dict[str, int]:
        counts: dict[str, int] = {name: 0 for name in self.generators_run}
        for record in self.records.values():
            if record.exclusive_to:
                counts[record.exclusive_to] = counts.get(record.exclusive_to, 0) + 1
        return counts

    def telemetry(self) -> dict[str, object]:
        return {
            "union_size": len(self),
            "generators_run": list(self.generators_run),
            "generators_failed": dict(self.generators_failed),
            "candidates_by_generator": self.counts_by_generator(),
            "source_exclusive_by_generator": self.source_exclusive_counts(),
            "source_exclusive_total": sum(
                1 for r in self.records.values() if r.source_exclusive
            ),
        }


def reciprocal_rank_fusion(union: CandidateUnion, constant: int = 60) -> None:
    """Score and order the union by RRF, in place.

    A generator that missed a candidate contributes nothing to its sum -- that is
    RRF's native handling of a missing source and is consistent with the mask: no
    vote, rather than a vote against.
    """
    for record in union.records.values():
        record.fused_score = sum(
            1.0 / (constant + rank) for rank in record.ranks.values() if rank
        )
    ordered = sorted(
        union.records.values(),
        # Deterministic: score, then best single rank, then id.
        key=lambda r: (-r.fused_score, r.best_rank(), r.instance_id),
    )
    for position, record in enumerate(ordered, start=1):
        record.fused_rank = position
    union.ordering = tuple(r.instance_id for r in ordered)


def build_union(
    results,
    *,
    family_of,
    rrf_constant: int = 60,
) -> CandidateUnion:
    """Merge generator results, preserving every source.

    `results` is an iterable of `GeneratorResult`. Failed generators are recorded
    with their error and excluded from the mask set -- masking against a generator
    that never ran would misreport "did not retrieve" as evidence.
    """
    results = list(results)
    ran = tuple(r.generator for r in results if r.ok)
    failed = {r.generator: r.error for r in results if not r.ok}

    union = CandidateUnion(generators_run=ran, generators_failed=failed)
    by_instance: dict[str, UnionRecord] = {}

    for result in results:
        if not result.ok:
            continue
        for instance_id, rank, score in result.hits:
            record = by_instance.get(instance_id)
            if record is None:
                record = UnionRecord(
                    instance_id=instance_id, family_id=family_of(instance_id)
                )
                by_instance[instance_id] = record
            # Deduplicated by instance id; the better rank from a generator wins if
            # it somehow reports the same instance twice.
            existing = record.ranks.get(result.generator)
            if existing is None or rank < existing:
                record.ranks[result.generator] = rank
                record.scores[result.generator] = float(score)
                record.variants[result.generator] = result.variant_for(instance_id)
                record.views[result.generator] = result.view_type
                evidence = result.match_evidence.get(instance_id)
                if evidence:
                    record.evidence[result.generator] = evidence

    # Explicit masks: for every generator that ran, every candidate it did not
    # retrieve gets None rather than being silently absent from the dict.
    for record in by_instance.values():
        for name in ran:
            record.scores.setdefault(name, None)

    union.records = by_instance
    reciprocal_rank_fusion(union, rrf_constant)
    return union


def survival_snapshot(union: CandidateUnion) -> dict[str, list[str]]:
    """Which generators reached each candidate. Used by candidate-survival evaluation."""
    snapshot: dict[str, list[str]] = defaultdict(list)
    for record in union.records.values():
        snapshot[record.instance_id] = sorted(record.found_by)
    return dict(snapshot)
