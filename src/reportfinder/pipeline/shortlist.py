"""Source-preserving shortlist selection.

The old pipeline reduced everything to one fused ordering and cut it at 100. That
single cut is where recall died: a report that only the schema view found sits low
in a fused ordering *precisely because* only one generator voted for it, so the
candidates most in need of protection were the first to go.

Here the depth is spent deliberately. Each source gets a reserved floor of slots for
candidates only it found; whatever the floors do not consume is filled from the
fused ordering. A report the schema view alone surfaced therefore reaches the
cross-encoder even when fifty better-fused decoys exist.

Three properties this must have, all asserted in tests:

* quotas are **floors**, not a partition -- unused reservations flow to `fused`, so
  a quiet generator never shrinks the shortlist;
* every admitted candidate is counted **once**, and records which quota admitted it;
* gold labels are never consulted. Selection is a serving-time decision.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .union import UnionRecord

# Which union query maps to which quota. Keeping this explicit means a new generator
# has to be given a quota deliberately rather than silently competing for `fused`.
EXCLUSIVE_QUOTA_SOURCES = {
    "bm25_exclusive": ("generator", "bm25f"),
    "splade_exclusive": ("generator", "splade"),
    "schema_exclusive": ("generator", "dense_schema"),
    "purpose_exclusive": ("generator", "dense_purpose"),
    "prototype_exclusive": ("generator", "prototype"),
    "late_interaction_exclusive": ("generator", "late_interaction"),
    "alternate_query_exclusive": ("variant", "Q5"),
}


@dataclass
class ShortlistEntry:
    """One candidate admitted to reranking, and why."""

    record: UnionRecord
    admitted_via: str
    position: int

    @property
    def instance_id(self) -> str:
        return self.record.instance_id


@dataclass
class Shortlist:
    """What the cross-encoder will score."""

    entries: list[ShortlistEntry] = field(default_factory=list)
    depth: int = 0
    risk_band: str = "LOW"
    quota_usage: dict[str, int] = field(default_factory=dict)
    quota_available: dict[str, int] = field(default_factory=dict)
    redistributed: dict[str, int] = field(default_factory=dict)
    family_diversified: bool = False
    deferred_then_admitted: int = 0

    def __len__(self) -> int:
        return len(self.entries)

    def __iter__(self):
        return iter(self.entries)

    def __contains__(self, instance_id: str) -> bool:
        return any(e.instance_id == instance_id for e in self.entries)

    @property
    def instance_ids(self) -> list[str]:
        return [e.instance_id for e in self.entries]

    def admitted_via(self, instance_id: str) -> str | None:
        return next(
            (e.admitted_via for e in self.entries if e.instance_id == instance_id), None
        )

    def telemetry(self) -> dict[str, object]:
        return {
            "shortlist_size": len(self),
            "shortlist_depth": self.depth,
            "risk_band": self.risk_band,
            "quota_usage": dict(self.quota_usage),
            "quota_available": dict(self.quota_available),
            "quota_redistributed_to_fused": dict(self.redistributed),
            "source_exclusive_admitted": sum(
                1 for e in self.entries if e.admitted_via != "fused"
            ),
            "family_diversity_enabled": self.family_diversified,
            "distinct_families": len({e.record.family_id for e in self.entries}),
            "sibling_entries": (
                len(self.entries) - len({e.record.family_id for e in self.entries})
            ),
            "deferred_then_admitted": self.deferred_then_admitted,
        }


def _pool_for(union, kind: str, key: str) -> list:
    if kind == "generator":
        return union.exclusive_to(key)
    if kind == "variant":
        return union.found_only_via_variant(key)
    raise ValueError(f"unknown quota source kind: {kind}")


def select_shortlist(union, risk, cfg) -> Shortlist:
    """Choose what reaches the cross-encoder, preserving every source."""
    depth = min(risk.rerank_depth, len(union))
    quotas = dict(cfg.shortlist.quota_map)
    scale = (
        cfg.shortlist.high_risk_quota_scale if risk.risk_band == "HIGH" else 1.0
    )

    shortlist = Shortlist(depth=depth, risk_band=risk.risk_band)
    if depth <= 0:
        shortlist.quota_available = quotas
        return shortlist

    if not cfg.shortlist.preserve_source_exclusive_candidates:
        # Explicit ablation path: one global ordering, the behaviour being replaced.
        for position, record in enumerate(union.ordered()[:depth]):
            shortlist.entries.append(ShortlistEntry(record, "fused", position))
        shortlist.quota_usage = {"fused": len(shortlist.entries)}
        shortlist.quota_available = {"fused": depth}
        return shortlist

    admitted: set[str] = set()
    available: dict[str, int] = {}

    # Exclusive floors first: these are the candidates a fused cut would drop.
    for quota_name, (kind, key) in EXCLUSIVE_QUOTA_SOURCES.items():
        allowance = int(round(quotas.get(quota_name, 0) * scale))
        available[quota_name] = allowance
        if allowance <= 0:
            continue

        used = 0
        for record in _pool_for(union, kind, key):
            if used >= allowance or len(admitted) >= depth:
                break
            if record.instance_id in admitted:
                continue
            admitted.add(record.instance_id)
            shortlist.entries.append(
                ShortlistEntry(record, quota_name, len(shortlist.entries))
            )
            used += 1
        shortlist.quota_usage[quota_name] = used
        if used < allowance:
            # Unclaimed reservations are not lost; they widen the fused fill.
            shortlist.redistributed[quota_name] = allowance - used

    # `fused` takes the remainder, so the shortlist always reaches its full depth.
    fused_used = 0
    if cfg.shortlist.diversify_families:
        # Two passes over the same fused ordering. The first spends the remainder
        # on families nothing has admitted yet; the second gives whatever is left
        # to the siblings it passed over.
        #
        # The reason is that aggregation is max-oriented over families, so the
        # second copy of a family already in the shortlist can only improve *which
        # copy* that family offers -- while a family with no representative at all
        # cannot be returned no matter how good it is. Family-first expansion then
        # cross-encodes the skipped siblings anyway, so a deferred sibling is not
        # lost, only deprioritised for a slot.
        #
        # This never touches the exclusive floors above, and it cannot change the
        # shortlist's size: every deferred candidate is reconsidered in pass two.
        shortlist.family_diversified = True
        represented = {entry.record.family_id for entry in shortlist.entries}
        deferred: list[UnionRecord] = []

        for record in union.ordered():
            if len(admitted) >= depth:
                break
            if record.instance_id in admitted:
                continue
            if record.family_id in represented:
                deferred.append(record)
                continue
            admitted.add(record.instance_id)
            represented.add(record.family_id)
            shortlist.entries.append(
                ShortlistEntry(record, "fused", len(shortlist.entries))
            )
            fused_used += 1

        for record in deferred:
            if len(admitted) >= depth:
                break
            if record.instance_id in admitted:
                continue
            admitted.add(record.instance_id)
            shortlist.entries.append(
                ShortlistEntry(record, "fused", len(shortlist.entries))
            )
            fused_used += 1
            shortlist.deferred_then_admitted += 1
    else:
        for record in union.ordered():
            if len(admitted) >= depth:
                break
            if record.instance_id in admitted:
                continue
            admitted.add(record.instance_id)
            shortlist.entries.append(
                ShortlistEntry(record, "fused", len(shortlist.entries))
            )
            fused_used += 1

    shortlist.quota_usage["fused"] = fused_used
    available["fused"] = depth - sum(
        v for k, v in available.items() if k != "fused"
    )
    shortlist.quota_available = available

    # Ordering is by fused rank so the cross-encoder sees a sensible sequence; the
    # admitting quota is retained on each entry regardless of position.
    shortlist.entries.sort(key=lambda e: e.record.fused_rank)
    for position, entry in enumerate(shortlist.entries):
        entry.position = position
    return shortlist
