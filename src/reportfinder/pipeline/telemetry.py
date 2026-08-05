"""Structured telemetry, including where candidates died.

The most useful field here is `survival_trace`. Recall failures are hard to debug
after the fact because every stage looks reasonable in isolation -- the answer is
simply absent at the end. Recording which boundary each candidate crossed turns
"the right report did not come back" into "it was found by dense_schema, entered
the union, and was dropped at shortlist selection", which names the bug.

What is deliberately *not* logged: the raw query text and unrestricted report
metadata. Queries are user content, and this repository has no policy permitting
their retention. Ids, counts and versions are safe; the words people typed are not.
"""

from __future__ import annotations

from dataclasses import dataclass, field

# The pruning boundaries a candidate can be lost at, in order.
#
# `expansion` sits between fusion and final because it is the one stage that can
# *add* a candidate rather than only remove one: an instance that no retriever
# nominated can enter here through its family. Giving it its own boundary keeps
# loss attribution truthful -- a candidate lost after it must have been lost at
# final ranking, not at retrieval.
STAGES = (
    "generated", "union", "shortlist", "rerank", "fusion", "expansion", "final",
)


@dataclass
class SearchTelemetry:
    """One request's structured record."""

    request_id: str
    catalog_version: str = ""
    model_bundle_version: str = ""
    code_version: str = ""

    authorization: dict[str, object] = field(default_factory=dict)
    query: dict[str, object] = field(default_factory=dict)
    retrieval: dict[str, object] = field(default_factory=dict)
    risk: dict[str, object] = field(default_factory=dict)
    shortlist: dict[str, object] = field(default_factory=dict)
    rerank: dict[str, object] = field(default_factory=dict)
    fusion: dict[str, object] = field(default_factory=dict)
    aggregation: dict[str, object] = field(default_factory=dict)
    decision: dict[str, object] = field(default_factory=dict)

    active_fallbacks: list[str] = field(default_factory=list)
    degraded_components: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    latency_ms: dict[str, float] = field(default_factory=dict)

    # instance_id -> the boundaries it survived.
    survival_trace: dict[str, tuple[str, ...]] = field(default_factory=dict)
    # generator -> the instances it nominated. Needed for per-generator recall and
    # for measuring what each source *uniquely* contributes; an aggregate count
    # cannot answer either question.
    generator_hits: dict[str, tuple[str, ...]] = field(default_factory=dict)

    def record_stage(self, instance_id: str, stage: str) -> None:
        current = self.survival_trace.get(instance_id, ())
        if stage not in current:
            self.survival_trace[instance_id] = (*current, stage)

    def record_many(self, instance_ids, stage: str) -> None:
        for instance_id in instance_ids:
            self.record_stage(instance_id, stage)

    def survived_to(self, instance_id: str) -> str | None:
        """The last boundary this candidate crossed."""
        trace = self.survival_trace.get(instance_id)
        return trace[-1] if trace else None

    def lost_at(self, instance_id: str) -> str | None:
        """The boundary that dropped this candidate, or None if it survived."""
        last = self.survived_to(instance_id)
        if last is None:
            return "never_generated"
        if last == "final":
            return None
        index = STAGES.index(last)
        return STAGES[index + 1] if index + 1 < len(STAGES) else None

    def as_dict(self) -> dict[str, object]:
        return {
            "request_id": self.request_id,
            "catalog_version": self.catalog_version,
            "model_bundle_version": self.model_bundle_version,
            "code_version": self.code_version,
            "authorization": self.authorization,
            "query": self.query,
            "retrieval": self.retrieval,
            "risk": self.risk,
            "shortlist": self.shortlist,
            "rerank": self.rerank,
            "fusion": self.fusion,
            "aggregation": self.aggregation,
            "decision": self.decision,
            "active_fallbacks": list(self.active_fallbacks),
            "degraded_components": list(self.degraded_components),
            "warnings": list(self.warnings),
            "latency_ms": {k: round(v, 2) for k, v in self.latency_ms.items()},
        }

    def with_survival(self) -> dict[str, object]:
        """The full record including per-candidate survival.

        Kept separate from `as_dict` because it is large and is only wanted for
        offline judged evaluation, not for every served request.
        """
        payload = self.as_dict()
        payload["generator_hits"] = {
            generator: list(ids) for generator, ids in sorted(self.generator_hits.items())
        }
        payload["survival_trace"] = {
            instance_id: list(stages)
            for instance_id, stages in sorted(self.survival_trace.items())
        }
        return payload
