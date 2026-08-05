"""Candidate-survival evaluation: where relevant reports are lost.

Final ranking metrics cannot diagnose a recall failure. If the right report was cut
at the shortlist, nDCG@5 simply reads low, and it reads exactly the same as if the
report had been retrieved and ranked badly -- two problems with opposite fixes. So
this measures survival at every pruning boundary and attributes each loss to the
stage that caused it.

It also measures what each generator *uniquely* contributes. A generator that only
ever finds reports the others already found is cost without recall, and that is
invisible in any aggregate metric.
"""

from __future__ import annotations

import math
from collections import Counter, defaultdict
from dataclasses import dataclass, field

from ..pipeline.telemetry import STAGES

GENERATOR_NAMES = (
    "bm25f", "splade", "dense_identity", "dense_purpose", "dense_schema",
    "dense_interface", "prototype", "late_interaction",
)


@dataclass
class QuerySurvival:
    """One judged query's survival record."""

    query_id: str
    relevant_instances: set[str]
    relevant_families: set[str]
    found_by_generator: dict[str, set[str]] = field(default_factory=dict)
    reached: dict[str, set[str]] = field(default_factory=dict)
    returned_families: list[str] = field(default_factory=list)
    returned_instances: list[str] = field(default_factory=list)
    decision: str = ""

    # The decision policy's own inputs. Persisted because both of its thresholds
    # are post-hoc functions of these, so keeping them turns a threshold sweep
    # into arithmetic over one run instead of one run per grid point -- and they
    # were previously recorded nowhere, which is why the least-evidenced numbers
    # in the system were also the most expensive to re-derive.
    ce_top: float | None = None
    ce_second: float | None = None
    risk_band: str = ""

    def recall_at(self, stage: str) -> float:
        if not self.relevant_instances:
            return 0.0
        return len(self.reached.get(stage, set()) & self.relevant_instances) / len(
            self.relevant_instances
        )

    def generator_recall(self, generator: str) -> float:
        if not self.relevant_instances:
            return 0.0
        found = self.found_by_generator.get(generator, set())
        return len(found & self.relevant_instances) / len(self.relevant_instances)

    def loss_boundary(self) -> Counter:
        """Which boundary each relevant instance was lost at."""
        counts: Counter = Counter()
        for instance_id in self.relevant_instances:
            last = None
            for stage in STAGES:
                if instance_id in self.reached.get(stage, set()):
                    last = stage
            if last == "final":
                counts["survived"] += 1
            elif last is None:
                counts["never_generated"] += 1
            else:
                counts[STAGES[STAGES.index(last) + 1]] += 1
        return counts

    def unique_contributions(self) -> dict[str, int]:
        """Relevant instances only this generator found."""
        out: dict[str, int] = {}
        for generator, found in self.found_by_generator.items():
            others: set[str] = set()
            for other, found_other in self.found_by_generator.items():
                if other != generator:
                    others |= found_other
            out[generator] = len((found & self.relevant_instances) - others)
        return out


def observe(outcome, *, query_id: str, relevant_instances: set[str],
            family_of) -> QuerySurvival:
    """Turn one pipeline outcome into a survival record."""
    telemetry = outcome.telemetry
    record = QuerySurvival(
        query_id=query_id,
        relevant_instances=set(relevant_instances),
        relevant_families={family_of(i) for i in relevant_instances},
        decision=outcome.decision.value,
        returned_families=[f.family_id for f in outcome.families],
        returned_instances=[f.selected.instance_id for f in outcome.families],
    )
    # Read from the decision's own inputs, not from `outcome.families`: that list
    # is empty on NO_CONFIDENT_MATCH, which is exactly the case a sweep needs.
    detail = outcome.decision_detail
    ranked = getattr(detail, "ranked_families", None) or outcome.families
    ce = [
        f.selected.cross_encoder_score for f in ranked
        if f.selected.cross_encoder_score is not None
    ]
    record.ce_top = ce[0] if ce else None
    record.ce_second = ce[1] if len(ce) > 1 else None
    if telemetry is not None:
        record.risk_band = str(telemetry.risk.get("risk_band", ""))
    if telemetry is None:
        return record

    for stage in STAGES:
        record.reached[stage] = {
            instance_id for instance_id, stages in telemetry.survival_trace.items()
            if stage in stages
        }
    record.found_by_generator = {
        generator: set(ids) for generator, ids in telemetry.generator_hits.items()
    }
    return record


def _ndcg(returned: list[str], relevant: set[str], k: int) -> float:
    gains = [1.0 if item in relevant else 0.0 for item in returned[:k]]
    dcg = sum(g / math.log2(i + 2) for i, g in enumerate(gains))
    ideal = sum(1.0 / math.log2(i + 2) for i in range(min(len(relevant), k)))
    return dcg / ideal if ideal else 0.0


def _mrr(returned: list[str], relevant: set[str]) -> float:
    for position, item in enumerate(returned, start=1):
        if item in relevant:
            return 1.0 / position
    return 0.0


def summarize(
    records: list[QuerySurvival], *, slices=None, per_query: bool = True,
) -> dict[str, object]:
    """Aggregate survival, per-generator recall, and final ranking quality."""
    if not records:
        return {"query_count": 0}

    def mean(values):
        values = list(values)
        return round(sum(values) / len(values), 4) if values else 0.0

    stage_recall = {
        stage: mean(r.recall_at(stage) for r in records) for stage in STAGES
    }
    generator_recall = {
        generator: mean(r.generator_recall(generator) for r in records)
        for generator in GENERATOR_NAMES
        if any(generator in r.found_by_generator for r in records)
    }
    unique: defaultdict[str, int] = defaultdict(int)
    for record in records:
        for generator, count in record.unique_contributions().items():
            unique[generator] += count

    losses: Counter = Counter()
    for record in records:
        losses.update(record.loss_boundary())
    total_relevant = sum(len(r.relevant_instances) for r in records)

    summary: dict[str, object] = {
        "query_count": len(records),
        "relevant_instances_total": total_relevant,
        # The headline: what fraction of relevant reports survived each boundary.
        "recall_by_stage": stage_recall,
        "recall_by_generator": generator_recall,
        "unique_relevant_contributed_by_generator": dict(sorted(unique.items())),
        "loss_by_boundary": dict(losses),
        "positive_loss_rate_by_boundary": {
            stage: round(count / total_relevant, 4)
            for stage, count in losses.items()
            if total_relevant
        },
        "family_ndcg@5": mean(
            _ndcg(r.returned_families, r.relevant_families, 5) for r in records
        ),
        "family_mrr": mean(_mrr(r.returned_families, r.relevant_families) for r in records),
        "instance_recall@1_within_correct_family": mean(
            1.0 if r.returned_instances[:1] and r.returned_instances[0] in r.relevant_instances
            else 0.0
            for r in records
        ),
        "decisions": dict(Counter(r.decision for r in records)),
        # Stated on every summary: these labels are synthetic, so none of this is
        # production accuracy.
        "label_basis": "synthetic weak supervision; not production accuracy",
    }

    if per_query:
        # The decision policy's inputs, per query, so its two thresholds can be
        # swept offline instead of by re-running the pipeline per grid point.
        # Omitted inside slices, where it would just be the same rows again.
        summary["per_query"] = [
            {
                "query_id": r.query_id, "decision": r.decision,
                "ce_top": r.ce_top, "ce_second": r.ce_second,
                "risk_band": r.risk_band,
                "relevant_instances": sorted(r.relevant_instances),
                "returned_instances": list(r.returned_instances),
                "recall_final": round(r.recall_at("final"), 4),
            }
            for r in records
        ]

    if slices:
        summary["slices"] = {
            name: summarize(
                [r for r in records if r.query_id in ids], per_query=False,
            )
            for name, ids in slices.items()
        }
    return summary
