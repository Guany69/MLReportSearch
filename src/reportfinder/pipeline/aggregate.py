"""Instance scoring, family aggregation, family-first expansion, instance selection.

Order matters. Instances are scored first because the concrete row is what a user
runs: a family is only useful once you can say *which* copy answers the request.

Aggregation is max-oriented. A sum would reward families for having more instances,
which on this estate is an artefact of how the catalog was maintained rather than a
sign of relevance -- 533 titles appear more than once, up to seven times. Under a
sum, a seven-copy family would beat a better single-copy one on volume alone.

Bounded family-level features may nudge the score, but nothing that grows with
instance count is allowed to.

**Family-first expansion.** Retrieval nominates instances, so aggregating only the
instances that survived the shortlist answers the wrong question: it reports the
best *retrieved* copy, not the best copy. A family whose second row carries exactly
the requested fields loses to its own first row purely because that row ranked
higher lexically. So once the leading families are known, every authorized instance
inside them is pulled in and scored -- expansion is a second chance for the corpus,
not for the retrievers.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from ..corpus import authoritative_text

if TYPE_CHECKING:
    from .union import UnionRecord


@dataclass
class ScoredInstance:
    """One authorized instance, with its score and where it came from."""

    instance_id: str
    family_id: str
    score: float
    rank: int
    admitted_via: str
    cross_encoder_score: float | None
    record: UnionRecord
    # True when no retriever nominated this instance and it arrived through its
    # family. Kept explicit so nothing downstream mistakes it for retrieval evidence.
    expanded: bool = False
    suitability: float = 0.0

    @property
    def source_exclusive(self) -> bool:
        return self.record.source_exclusive


@dataclass
class FamilyResult:
    """A family, and the best authorized instance within it."""

    family_id: str
    canonical_title: str
    score: float
    rank: int
    selected: ScoredInstance
    instances: list[ScoredInstance] = field(default_factory=list)
    aliases: tuple[str, ...] = ()

    @property
    def instance_count(self) -> int:
        return len(self.instances)


def score_instances(shortlist, fusion, rerank) -> list[ScoredInstance]:
    """Attach the final score to every shortlisted instance."""
    scored = [
        ScoredInstance(
            instance_id=entry.instance_id,
            family_id=entry.record.family_id,
            score=fusion.scores.get(entry.instance_id, 0.0),
            rank=fusion.ranks.get(entry.instance_id, 10**6),
            admitted_via=entry.admitted_via,
            cross_encoder_score=rerank.scores.get(entry.instance_id),
            record=entry.record,
        )
        for entry in shortlist
    ]
    scored.sort(key=lambda s: (s.rank, s.instance_id))
    return scored


def aggregate_families(
    scored: list[ScoredInstance],
    corpus,
    *,
    universe=None,
) -> list[FamilyResult]:
    """Group instances into families and pick the best authorized one in each.

    `universe` is checked again here even though generators already filtered. This
    is the last point before results are exposed, and an unauthorized instance
    reaching it -- through a family mapping, a cache, or a future code path -- must
    not be selectable or able to influence a family's score.
    """
    grouped: dict[str, list[ScoredInstance]] = {}
    for instance in scored:
        if universe is not None:
            position = corpus.position_of(instance.instance_id)
            if not universe.allows_position(position):
                continue
        grouped.setdefault(instance.family_id, []).append(instance)

    families: list[FamilyResult] = []
    for family_id, members in grouped.items():
        family = corpus.families.get(family_id)
        # Best instance by final rank; ties broken deterministically by id.
        best = min(members, key=lambda m: (m.rank, m.instance_id))
        families.append(
            FamilyResult(
                family_id=family_id,
                canonical_title=family.canonical_title if family else family_id,
                # Max-oriented: the family is as good as its best instance, and
                # gains nothing from having more of them.
                score=max(m.score for m in members),
                rank=0,
                selected=best,
                instances=sorted(members, key=lambda m: (m.rank, m.instance_id)),
                aliases=family.aliases if family else (),
            )
        )

    families.sort(key=lambda f: (-f.score, f.selected.rank, f.family_id))
    for position, family in enumerate(families, start=1):
        family.rank = position
    return families


def aggregation_telemetry(families: list[FamilyResult]) -> dict[str, object]:
    return {
        "family_count": len(families),
        "instances_considered": sum(f.instance_count for f in families),
        "multi_instance_families": sum(1 for f in families if f.instance_count > 1),
        "source_exclusive_selected": sum(
            1 for f in families if f.selected.source_exclusive
        ),
        "expanded_selected": sum(1 for f in families if f.selected.expanded),
    }


# -- family-first expansion ------------------------------------------------

# What an expanded instance is judged on. These are the distinctions that decide
# *which copy* of a report to run, which is a different question from which report
# the user meant -- the family already answered that.
_SUITABILITY_WEIGHTS = {
    "fields": 0.45,
    "prompts": 0.20,
    "data_source": 0.15,
    "report_type": 0.10,
    "interface": 0.10,
}


@dataclass
class ExpansionResult:
    """What family expansion added, and what a cost bound stopped it adding."""

    scored: list[ScoredInstance] = field(default_factory=list)
    added: int = 0
    families_expanded: int = 0
    considered: int = 0
    dropped_by_cap: int = 0
    unauthorized_skipped: int = 0
    reranked: int = 0
    ran: bool = True
    skipped_reason: str | None = None

    def telemetry(self) -> dict[str, object]:
        return {
            "expansion_ran": self.ran,
            "expansion_skipped_reason": self.skipped_reason,
            "families_expanded": self.families_expanded,
            "instances_considered": self.considered,
            "instances_added": self.added,
            # Reported rather than silently truncated: a non-zero value means the
            # cost bound, not the catalog, decided what the user could see.
            "instances_dropped_by_cap": self.dropped_by_cap,
            "unauthorized_skipped": self.unauthorized_skipped,
            "expanded_cross_encoded": self.reranked,
        }


def _tokens(values) -> set[str]:
    out: set[str] = set()
    for value in values:
        for token in str(value).replace("/", " ").replace("-", " ").split():
            token = token.strip(",;:()").casefold()
            if len(token) > 2:
                out.add(token)
    return out


def instance_suitability(instance, plan) -> float:
    """How well one concrete instance fits the request, in [0, 1].

    Deterministic metadata compatibility, not a learned relevance estimate, and it
    is never presented as one. Requirements the query did not express contribute
    nothing in either direction -- an unmentioned facet is not evidence.
    """
    intent = getattr(plan, "intent", None)
    if intent is None:
        return 0.0

    def overlap(wanted: set[str], have: set[str]) -> float | None:
        if not wanted:
            return None  # not asked for; must not count as satisfied or violated
        return len(wanted & have) / len(wanted)

    parts = {
        "fields": overlap(
            _tokens([*intent.required_fields(), *intent.optional_fields()]),
            _tokens(instance.fields),
        ),
        "prompts": overlap(
            _tokens(c.value for c in intent.fields if c.mandatory),
            _tokens(instance.prompts),
        ),
        "data_source": overlap(
            _tokens(c.value for c in intent.data_sources),
            _tokens([instance.data_source]),
        ),
        "report_type": overlap(
            _tokens(c.value for c in intent.business_objects),
            _tokens([instance.report_type]),
        ),
        "interface": overlap(
            _tokens(c.value for c in intent.categories),
            _tokens([instance.category, instance.tags]),
        ),
    }

    scored = {k: v for k, v in parts.items() if v is not None}
    if not scored:
        return 0.0
    weight = sum(_SUITABILITY_WEIGHTS[k] for k in scored)
    return sum(_SUITABILITY_WEIGHTS[k] * v for k, v in scored.items()) / weight


def _expanded_record(instance_id: str, family_id: str, generators_run):
    """A union record for an instance no retriever nominated.

    Every generator is masked `None`, never 0.0 -- the same contract the union
    enforces. A zero would say "the retrievers looked and rejected it"; the truth is
    that this instance arrived through its family and the retrievers never voted.
    """
    from .union import UnionRecord

    return UnionRecord(
        instance_id=instance_id,
        family_id=family_id,
        ranks={},
        scores={name: None for name in generators_run},
        fused_score=0.0,
        fused_rank=10**6,
    )


def expand_families(
    families: list[FamilyResult],
    scored: list[ScoredInstance],
    corpus,
    universe,
    *,
    plan,
    cfg,
    scorer=None,
    union=None,
) -> ExpansionResult:
    """Pull in every authorized instance of the leading families and score it.

    Authorization is re-resolved here rather than trusted from retrieval: expansion
    is a *new* route by which an instance can reach a response, and a route that
    bypasses the generators must not also bypass the check the generators applied.
    """
    aggregation = cfg.aggregation
    if not aggregation.enabled:
        return ExpansionResult(ran=False, skipped_reason="aggregation.enabled=false")
    if not families:
        return ExpansionResult(ran=False, skipped_reason="no families to expand")

    already = {s.instance_id for s in scored}
    generators_run = tuple(union.generators_run) if union is not None else ()
    result = ExpansionResult()

    pending: list[tuple[str, str]] = []
    for family in families[: aggregation.family_expansion_k]:
        model = corpus.families.get(family.family_id)
        if model is None:
            continue
        family_added = False
        for instance_id in model.instance_ids:
            if instance_id in already:
                continue
            result.considered += 1
            if not universe.allows_position(corpus.position_of(instance_id)):
                result.unauthorized_skipped += 1
                continue
            if len(pending) >= aggregation.max_expanded_instances:
                result.dropped_by_cap += 1
                continue
            pending.append((instance_id, family.family_id))
            family_added = True
        if family_added:
            result.families_expanded += 1

    if not pending:
        return result

    # Cross-encode against the raw query, exactly as the shortlist was, so expanded
    # instances land on the same scale as the ones that were retrieved. Without a
    # scorer there is no comparable score, so suitability alone orders them and the
    # cross-encoder score stays honestly absent.
    ce_scores: dict[str, float] = {}
    if scorer is not None:
        import numpy as np

        texts = [
            authoritative_text(
                corpus.instance(instance_id), max_chars=cfg.rerank.max_report_chars
            )
            for instance_id, _ in pending
        ]
        raw = scorer.score_pairs(plan.raw_query, texts)
        for (instance_id, _), value, ok in zip(pending, raw, np.isfinite(raw), strict=True):
            if ok:
                ce_scores[instance_id] = float(value)
        result.reranked = len(ce_scores)

    for instance_id, family_id in pending:
        instance = corpus.instance(instance_id)
        result.scored.append(ScoredInstance(
            instance_id=instance_id,
            family_id=family_id,
            # No fused score exists: nothing fused it. The cross-encoder logit is
            # the only comparable evidence, and its absence is not a zero vote.
            score=ce_scores.get(instance_id, float("-inf")),
            rank=10**6,
            admitted_via="family_expansion",
            cross_encoder_score=ce_scores.get(instance_id),
            record=_expanded_record(instance_id, family_id, generators_run),
            expanded=True,
            suitability=instance_suitability(instance, plan),
        ))
    result.added = len(result.scored)
    return result


def attach_expanded(
    families: list[FamilyResult],
    expansion: ExpansionResult,
    corpus,
    plan,
) -> None:
    """Add expanded instances to their families and re-select within each.

    **Family scores and family ranks are deliberately left untouched.** An expanded
    instance carries a cross-encoder logit and no fused score; folding that into the
    family's max would mix two incomparable scales and let expansion silently
    reorder families. Expansion answers "which copy of this report", which is a
    different question from "which report" -- the ranking already answered that.

    Within a family, selection is by cross-encoder score first, then suitability,
    then retrieval rank. The reranker is the only stage that read the query and the
    report together, so it is the right arbiter; suitability breaks its ties on the
    concrete distinctions -- fields, prompts, data source -- that decide which copy
    someone actually runs. With no reranker every member's score is absent, and the
    ordering degrades exactly to the retrieval-rank ordering used before expansion.
    """
    by_family: dict[str, list[ScoredInstance]] = {}
    for instance in expansion.scored:
        by_family.setdefault(instance.family_id, []).append(instance)

    for family in families:
        extra = by_family.get(family.family_id)
        if extra:
            family.instances.extend(extra)
        # Suitability is computed for retrieved and expanded members alike. Scoring
        # only the expanded ones would hand them a tiebreak the retrieved ones were
        # never measured on.
        for member in family.instances:
            member.suitability = instance_suitability(
                corpus.instance(member.instance_id), plan
            )
        family.instances.sort(key=lambda m: (
            -(m.cross_encoder_score if m.cross_encoder_score is not None else -1e9),
            -m.suitability,
            m.rank,
            m.instance_id,
        ))
        family.selected = family.instances[0]
