"""Backward-compatible adapter over the corpus-aware expansion engine."""

from __future__ import annotations

from dataclasses import dataclass, field, replace

from ..retrieval.forms import build_forms
from .expansion import ExpansionEngine, ExpansionResult, Requirement


@dataclass(frozen=True)
class IntentConcept:
    value: str
    evidence: str
    confidence: float
    mandatory: bool = False
    requirement: Requirement = Requirement.MENTIONED
    origin: str = "literal"


@dataclass
class QueryIntent:
    raw_query: str
    objectives: list[IntentConcept] = field(default_factory=list)
    populations: list[IntentConcept] = field(default_factory=list)
    fields: list[IntentConcept] = field(default_factory=list)
    measures: list[IntentConcept] = field(default_factory=list)
    dimensions: list[IntentConcept] = field(default_factory=list)
    granularity: str = ""
    comparison: str = ""
    time_freshness: str = ""
    categories: list[IntentConcept] = field(default_factory=list)
    data_sources: list[IntentConcept] = field(default_factory=list)
    business_objects: list[IntentConcept] = field(default_factory=list)
    subqueries: list[str] = field(default_factory=list)
    parser_confidence: float = 0.0
    normalized_query: str = ""
    expanded_query: str = ""
    expansion: ExpansionResult | None = None
    ambiguity_reasons: list[str] = field(default_factory=list)

    @property
    def mandatory_fields(self) -> list[str]:
        return [concept.value for concept in self.fields if concept.mandatory]

    def required_fields(self) -> list[str]:
        return self.mandatory_fields

    def optional_fields(self) -> list[str]:
        return [concept.value for concept in self.fields if not concept.mandatory]


class IntentParser:
    def __init__(self, frame, cfg=None):
        self.use_query_expansion = True if cfg is None else cfg.use_query_expansion
        options = {}
        if cfg is not None:
            options = {
                "use_typo_correction": cfg.use_typo_correction,
                "typo_max_distance": cfg.typo_max_distance,
                "max_typo_corrections": cfg.max_typo_corrections,
                "max_mandatory_fields": cfg.max_mandatory_fields,
                "expansion_weight": cfg.expansion_weight,
                "subquery_weight": cfg.subquery_weight,
            }
        self.engine = ExpansionEngine(frame, **options)

    def parse(self, query: str) -> QueryIntent:
        expansion = self.engine.expand(query.strip())
        if not self.use_query_expansion:
            literal = tuple(c for c in expansion.concepts if c.origin == "literal")
            expansion = replace(
                expansion, concepts=literal, suppressed=(), typos=(), ambiguities=(),
                subqueries=(), expanded_query=query.strip(),
                query_forms=build_forms(query.strip(), query.strip(), ()),
            )
        intent = QueryIntent(
            raw_query=query.strip(), normalized_query=expansion.normalized,
            expanded_query=expansion.expanded_query, expansion=expansion,
            subqueries=list(expansion.subqueries),
            ambiguity_reasons=[
                f"'{item.evidence}' could mean {', '.join(item.candidates)}"
                for item in expansion.ambiguities
            ],
        )
        targets = {
            "field": intent.fields, "category": intent.categories,
            "data_source": intent.data_sources,
            "business_object": intent.business_objects,
            "topic": intent.objectives,
        }
        for concept in expansion.concepts:
            targets[concept.kind].append(IntentConcept(
                concept.canonical, concept.evidence, concept.confidence,
                concept.mandatory, concept.requirement, concept.origin,
            ))
        lowered = query.casefold()
        intent.time_freshness = next(
            (term for term in ("today", "yesterday", "last quarter", "current", "previous", "month", "year", "week")
             if term in lowered), "",
        )
        intent.comparison = "comparison" if any(
            term in lowered for term in (" vs ", "versus", "compare", "difference")
        ) else ""
        signals = len(expansion.concepts)
        ambiguity_penalty = .15 * len(expansion.ambiguities)
        intent.parser_confidence = max(0.0, min(1.0, .35 + .15 * signals - ambiguity_penalty))
        return intent
