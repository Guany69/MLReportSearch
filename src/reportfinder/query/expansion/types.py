"""Stable query-expansion data contract."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Literal

from ...retrieval.forms import QueryForm
from ..normalize import Token

ConceptKind = Literal["field", "category", "data_source", "business_object", "topic"]
# The same five as a runtime set, for validating values that arrive as text.
CONCEPT_KINDS: frozenset[str] = frozenset(
    ("field", "category", "data_source", "business_object", "topic")
)
Origin = Literal["literal", "phrase_rule", "acronym", "morphological", "typo"]


@dataclass(frozen=True)
class Span:
    tok_start: int
    tok_end: int
    char_start: int
    char_end: int
    text: str

    def contains(self, other: Span) -> bool:
        return self.tok_start <= other.tok_start and other.tok_end <= self.tok_end


class Requirement(str, Enum):
    MENTIONED = "mentioned"
    PREFERRED = "preferred"
    REQUIRED = "required"


@dataclass(frozen=True)
class ExpansionConcept:
    canonical: str
    kind: ConceptKind
    confidence: float
    origin: Origin
    evidence: str
    span: Span
    requirement: Requirement = Requirement.MENTIONED
    rule_ids: tuple[str, ...] = ()
    suppressed_by: tuple[str, ...] = ()

    @property
    def mandatory(self) -> bool:
        return self.requirement is Requirement.REQUIRED


@dataclass(frozen=True)
class TypoCorrection:
    surface: str
    corrected: str
    distance: int
    runner_up: str | None
    runner_up_distance: int | None
    confidence: float
    span: Span


@dataclass(frozen=True)
class ExpansionAmbiguity:
    evidence: str
    candidates: tuple[str, ...]
    reason: str


@dataclass(frozen=True)
class ExpansionResult:
    original: str
    normalized: str
    tokens: tuple[Token, ...]
    concepts: tuple[ExpansionConcept, ...]
    suppressed: tuple[ExpansionConcept, ...]
    typos: tuple[TypoCorrection, ...]
    ambiguities: tuple[ExpansionAmbiguity, ...]
    query_forms: tuple[QueryForm, ...]
    lexicon_hash: str
    subqueries: tuple[str, ...] = ()
    expanded_query: str = ""

    @property
    def fired(self) -> bool:
        return any(concept.origin != "literal" for concept in self.concepts)

    def required_fields(self) -> tuple[str, ...]:
        return tuple(c.canonical for c in self.concepts
                     if c.kind == "field" and c.requirement is Requirement.REQUIRED)

    def optional_fields(self) -> tuple[str, ...]:
        return tuple(c.canonical for c in self.concepts
                     if c.kind == "field" and c.requirement is not Requirement.REQUIRED)
