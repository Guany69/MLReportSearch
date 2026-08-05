"""Grounding support measured over meaningful terms in the original query."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from ..query.normalize import depluralize

_STOPWORDS = frozenset(
    ["a", "an", "the", "of", "for", "by", "with", "to", "in", "on", "and", "or", "is", "are", "was", "were", "be", "been", "me", "all", "that", "this", "these", "those", "from", "at", "as", "it", "its", "any", "some"]
)
_INSTRUCTIONS = frozenset(["show", "list", "find", "get", "give", "need", "want", "pull", "run", "report", "reports"])


@dataclass(frozen=True)
class TermSupport:
    term: str
    supported: bool
    via: Literal["literal", "expansion", "typo", "acronym", "morphological", "none"]
    strength: float


@dataclass(frozen=True)
class SupportAccounting:
    terms: tuple[TermSupport, ...]
    ratio: float
    unsupported_terms: tuple[str, ...]
    unmet_required_fields: tuple[str, ...]
    required_field_count: int
    unknown_requested_fields: tuple[str, ...] = ()
    required_combination_exists: bool = True


def compute_support(
    expansion, corpus_tokens, *, top_fields=(), unknown_fields=(),
    required_combination_exists=True,
) -> SupportAccounting:
    required = expansion.required_fields()
    present = {str(field).casefold() for field in top_fields}
    unmet = tuple(field for field in required if field.casefold() not in present)
    typo_by_span = {item.span.tok_start: item for item in expansion.typos}
    terms = []
    for index, token in enumerate(expansion.tokens):
        value = token.text
        if value in _STOPWORDS or value in _INSTRUCTIONS:
            continue
        if value in corpus_tokens:
            terms.append(TermSupport(value, True, "literal", 1.0))
            continue
        singular = depluralize(value)
        if singular != value and singular in corpus_tokens:
            terms.append(TermSupport(value, True, "morphological", .95))
            continue
        typo = typo_by_span.get(index)
        if typo is not None and typo.confidence >= .75:
            terms.append(TermSupport(value, True, "typo", .80))
            continue
        covering = [
            concept for concept in expansion.concepts
            if concept.span.tok_start <= index < concept.span.tok_end
            and concept.confidence >= .75
        ]
        if covering:
            best = max(covering, key=lambda concept: concept.confidence)
            via = "acronym" if best.origin == "acronym" else "expansion"
            strength = .90 if via == "acronym" else best.confidence
            terms.append(TermSupport(value, True, via, strength))
        else:
            terms.append(TermSupport(value, False, "none", 0.0))
    ratio = sum(term.strength for term in terms) / len(terms) if terms else 1.0
    return SupportAccounting(
        tuple(terms), ratio, tuple(term.term for term in terms if not term.supported),
        unmet, len(required), tuple(dict.fromkeys(map(str, unknown_fields))),
        bool(required_combination_exists),
    )
