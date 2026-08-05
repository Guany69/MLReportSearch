"""Structured, auditable explanations for query interpretation and ranking."""

from __future__ import annotations

from dataclasses import dataclass

from .query.expansion.types import TypoCorrection


@dataclass(frozen=True)
class InterpretedConcept:
    evidence: str
    canonical: str
    confidence: float
    mandatory: bool
    source: str


@dataclass(frozen=True)
class ResultExplanation:
    original_query: str
    normalized_query: str
    expanded_query: str
    interpreted_concepts: tuple[InterpretedConcept, ...]
    suppressed_concepts: tuple[InterpretedConcept, ...]
    typo_corrections: tuple[TypoCorrection, ...]
    matched_required_fields: tuple[str, ...]
    missing_required_fields: tuple[str, ...]
    matched_optional_concepts: tuple[str, ...]
    retriever_support: tuple[str, ...]
    ranking_reasons: tuple[str, ...]
    linkage_warnings: tuple[str, ...]
    ambiguity_reasons: tuple[str, ...] = ()


def _concept(value) -> InterpretedConcept:
    source = value.rule_ids[0] if value.rule_ids else value.origin
    return InterpretedConcept(
        value.evidence, value.canonical, value.confidence, value.mandatory, source,
    )


def build_result_explanation(intent, row, retriever_ranks, features) -> ResultExplanation:
    expansion = intent.expansion
    present = {str(field).casefold() for field in row.get("fields", ())}
    required = expansion.required_fields()
    optional = expansion.optional_fields()
    matched_required = tuple(field for field in required if field.casefold() in present)
    missing_required = tuple(field for field in required if field.casefold() not in present)
    matched_optional = tuple(field for field in optional if field.casefold() in present)
    warnings = []
    if bool(row.get("has_ambiguous_fields", False)) or float(
        row.get("ambiguous_link_fraction", 0.0) or 0.0
    ) > 0:
        warnings.append(
            "Some report-to-field links are ambiguous because duplicate report titles exist."
        )
    if float(row.get("field_link_confidence", 1.0) or 0.0) < 1.0:
        warnings.append("Field metadata linkage is inferred and is not independently verified.")
    reasons = []
    if matched_required:
        reasons.append("carries required fields: " + ", ".join(matched_required))
    if matched_optional:
        reasons.append("carries related fields: " + ", ".join(matched_optional))
    if features is not None and features.values.get("title_token_jaccard", 0) > 0:
        reasons.append("original query overlaps the report title")
    return ResultExplanation(
        expansion.original, expansion.normalized, expansion.expanded_query,
        tuple(_concept(value) for value in expansion.concepts),
        tuple(_concept(value) for value in expansion.suppressed),
        expansion.typos, matched_required, missing_required, matched_optional,
        tuple(f"{name} rank {rank}" for name, rank in sorted(retriever_ranks.items())),
        tuple(reasons), tuple(warnings), tuple(intent.ambiguity_reasons),
    )
