"""Bounded structured feature schema and deterministic fallback ranker."""

from __future__ import annotations

import hashlib
import math
import re
from dataclasses import dataclass

from ..query.normalize import depluralize

FEATURE_SCHEMA_VERSION = "4"
RETRIEVERS = ("bm25", "dense", "lsa", "char", "exact_field")
FEATURE_NAMES = (
    "rrf", *(f"{r}_{suffix}" for r in RETRIEVERS for suffix in
             ("raw", "zscore", "percentile", "reciprocal_rank", "present")),
    "retriever_agreement_count", "retriever_agreement_strength",
    "mandatory_coverage", "has_mandatory_requirements", "mandatory_coverage_missing",
    "missing_mandatory", "optional_concept_coverage", "granularity_match",
    "time_match", "prompt_compatibility", "filter_compatibility", "negation_violation",
    "exclusion_violation", "duplicate_title_count", "family_size", "intra_family_rank",
    "title_token_jaccard", "description_token_jaccard", "exact_field_match_strength",
    "interpreted_title_coverage",
    "subquery_coverage", "subquery_count_satisfied", "intent_coverage",
    "candidate_rank_before_ltr", "field_link_confidence",
    "ambiguous_link_fraction", "category_match", "data_source_match",
    "business_object_match", "usage_prior", "freshness_prior", "shared_status_prior",
    "rrf_norm", "bm25_norm", "dense_norm", "lsa_norm", "char_norm",
    "exact_field_norm", "usage_prior_n", "retriever_agreement_count_n",
)
FEATURE_COMPUTATION_VERSION = "4.1"
FEATURE_HASH = hashlib.sha256((
    FEATURE_SCHEMA_VERSION + "\0" + "\0".join(FEATURE_NAMES)
    + "\0" + FEATURE_COMPUTATION_VERSION
).encode()).hexdigest()

_TOKEN_RE = re.compile(r"[a-z0-9]+")
_STOP = frozenset(
    ["a", "an", "the", "of", "for", "by", "with", "to", "in", "on", "and", "or", "is", "are", "show", "me", "list", "get", "find", "give", "report", "reports", "which", "what", "who", "how", "many"]
)


@dataclass(frozen=True)
class RankingFeatures:
    values: dict[str, float]

    def vector(self) -> list[float]:
        return [float(self.values.get(name, 0.0)) for name in FEATURE_NAMES]


def _tokens(value) -> set[str]:
    if isinstance(value, (list, tuple)):
        value = " ".join(map(str, value))
    return {token for token in _TOKEN_RE.findall(str(value or "").casefold()) if token not in _STOP}


def _jaccard(left, right) -> float:
    a, b = _tokens(left), _tokens(right)
    return len(a & b) / len(a | b) if a | b else 0.0


def extract_features(row, intent, scores: dict[str, float],
                     relative: dict[str, dict[str, float]] | None = None) -> RankingFeatures:
    requested = intent.mandatory_fields
    optional = intent.optional_fields() if hasattr(intent, "optional_fields") else []
    present = {str(field).casefold() for field in row["fields"]}
    covered = sum(field.casefold() in present for field in requested)
    coverage = covered / len(requested) if requested else 0.0
    optional_covered = sum(field.casefold() in present for field in optional)
    optional_coverage = optional_covered / len(optional) if optional else 0.0
    values = {name: float(scores.get(name, 0.0)) for name in FEATURE_NAMES}
    relative = relative or {}
    active = int(scores.get("n_active_retrievers", 0)) or sum(
        retriever in relative for retriever in RETRIEVERS
    )
    agreement_count = 0.0
    agreement_strength = 0.0
    for retriever in RETRIEVERS:
        values[f"{retriever}_raw"] = float(scores.get(retriever, 0.0))
        for suffix in ("zscore", "percentile", "reciprocal_rank", "present"):
            values[f"{retriever}_{suffix}"] = float(
                relative.get(retriever, {}).get(suffix, 0.0)
            )
        agreement_count += values[f"{retriever}_present"]
        agreement_strength += values[f"{retriever}_reciprocal_rank"]
    agreement_strength = agreement_strength / max(1, active)

    row_fields = " ".join(map(str, row.get("fields", ())))
    row_prompts = " ".join(map(str, row.get("prompts", ())))
    full_evidence = " ".join((
        str(row.get("title", "")), row_fields, str(row.get("category", "")),
        str(row.get("data_source", "")),
        " ".join(getattr(meta, "business_object", "") for meta in row.get("field_meta", ())),
    )).casefold()
    concepts = []
    if getattr(intent, "expansion", None) is not None:
        concepts = list(intent.expansion.concepts)
    intent_coverage = (
        sum(concept.canonical.casefold() in full_evidence for concept in concepts) / len(concepts)
        if concepts else 0.0
    )
    title_tokens = {depluralize(token) for token in _tokens(row.get("title", ""))}
    generic_title_concepts = {"worker", "manager", "leader", "organization", "company"}
    title_concepts = [
        concept for concept in concepts
        if concept.canonical.casefold() not in generic_title_concepts
    ]
    interpreted_title_coverage = (
        sum(bool(
            {depluralize(token) for token in _tokens(concept.canonical)} & title_tokens
        ) for concept in title_concepts)
        / len(title_concepts) if title_concepts else 0.0
    )
    negated = re.findall(
        r"\b(?:without|excluding|exclude|no|not)\s+([a-z0-9]+(?:\s+[a-z0-9]+){0,3})",
        intent.raw_query.casefold(),
    )
    negation_violation = float(any(
        bool(_tokens(term)) and _tokens(term) <= _tokens(row_fields)
        for term in negated
    ))
    temporal_requested = bool(intent.time_freshness)
    time_capable = bool(
        _tokens(row_fields) & {"date", "period", "month", "quarter", "year", "week"}
        or _tokens(row_prompts) & {"date", "period", "month", "quarter", "year", "week"}
    )
    title_jaccard = _jaccard(intent.raw_query, row.get("title", ""))
    description_jaccard = _jaccard(intent.raw_query, row.get("description", ""))
    usage_prior = math.log1p(max(0, int(row.get("runs", 0) or 0)))
    rrf_k = float(scores.get("rrf_k", 60))
    query_lower = intent.raw_query.casefold()
    prompt_tokens = _tokens(row_prompts)
    requested_concepts = [
        concept for concept in concepts
        if concept.kind == "field" and concept.confidence >= .80
    ]

    def concept_coverage(evidence_tokens: set[str]) -> float:
        if not requested_concepts:
            return .5
        covered_concepts = sum(
            bool(_tokens(concept.canonical) & evidence_tokens)
            for concept in requested_concepts
        )
        return covered_concepts / len(requested_concepts)

    has_filter_request = bool(re.search(
        r"\b(?:where|filter(?:ed)?|with|whose|having|between|before|after)\b",
        query_lower,
    ))
    has_granularity_request = bool(re.search(
        r"\b(?:by|per|for each|grouped by|broken down by|split by)\b",
        query_lower,
    ))
    prompt_compatibility = concept_coverage(prompt_tokens)
    filter_compatibility = (
        concept_coverage(prompt_tokens | _tokens(row_fields))
        if has_filter_request else .5
    )
    granularity_match = (
        concept_coverage(_tokens(row_fields) | _tokens(row.get("title", "")))
        if has_granularity_request else .5
    )
    values.update({
        "mandatory_coverage": coverage,
        "has_mandatory_requirements": float(bool(requested)),
        "mandatory_coverage_missing": float(not requested),
        "missing_mandatory": min(1.0, max(0.0, (len(requested) - covered) / max(1, len(requested)))),
        "optional_concept_coverage": optional_coverage,
        "retriever_agreement_count": agreement_count,
        "retriever_agreement_strength": agreement_strength,
        "retriever_agreement_count_n": agreement_count / max(1, active),
        "field_link_confidence": float(row.get("field_link_confidence", 0.0) or 0.0),
        "ambiguous_link_fraction": float(row.get("ambiguous_link_fraction", 0.0) or 0.0),
        "category_match": float(any(
            concept.value.casefold() == str(row.get("category", "")).casefold()
            for concept in intent.categories
        )),
        "data_source_match": float(any(
            concept.value.casefold() == str(row.get("data_source", "")).casefold()
            for concept in intent.data_sources
        )),
        "business_object_match": float(any(
            concept.value.casefold() in {
                getattr(meta, "business_object", "").casefold()
                for meta in row.get("field_meta", ())
            } for concept in intent.business_objects
        )),
        "usage_prior": usage_prior,
        "usage_prior_n": min(1.0, usage_prior / math.log1p(10_000)),
        "freshness_prior": float(not bool(row.get("last_updated_missing", True))),
        "shared_status_prior": float(str(row.get("shared", "")).casefold() in {"yes", "true", "1"}),
        "family_size": float(row.get("family_size", 1) or 1),
        "intra_family_rank": float(row.get("family_rank", 1) or 1),
        "duplicate_title_count": float(row.get("family_size", 1) or 1),
        "exact_field_match_strength": float(scores.get("exact_field", coverage)),
        "exact_field_norm": min(1.0, max(0.0, float(scores.get("exact_field", 0.0)))),
        "rrf_norm": min(1.0, max(0.0, float(scores.get("rrf", 0.0))
                                * (rrf_k + 1) / max(1, active))),
        "bm25_norm": min(1.0, max(0.0, float(scores.get("bm25", 0.0)))),
        "dense_norm": min(1.0, max(0.0, float(scores.get("dense", 0.0)))),
        "lsa_norm": min(1.0, max(0.0, float(scores.get("lsa", 0.0)))),
        "char_norm": min(1.0, max(0.0, float(scores.get("char", 0.0)))),
        "title_token_jaccard": title_jaccard,
        "description_token_jaccard": description_jaccard,
        "intent_coverage": intent_coverage,
        "interpreted_title_coverage": interpreted_title_coverage,
        "subquery_coverage": min(1.0, max(0.0, float(scores.get("subquery_coverage", 0.0)))),
        "subquery_count_satisfied": min(1.0, max(0.0, float(scores.get("subquery_count_satisfied", 0.0)))),
        "candidate_rank_before_ltr": min(1.0, max(0.0, float(scores.get("candidate_rank_before_ltr", 0.0)))),
        "time_match": float(time_capable) if temporal_requested else .5,
        "prompt_compatibility": float(scores.get("prompt_compatibility", prompt_compatibility)),
        "filter_compatibility": float(scores.get("filter_compatibility", filter_compatibility)),
        "granularity_match": float(scores.get("granularity_match", granularity_match)),
        "negation_violation": negation_violation,
        "exclusion_violation": float(scores.get("exclusion_violation", 0.0)),
    })
    return RankingFeatures(values)


def mandatory_gate(values: dict[str, float], cfg=None) -> float:
    if not values.get("has_mandatory_requirements"):
        return 1.0
    coverage = min(1.0, max(0.0, values.get("mandatory_coverage", 0.0)))
    floor = .15 if cfg is None else cfg.ranking_gate_floor
    exponent = 1.5 if cfg is None else cfg.ranking_gate_exponent
    return floor + (1.0 - floor) * coverage ** exponent


def linear_score(features: RankingFeatures, cfg=None) -> float:
    if cfg is None:
        from ..config import DEFAULT
        cfg = DEFAULT
    values = features.values
    weights = (
        cfg.ranking_legacy_weights
        if cfg.ranking_preset == "legacy" else cfg.ranking_improved_weights
    )
    evidence = sum(weight * values.get(name, 0.0) for name, weight in weights)
    score = mandatory_gate(values, cfg) * evidence
    score += sum(weight * values.get(name, 0.0) for name, weight in cfg.ranking_bonuses)
    score -= sum(weight * values.get(name, 0.0) for name, weight in cfg.ranking_penalties)
    if values.get("negation_violation") or values.get("exclusion_violation"):
        score -= cfg.ranking_hard_violation_penalty
    return score
