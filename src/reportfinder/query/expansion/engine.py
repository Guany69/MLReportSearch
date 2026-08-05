"""Orchestrates deterministic corpus-aware query expansion."""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, replace
from typing import cast

from ...retrieval.forms import build_forms
from ..normalize import Token, tokenize
from ..trie import PhraseTrie
from . import acronyms
from .hierarchy import ConceptHierarchy
from .requirement import apply_requirements
from .rules import Rule, load_lexicon
from .types import (
    ConceptKind,
    ExpansionAmbiguity,
    ExpansionConcept,
    ExpansionResult,
    Origin,
    Requirement,
    Span,
    TypoCorrection,
)
from .typos import CorrectionLexicon
from .vocabulary import CorpusVocabulary, normalize_phrase


@dataclass(frozen=True)
class VocabularyHit:
    """A trie match against corpus vocabulary rather than a phrase rule."""

    origin: str
    kind: ConceptKind
    canonical: str


_ORIGIN_TRUST = {
    "literal": 5, "phrase_rule": 4, "acronym": 3, "morphological": 2, "typo": 1,
}
_REQ_ORDER = {Requirement.MENTIONED: 0, Requirement.PREFERRED: 1, Requirement.REQUIRED: 2}


class ExpansionEngine:
    def __init__(self, frame, *, use_typo_correction: bool = True,
                 typo_max_distance: int = 2, max_typo_corrections: int = 2,
                 max_mandatory_fields: int = 4, expansion_weight: float = .85,
                 subquery_weight: float = .75) -> None:
        self.vocabulary = CorpusVocabulary.from_frame(frame)
        self.lexicon = load_lexicon(self.vocabulary)
        self.hierarchy = ConceptHierarchy(self.vocabulary.normalized_field_to_canonical.values())
        safe_phrase_tokens = {
            token
            for rule in self.lexicon.rules
            for phrase in rule.phrases
            for token in normalize_phrase(phrase).split()
        }
        self.corrections = CorrectionLexicon(self.vocabulary, safe_phrase_tokens)
        self.use_typo_correction = use_typo_correction
        self.typo_max_distance = typo_max_distance
        self.max_typo_corrections = max_typo_corrections
        self.max_mandatory_fields = max_mandatory_fields
        self.expansion_weight = expansion_weight
        self.subquery_weight = subquery_weight
        self.trie = PhraseTrie()
        for kind, values in (
            ("field", self.vocabulary.normalized_field_to_canonical),
            ("category", self.vocabulary.normalized_category_to_canonical),
            ("data_source", self.vocabulary.normalized_source_to_canonical),
            ("business_object", self.vocabulary.normalized_object_to_canonical),
        ):
            for phrase, canonical in values.items():
                self.trie.add(tuple(phrase.split()), ("literal", kind, canonical))
        for rule in self.lexicon.rules:
            for phrase in rule.phrases:
                normalized = normalize_phrase(phrase)
                if normalized:
                    self.trie.add(tuple(normalized.split()), rule)

    @staticmethod
    def _vocabulary_hit(payload: object) -> VocabularyHit:
        """Narrow a trie payload that is not a Rule.

        The trie stores `object` because it holds two payload shapes; this states
        the other one so the unpacking sites are checked rather than assumed.
        """
        assert isinstance(payload, tuple) and len(payload) == 3, payload
        origin, kind, canonical = payload
        return VocabularyHit(str(origin), cast(ConceptKind, kind), str(canonical))

    @staticmethod
    def _span(query: str, tokens, start: int, end: int) -> Span:
        return Span(start, end, tokens[start].char_start, tokens[end - 1].char_end,
                    query[tokens[start].char_start:tokens[end - 1].char_end])

    @staticmethod
    def _context_satisfied(rule: Rule, tokens, start: int, end: int) -> bool:
        window = rule.window or len(tokens)
        lo, hi = max(0, start - window), min(len(tokens), end + window)
        text = " ".join(token.text for token in tokens[lo:hi])
        if rule.requires_any and not any(term in text for term in rule.requires_any):
            return False
        return not (
            rule.requires_none and any(term in text for term in rule.requires_none)
        )

    def _initial_matches(self, query: str, tokens):
        concepts = []
        rule_hits: dict[tuple[int, int], list[tuple[Rule, Span]]] = {}
        for match in self.trie.find_all(tokens):
            span = self._span(query, tokens, match.tok_start, match.tok_end)
            if isinstance(match.payload, Rule):
                if self._context_satisfied(match.payload, tokens, match.tok_start, match.tok_end):
                    rule_hits.setdefault((match.tok_start, match.tok_end), []).append((match.payload, span))
                continue
            hit = self._vocabulary_hit(match.payload)
            kind, canonical = hit.kind, hit.canonical
            origin: Origin = "morphological" if match.morphological else "literal"
            concepts.append(ExpansionConcept(
                canonical, kind, .95 if match.morphological else 1.0, origin,
                span.text, span,
            ))
        suppressed = []
        for hits in rule_hits.values():
            priority = max(rule.priority for rule, _ in hits)
            for rule, span in hits:
                if rule.priority < priority:
                    for emission in rule.emits:
                        suppressed.append(ExpansionConcept(
                            emission.canonical, emission.kind, emission.confidence,
                            "phrase_rule", span.text, span, rule_ids=(rule.id,),
                            suppressed_by=tuple(winner.id for winner, _ in hits if winner.priority == priority),
                        ))
                    continue
                for emission in rule.emits:
                    concepts.append(ExpansionConcept(
                        emission.canonical, emission.kind, emission.confidence,
                        "phrase_rule", span.text, span, rule_ids=(rule.id,),
                    ))
        return concepts, suppressed

    def _suppress(self, concepts):
        suppressed_ids = set()
        reasons: dict[int, tuple[str, ...]] = {}
        for i, generic in enumerate(concepts):
            for j, specific in enumerate(concepts):
                if i == j:
                    continue
                if (self.hierarchy.specializes(specific.canonical, generic.canonical)
                        and specific.span.contains(generic.span)):
                    suppressed_ids.add(i)
                    reasons[i] = (specific.canonical,)
                    break
        # Semantic, context-gated overrides from the winning rules.
        suppress_targets: dict[str, list[str]] = {}
        rules_by_id = {rule.id: rule for rule in self.lexicon.rules}
        for concept in concepts:
            for rule_id in concept.rule_ids:
                for target in rules_by_id[rule_id].suppresses:
                    suppress_targets.setdefault(target.casefold(), []).append(rule_id)
        for i, concept in enumerate(concepts):
            if concept.canonical.casefold() in suppress_targets:
                suppressed_ids.add(i)
                reasons[i] = tuple(suppress_targets[concept.canonical.casefold()])
        surviving = [concept for i, concept in enumerate(concepts) if i not in suppressed_ids]
        suppressed = [
            replace(concept, suppressed_by=reasons.get(i, ()))
            for i, concept in enumerate(concepts) if i in suppressed_ids
        ]
        return surviving, suppressed

    def _typo_and_acronyms(self, query: str, tokens, suppressed_canonicals):
        concepts = []
        typos = []
        ambiguities: list[ExpansionAmbiguity] = []
        corrected_tokens = list(tokens)
        meaningful = [token for token in tokens if len(token.text) > 2]
        budget = min(self.max_typo_corrections, max(1, math.ceil(len(meaningful) * .4)))
        corrections_used = 0
        for index, token in enumerate(tokens):
            canonical, kind, ambiguity = acronyms.resolve(token, self.vocabulary, tokens)
            if ambiguity:
                ambiguities.append(ambiguity)
            elif canonical and canonical.casefold() not in suppressed_canonicals:
                span = self._span(query, tokens, index, index + 1)
                concepts.append(ExpansionConcept(
                    canonical, kind, .9, "acronym", span.text, span,
                ))
            if not self.use_typo_correction or corrections_used >= budget:
                continue
            result = self.corrections.correct(token.text, self.typo_max_distance)
            if result is None:
                continue
            corrected, distance, runner, runner_distance, confidence, origin = result
            corrected_tokens[index] = Token(corrected, token.surface, token.char_start, token.char_end)
            span = self._span(query, tokens, index, index + 1)
            if origin == "typo":
                typos.append(TypoCorrection(
                    token.surface, corrected, distance, runner, runner_distance,
                    confidence, span,
                ))
                corrections_used += 1
        # Re-probe the complete phrase trie so corrected multi-token expressions work.
        for match in self.trie.find_all(corrected_tokens):
            if isinstance(match.payload, Rule):
                if not self._context_satisfied(
                    match.payload, corrected_tokens, match.tok_start, match.tok_end
                ):
                    continue
                payloads = [(em.canonical, em.kind, em.confidence, match.payload.id)
                            for em in match.payload.emits]
            else:
                hit = self._vocabulary_hit(match.payload)
                payloads = [(hit.canonical, hit.kind, .80, "")]
            if not any(
                corrected_tokens[i].text != tokens[i].text
                for i in range(match.tok_start, match.tok_end)
            ):
                continue
            span = self._span(query, tokens, match.tok_start, match.tok_end)
            for canonical, kind, confidence, rule_id in payloads:
                if canonical.casefold() in suppressed_canonicals:
                    continue
                concepts.append(ExpansionConcept(
                    canonical, kind, confidence, "typo", span.text, span,
                    rule_ids=(rule_id,) if rule_id else (),
                ))
        return concepts, typos, ambiguities

    @staticmethod
    def _merge(concepts):
        grouped = {}
        for concept in concepts:
            key = (concept.kind, concept.canonical.casefold())
            grouped.setdefault(key, []).append(concept)
        merged = []
        for group in grouped.values():
            representative = sorted(
                group,
                key=lambda c: (-c.confidence, c.span.tok_start,
                               -(c.span.tok_end - c.span.tok_start),
                               -_ORIGIN_TRUST[c.origin]),
            )[0]
            requirement = max((c.requirement for c in group), key=_REQ_ORDER.get)
            origin = max((c.origin for c in group), key=_ORIGIN_TRUST.get)
            evidences = []
            for concept in sorted(group, key=lambda c: c.span.tok_start):
                if concept.evidence not in evidences:
                    evidences.append(concept.evidence)
            merged.append(replace(
                representative, confidence=max(c.confidence for c in group),
                requirement=requirement, origin=origin,
                evidence="; ".join(evidences[:3]),
                rule_ids=tuple(sorted({rule for c in group for rule in c.rule_ids})),
            ))
        return tuple(sorted(merged, key=lambda c: (-c.confidence, c.kind, c.canonical.casefold())))

    @staticmethod
    def _subqueries(query: str, concepts) -> tuple[str, ...]:
        parts = [part.strip(" ,") for part in re.split(
            r"\s*(?:;|\band then\b|\bversus\b|\bvs\.?\b)\s*", query, flags=re.I
        ) if part.strip(" ,")]
        if len(parts) == 1:
            # Canonical concept phrases are safe focused retrieval forms.
            parts.extend(concept.canonical for concept in concepts if concept.origin != "literal")
        return tuple(dict.fromkeys(parts))

    def expand(self, query: str) -> ExpansionResult:
        tokens = tokenize(query)
        normalized = " ".join(token.text for token in tokens)
        concepts, priority_suppressed = self._initial_matches(query, tokens)
        concepts, structural_suppressed = self._suppress(concepts)
        suppressed = priority_suppressed + structural_suppressed
        suppressed_canonicals = {c.canonical.casefold() for c in suppressed}
        later, typos, ambiguities = self._typo_and_acronyms(
            query, tokens, suppressed_canonicals,
        )
        concepts.extend(later)
        # Later typo/acronym matches participate in the same hierarchy and
        # semantic suppression pass. This prevents a corrected generic field
        # from being reintroduced after a specific rule suppressed it.
        concepts, later_suppressed = self._suppress(concepts)
        suppressed.extend(later_suppressed)
        concepts = apply_requirements(concepts, tokens, self.max_mandatory_fields)
        merged = self._merge(concepts)
        corrections_by_index = {
            item.span.tok_start: item.corrected for item in typos
        }
        expanded_base = " ".join(
            corrections_by_index.get(index, token.surface)
            for index, token in enumerate(tokens)
        )
        appended = [
            c.canonical for c in merged
            if c.confidence >= .70 and normalize_phrase(c.canonical) not in normalize_phrase(expanded_base)
        ]
        expanded = expanded_base if typos else query
        if appended:
            expanded += " " + " ".join(dict.fromkeys(appended))
        subqueries = self._subqueries(query, merged)
        forms = build_forms(
            query, expanded, subqueries,
            expansion_weight=self.expansion_weight,
            subquery_weight=self.subquery_weight,
        )
        return ExpansionResult(
            query, normalized, tokens, merged, tuple(suppressed), tuple(typos),
            tuple(ambiguities), forms, self.lexicon.hash, subqueries, expanded,
        )
