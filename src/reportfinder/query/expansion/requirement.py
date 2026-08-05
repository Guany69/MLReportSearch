"""Span-local required/preferred/mentioned requirement inference."""

from __future__ import annotations

from dataclasses import replace

from .types import ExpansionConcept, Requirement

_REQUIRED = {"including", "include", "containing", "contains", "require", "required"}
_PREFERRED = {"by", "per", "where"}
_DOWNGRADE = {
    "about", "like", "maybe", "ideally", "prefer", "preferably", "optionally", "any",
}
_BOUNDARIES = {"but", "then", "also"}


def determine(concept: ExpansionConcept, tokens) -> tuple[Requirement, str]:
    start, end = concept.span.tok_start, concept.span.tok_end
    left_start = max(0, start - 8)
    left = [token.text for token in tokens[left_start:start]]
    right = [token.text for token in tokens[end:min(len(tokens), end + 3)]]
    for boundary in _BOUNDARIES:
        if boundary in left:
            left = left[left.index(boundary) + 1:]
        if boundary in right:
            right = right[:right.index(boundary)]
    nearby = left[-4:] + right[:2]
    phrase = " ".join(left[-4:])
    if any(marker in nearby for marker in _DOWNGRADE) or any(
        marker in phrase for marker in ("related to", "such as", "similar to", "if possible")
    ):
        return Requirement.MENTIONED, "downgrade marker"
    # An exact canonical field mention is unambiguous enough to act as a hard
    # constraint. Inferred phrases need both high confidence and explicit
    # request syntax; generic "show/with/need" wording never upgrades every
    # nearby concept.
    if concept.kind == "field" and concept.origin == "literal" and concept.confidence == 1.0:
        tier, evidence = Requirement.REQUIRED, "literal canonical field"
    else:
        required_marker = next((marker for marker in reversed(left) if marker in _REQUIRED), None)
        immediate_request = bool(
            left and left[-1] in {"with", "show", "showing", "need", "needs"}
        )
        explicitly_requested = bool(
            required_marker or phrase.endswith("must have") or immediate_request
        )
        phrase_required = (
            concept.kind == "field"
            and concept.origin == "phrase_rule"
            and concept.confidence >= .88
            and explicitly_requested
        )
        if phrase_required:
            tier = Requirement.REQUIRED
            evidence = required_marker or ("explicit request" if immediate_request else "must have")
        elif any(marker in nearby for marker in _PREFERRED) or any(
            phrase.endswith(marker) for marker in ("grouped by", "broken down by", "split by", "for each", "filtered by")
        ):
            tier = Requirement.PREFERRED
            evidence = next((m for m in nearby if m in _PREFERRED), "grouping marker")
        elif start > 0 and tokens[start - 1].text == "showing":
            tier, evidence = Requirement.REQUIRED, "showing"
        else:
            tier, evidence = Requirement.MENTIONED, "bare mention"

    if (
        concept.origin in {"typo", "morphological"} or concept.confidence < .80
    ) and tier is Requirement.REQUIRED:
        tier = Requirement.PREFERRED
        evidence += " (softened)"
    return tier, evidence


def apply_requirements(concepts, tokens, max_mandatory: int = 4):
    updated = []
    for concept in concepts:
        tier, _ = determine(concept, tokens)
        updated.append(replace(concept, requirement=tier))
    required = [c for c in updated if c.requirement is Requirement.REQUIRED]
    if len(required) > max_mandatory:
        demote = {
            id(c) for c in sorted(
                required, key=lambda c: (c.confidence, -c.span.tok_start, c.canonical.casefold())
            )[:len(required) - max_mandatory]
        }
        updated = [
            replace(c, requirement=Requirement.PREFERRED) if id(c) in demote else c
            for c in updated
        ]
    return updated
