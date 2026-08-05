"""Layered, collision-aware business acronym resolution."""

from __future__ import annotations

from .types import ExpansionAmbiguity

RESERVED = {
    "TA": ("Time Away", "topic"),
    "HC": ("Headcount", "field"),
    "FTE": ("FTE", "topic"),
    "OOO": ("Out of Office", "topic"),
    "PTO": ("Time Off", "topic"),
    "LOA": ("Leave of Absence", "topic"),
    "FMLA": ("Leave of Absence", "topic"),
    "YTD": ("Year to Date", "topic"),
    "EEO": ("Equal Employment Opportunity", "topic"),
    "HRIS": ("HR Information System", "topic"),
    "LMS": ("Learning Management System", "topic"),
    "ATS": ("Applicant Tracking System", "topic"),
    "BU": ("Business Unit", "topic"),
    "CC": ("Cost Center", "topic"),
}
_CASE_INSENSITIVE = {"HC", "FTE", "OOO", "PTO"}
_CONTEXT = {
    "TA": {
        "recruiting": ({"partner", "recruiter", "talent", "acquisition"}, ("Recruiter", "field")),
        "time_away": ({"away", "leave", "absence", "time", "pto"}, ("Time Away", "topic")),
    },
}


def resolve(token, vocabulary, context_tokens=()):
    surface = token.surface
    upper = surface.upper()
    context = {getattr(item, "text", str(item)).casefold() for item in context_tokens}
    if upper == "TA":
        for _, (required, resolved) in _CONTEXT["TA"].items():
            if required & context:
                canonical, kind = resolved
                if kind == "field":
                    canonical = vocabulary.canonical("field", canonical) or canonical
                return canonical, kind, None
        if surface != upper:
            return None, None, None
    if upper in RESERVED:
        if surface != upper and upper not in _CASE_INSENSITIVE:
            return None, None, None
        canonical, kind = RESERVED[upper]
        if kind == "field":
            canonical = vocabulary.canonical("field", canonical) or canonical
        return canonical, kind, None
    if surface != upper and not any(char.isdigit() for char in surface):
        return None, None, None
    values = vocabulary.acronym_to_canonical_values.get(upper, ())
    if len(values) == 1:
        canonical = values[0]
        kind = "field" if canonical.casefold() in {
            value.casefold() for value in vocabulary.normalized_field_to_canonical.values()
        } else "topic"
        return canonical, kind, None
    if len(values) > 1:
        return None, None, ExpansionAmbiguity(
            surface, values, "acronym maps to multiple corpus concepts",
        )
    return None, None, None
