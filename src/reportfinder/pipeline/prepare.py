"""Query preparation: several representations, one of which is never touched.

The architecture's central rule is that no single rewrite may decide what is
retrievable. So preparation *adds* representations rather than replacing the query:
Q0 is the user's exact words and is always searched, always passed to the
cross-encoder, and never edited. Everything else is an additional lens.

* **Q0 raw** -- verbatim. The only representation guaranteed to be correct.
* **Q1 normalized** -- unicode, whitespace and case folded for matching. Codes,
  dates, punctuation and negation survive.
* **Q2 alias-expanded** -- deterministic catalog aliases, acronyms and typo
  corrections from the existing expansion engine. Additive: it appends canonical
  terms rather than substituting the user's.
* **Q3 business intent** -- outcome language, for the purpose view and prototypes.
* **Q4 schema-oriented** -- field and prompt language, for the schema view.
* **Q5 alternate** -- at most one bounded reinterpretation, and only when a genuine
  ambiguity was detected. Provenance-marked, never authoritative, never a filter.

Facets use explicit `NONE` / `UNKNOWN` / `MULTIPLE` states. An unstated facet is
recorded as unstated; it is never guessed, because a guessed facet becomes a filter
that removes the right answer.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

from ..query.intent import QueryIntent

# Deterministic vocabulary for facets the expansion engine does not itself model.
# Substring matching on the normalized query -- no inference beyond what is written.
_CURRENT_TERMS = ("current", "today", "now", "as of today", "active", "present")
_HISTORICAL_TERMS = ("historical", "history", "trend", "over time", "previous",
                     "prior", "last year", "past", "year over year")
_DETAIL_TERMS = ("detail", "list", "line item", "each", "per worker", "individual",
                 "roster", "who", "names")
_AGGREGATE_TERMS = ("total", "count", "summary", "aggregate", "rate", "average",
                    "how many", "headcount", "sum")
_NEGATION_TERMS = ("not", "without", "exclude", "excluding", "except", "no ")


class FacetState(str, Enum):
    """A facet the query did not determine. Never inferred into a value."""

    NONE = "NONE"
    UNKNOWN = "UNKNOWN"
    MULTIPLE = "MULTIPLE"


@dataclass(frozen=True)
class QueryVariant:
    """One representation of the request."""

    key: str
    text: str
    kind: str
    provenance: str
    # Relative trust, used only for tie-breaks in fusion. Q0 is always highest.
    weight: float = 1.0

    @property
    def is_raw(self) -> bool:
        return self.key == "Q0"


@dataclass(frozen=True)
class QueryPlan:
    """Everything the generators may search, plus what was understood."""

    raw_query: str
    variants: tuple[QueryVariant, ...]
    facets: dict[str, object] = field(default_factory=dict)
    intent: QueryIntent | None = None
    clarification_context: tuple[str, ...] = ()

    def variant(self, key: str) -> QueryVariant | None:
        return next((v for v in self.variants if v.key == key), None)

    def variants_for(self, keys: tuple[str, ...]) -> tuple[QueryVariant, ...]:
        """The requested variants that exist, always including the raw query.

        Q0 is unconditional: a generator may prefer another lens, but it must never
        stop searching what the user actually typed.
        """
        selected = [v for v in self.variants if v.key in keys]
        if not any(v.is_raw for v in selected):
            raw = self.variant("Q0")
            if raw is not None:
                selected.insert(0, raw)
        return tuple(selected)

    @property
    def alternate(self) -> QueryVariant | None:
        return self.variant("Q5")

    def facet(self, name: str) -> object:
        return self.facets.get(name, FacetState.UNKNOWN)

    def stated_facets(self) -> dict[str, object]:
        """Only facets the query actually determined."""
        return {
            name: value
            for name, value in self.facets.items()
            if not isinstance(value, FacetState)
        }

    def telemetry(self) -> dict[str, object]:
        return {
            "query_variant_count": len(self.variants),
            "query_variants": [v.key for v in self.variants],
            "stated_facets": sorted(self.stated_facets()),
            "has_alternate_interpretation": self.alternate is not None,
        }


def _contains(haystack: str, terms) -> bool:
    return any(term in haystack for term in terms)


def _orientation(normalized: str) -> object:
    current = _contains(normalized, _CURRENT_TERMS)
    historical = _contains(normalized, _HISTORICAL_TERMS)
    if current and historical:
        return FacetState.MULTIPLE
    if current:
        return "current"
    if historical:
        return "historical"
    return FacetState.UNKNOWN


def _granularity(normalized: str) -> object:
    detail = _contains(normalized, _DETAIL_TERMS)
    aggregate = _contains(normalized, _AGGREGATE_TERMS)
    if detail and aggregate:
        return FacetState.MULTIPLE
    if detail:
        return "detail"
    if aggregate:
        return "aggregate"
    return FacetState.UNKNOWN


def _concept_facet(values: list[str]) -> object:
    if not values:
        return FacetState.NONE
    return values if len(values) == 1 else values


def build_facets(intent: QueryIntent) -> dict[str, object]:
    """Facets the query determined, and explicit states for those it did not."""
    normalized = (intent.normalized_query or intent.raw_query).casefold()
    return {
        "subject": _concept_facet([c.value for c in intent.objectives]),
        "measure": _concept_facet([c.value for c in intent.fields if c.mandatory]),
        "dimension": _concept_facet([c.value for c in intent.business_objects]),
        "population": _concept_facet([c.value for c in intent.categories]),
        "data_source": _concept_facet([c.value for c in intent.data_sources]),
        "time_orientation": _orientation(normalized),
        "granularity": _granularity(normalized),
        "comparison": intent.comparison or FacetState.NONE,
        "time_freshness": intent.time_freshness or FacetState.UNKNOWN,
        # Recorded so downstream never drops a negated term as noise.
        "negation": _contains(normalized, _NEGATION_TERMS),
        # The catalog exposes report type and interface, but a query almost never
        # states them. UNKNOWN rather than a default.
        "report_type": FacetState.UNKNOWN,
        "interface_requirement": FacetState.UNKNOWN,
    }


def _business_intent_text(intent: QueryIntent) -> str:
    """Outcome-flavoured phrasing, for the purpose view and prototypes."""
    parts = [intent.raw_query]
    parts.extend(c.value for c in intent.objectives)
    parts.extend(c.value for c in intent.categories)
    return " ".join(dict.fromkeys(p for p in parts if p))


def _schema_text(intent: QueryIntent) -> str:
    """Field and prompt phrasing, for the schema view."""
    parts = [intent.raw_query]
    parts.extend(c.value for c in intent.fields)
    parts.extend(c.value for c in intent.data_sources)
    parts.extend(c.value for c in intent.business_objects)
    return " ".join(dict.fromkeys(p for p in parts if p))


def _alternate_text(intent: QueryIntent) -> str | None:
    """One bounded reinterpretation, only when the parser found real ambiguity.

    Built from the *suppressed* candidates the expansion engine set aside. If it had
    no second reading, there is no alternate -- one is not manufactured.
    """
    expansion = intent.expansion
    if expansion is None or not expansion.ambiguities:
        return None
    candidates: list[str] = []
    for ambiguity in expansion.ambiguities:
        candidates.extend(ambiguity.candidates)
    for concept in getattr(expansion, "suppressed", ()):
        candidates.append(concept.canonical)
    unique = list(dict.fromkeys(c for c in candidates if c))
    if not unique:
        return None
    # Bounded: a handful of terms appended to the raw query, never a rewrite.
    return f"{intent.raw_query} {' '.join(unique[:4])}"


def build_query_plan(
    intent: QueryIntent,
    *,
    raw_query: str,
    clarification_context: tuple[str, ...] = (),
    enable_alternate: bool = True,
) -> QueryPlan:
    """Assemble the plan. `raw_query` is stored verbatim and never derived from."""
    variants: list[QueryVariant] = [
        QueryVariant("Q0", raw_query, "raw", "user", 1.0),
    ]

    normalized = intent.normalized_query or raw_query
    if normalized and normalized != raw_query:
        variants.append(
            QueryVariant("Q1", normalized, "normalized", "deterministic", 0.95)
        )

    expanded = intent.expanded_query or ""
    if expanded and expanded not in {raw_query, normalized}:
        variants.append(
            QueryVariant("Q2", expanded, "alias_expanded", "lexicon", 0.85)
        )

    business = _business_intent_text(intent)
    if business and business != raw_query:
        variants.append(
            QueryVariant("Q3", business, "business_intent", "deterministic", 0.80)
        )

    schema = _schema_text(intent)
    if schema and schema not in {raw_query, business}:
        variants.append(
            QueryVariant("Q4", schema, "schema", "deterministic", 0.80)
        )

    if enable_alternate:
        alternate = _alternate_text(intent)
        if alternate and alternate != raw_query:
            variants.append(
                # Marked as an interpretation so nothing treats it as authoritative.
                QueryVariant("Q5", alternate, "alternate", "interpretation", 0.60)
            )

    if clarification_context:
        # A clarification answer is explicit context, not a rewrite of the query.
        context = " ".join(clarification_context)
        variants.append(
            QueryVariant("QC", f"{raw_query} {context}", "clarified", "user", 0.98)
        )

    return QueryPlan(
        raw_query=raw_query,
        variants=tuple(variants),
        facets=build_facets(intent),
        intent=intent,
        clarification_context=tuple(clarification_context),
    )
