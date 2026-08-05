"""Turning catalog facts into the words a person would actually type.

The important function is `lexicon_phrasings`. The shipped lexicon maps business
phrases to canonical field names -- "people who left" emits `Termination Date`
and `Termination Reason`. Reading it *backwards* gives, for any field, the
phrasings a user might reach for. That is the anti-circularity mechanism: the
query text and the graded answer come from opposite ends of a mapping that
already existed, rather than from matching one against the other.

The noise functions (acronyms, typos) are deliberately conservative. A corruption
that happens to spell another real vocabulary word is not a misspelling test, it
is a different query with a different right answer, so those are rejected.
"""

from __future__ import annotations

import re
from collections import defaultdict

# Words that carry no retrieval signal. Kept small and explicit: an aggressive
# stoplist would let a vague query smuggle in content tokens under the guise of
# being "just" a stopword, defeating the disjointness constraint it must satisfy.
STOPWORDS = frozenset({
    "a", "an", "and", "are", "as", "at", "be", "but", "by", "can", "do", "does",
    "for", "from", "get", "give", "how", "i", "in", "into", "is", "it", "me",
    "much", "my", "need", "of", "on", "or", "our", "out", "over", "per", "show",
    "so", "that", "the", "them", "then", "to", "up", "us", "want", "was", "we",
    "what", "when", "where", "which", "who", "why", "will", "with", "would",
    "you", "your", "list", "find", "see", "some", "any", "all", "more", "most",
    "each", "every", "please", "help", "understand", "know", "tell", "about",
})

_WORD = re.compile(r"[a-z0-9]+")

# Business-outcome frames. None of these contribute a content token, which is
# what lets a vague query stay token-disjoint from its target's title.
OUTCOME_TEMPLATES = (
    "why are we {phrase}",
    "help me understand {phrase}",
    "what is going on with {phrase}",
    "i need to get a handle on {phrase}",
    "can you show me something about {phrase}",
    "we are worried about {phrase}",
)

# Common workplace abbreviations. Applied to a finished query so the underlying
# intent is unchanged and only the surface is degraded.
ACRONYMS = {
    "employee": "EE", "employees": "EEs", "compensation": "comp",
    "organization": "org", "organizations": "orgs", "manager": "mgr",
    "management": "mgmt", "department": "dept", "headcount": "HC",
    "termination": "term", "terminations": "terms", "performance": "perf",
    "recruiting": "rec", "requisition": "req", "requisitions": "reqs",
    "year to date": "YTD", "full time equivalent": "FTE",
}

# Reserved out-of-domain vocabulary. `query/expansion/rules.py` refuses to load a
# lexicon that grounds any of these, and none appears anywhere in the estate, so
# a query built from them is unanswerable by construction rather than by
# threshold.
RESERVED_VOCABULARY = ("quantum", "submarine", "telemetry")

NO_ANSWER_TEMPLATES = (
    "{a} {b} report for the {c} division",
    "show me the {a} {b} dashboard",
    "which report covers {a} {c} readings",
)


def content_tokens(text: str, *, stopwords=STOPWORDS) -> set[str]:
    """Tokens that carry meaning, for the disjointness constraint."""
    return {
        token for token in _WORD.findall(text.casefold())
        if token not in stopwords and len(token) > 2
    }


def lexicon_phrasings(lexicon) -> dict[str, list[tuple[str, str]]]:
    """canonical field -> [(rule id, business phrase)], inverted from the lexicon."""
    out: dict[str, list[tuple[str, str]]] = defaultdict(list)
    for rule in lexicon.rules:
        for emission in rule.emits:
            for phrase in rule.phrases:
                out[emission.canonical].append((rule.id, phrase))
    return {key: sorted(set(value)) for key, value in sorted(out.items())}


def rule_emissions(lexicon) -> dict[str, tuple[str, ...]]:
    """rule id -> the canonicals it emits, for the vague-query construction."""
    return {
        rule.id: tuple(sorted({e.canonical for e in rule.emits}))
        for rule in lexicon.rules
    }


def apply_acronyms(query: str, rng) -> tuple[str, list[str]]:
    """Abbreviate one or two terms, as an internal user would."""
    applied: list[str] = []
    text = query
    for long, short in sorted(ACRONYMS.items()):
        if len(applied) >= 2:
            break
        pattern = re.compile(rf"\b{re.escape(long)}\b", re.IGNORECASE)
        if pattern.search(text) and rng.random() < 0.8:
            text = pattern.sub(short, text, count=1)
            applied.append(f"{long}->{short}")
    return text, applied


def apply_typo(query: str, rng, *, vocabulary: set[str]) -> tuple[str, str]:
    """Corrupt one content token by a single edit.

    Returns `(query, "")` when no safe corruption exists. A corruption that
    lands on another real vocabulary token is rejected: that is not a
    misspelling, it is a different question.
    """
    candidates = [
        token for token in _WORD.findall(query)
        if len(token) >= 5 and token.casefold() not in STOPWORDS
    ]
    if not candidates:
        return query, ""

    for token in sorted(set(candidates)):
        for corrupted in _single_edits(token, rng):
            if corrupted.casefold() in vocabulary or corrupted == token:
                continue
            pattern = re.compile(rf"\b{re.escape(token)}\b")
            return pattern.sub(corrupted, query, count=1), f"{token}->{corrupted}"
    return query, ""


def _single_edits(token: str, rng) -> list[str]:
    """Transposition and doubling: the two typos people actually make."""
    lowered = token.casefold()
    out = []
    positions = list(range(1, len(lowered) - 1))
    rng.shuffle(positions)
    for i in positions[:4]:
        out.append(lowered[:i] + lowered[i + 1] + lowered[i] + lowered[i + 2:])
        out.append(lowered[:i] + lowered[i] + lowered[i:])
    return out


def corpus_vocabulary_tokens(corpus) -> set[str]:
    """Every token that appears in any searchable catalog value."""
    out: set[str] = set()
    for instance in corpus.instances:
        for text in (
            instance.title, instance.category, instance.data_source,
            instance.report_type, instance.tags, *instance.fields, *instance.prompts,
        ):
            out |= {t for t in _WORD.findall(str(text).casefold())}
    return out


def humanize(value: str) -> str:
    """A field name as a person would say it, not as the catalog spells it."""
    return re.sub(r"\s+", " ", value.replace("_", " ")).strip().casefold()
