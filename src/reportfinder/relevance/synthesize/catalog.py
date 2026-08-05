"""Corpus facts the archetypes build queries *from*.

The distinction that matters: v1's labels were produced by *matching* generated
queries back against report text, which is the same operation BM25F performs. A
label made that way is a statement about lexical overlap, so measuring lexical
retrieval against it is circular -- and the measured result showed exactly that
(BM25F alone reached 0.864 union recall against the full union's 0.872).

Everything here instead reads structure: which sibling uniquely carries a field,
which two families are confusable, which field pairs no single report carries.
A query built from those facts has a correct answer by construction, and no
retriever was consulted to decide it.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass

from ...corpus import CorpusModel, ReportFamily, ReportInstance

# Facets a query can name to pick one sibling out of a family. Ordered by how
# naturally they appear in a request: a user says "the matrix one" or "from All
# Active and Terminated" far more often than they name a prompt.
DISTINGUISHING_KINDS = ("field", "data_source", "report_type", "prompt")


@dataclass(frozen=True)
class Distinguisher:
    """A value that exactly one instance of a family carries."""

    kind: str
    value: str
    carrier_instance_id: str

    @property
    def phrase(self) -> str:
        """How a person would say it."""
        if self.kind == "field":
            return f"including {self.value}"
        if self.kind == "data_source":
            return f"from {self.value}"
        if self.kind == "report_type":
            return f"as a {self.value} report"
        return f"prompted for {self.value}"


def _values(instance: ReportInstance, kind: str) -> tuple[str, ...]:
    if kind == "field":
        return instance.fields
    if kind == "prompt":
        return instance.prompts
    if kind == "data_source":
        return (instance.data_source,) if instance.data_source else ()
    if kind == "report_type":
        return (instance.report_type,) if instance.report_type else ()
    raise ValueError(f"unknown distinguisher kind: {kind}")


def multi_instance_families(corpus: CorpusModel) -> list[ReportFamily]:
    """Families with more than one instance, in deterministic id order."""
    return [
        family for _, family in sorted(corpus.families.items())
        if len(family.instance_ids) > 1
    ]


def sibling_distinguishers(
    family: ReportFamily, corpus: CorpusModel,
) -> list[Distinguisher]:
    """Values carried by exactly one instance of this family.

    Exactly one, not merely "not all": a value two of five siblings carry cannot
    identify a single correct answer, and grading one of them 2 and the other 1
    would be inventing a preference the catalog does not express.
    """
    instances = [corpus.instance(i) for i in family.instance_ids]
    out: list[Distinguisher] = []
    for kind in DISTINGUISHING_KINDS:
        counts: Counter[str] = Counter()
        for instance in instances:
            counts.update(set(_values(instance, kind)))
        for value, count in sorted(counts.items()):
            if count != 1:
                continue
            carrier = next(
                i for i in instances if value in set(_values(i, kind))
            )
            out.append(Distinguisher(kind, value, carrier.report_instance_id))
    return out


def field_document_frequency(corpus: CorpusModel) -> dict[str, int]:
    counts: Counter[str] = Counter()
    for instance in corpus.instances:
        counts.update(set(instance.fields))
    return dict(counts)


def rare_fields(corpus: CorpusModel, *, max_df: int = 3) -> dict[str, list[str]]:
    """Fields carried by at most `max_df` instances, all in one family.

    The single-family constraint is what makes a one-to-three-token query have a
    definite answer: "time to fill" is only a fair short query if every report
    carrying that field belongs to the same family.
    """
    carriers: dict[str, list[str]] = defaultdict(list)
    for instance in corpus.instances:
        for field in set(instance.fields):
            carriers[field].append(instance.report_instance_id)

    out = {}
    for field, ids in sorted(carriers.items()):
        if len(ids) > max_df:
            continue
        families = {corpus.instance(i).family_id for i in ids}
        if len(families) == 1:
            out[field] = sorted(ids)
    return out


def co_absent_field_pairs(
    corpus: CorpusModel, *, limit: int, rng,
) -> list[tuple[str, str]]:
    """Field pairs that exist separately but never together.

    This is what makes an unanswerable request *in-domain*: both terms are real
    catalog vocabulary, so retrieval will happily return candidates for either,
    and only the "no single report carries all requested fields" gate can catch
    it. A pair invented from nonsense words would be caught much earlier and
    would test nothing.
    """
    frequency = field_document_frequency(corpus)
    # Common fields only: a pair of rare fields is trivially co-absent and says
    # nothing about whether the gate works on realistic input.
    common = [f for f, n in sorted(frequency.items()) if n >= 20]
    if len(common) < 2:
        return []

    holders = {
        field: {
            i.report_instance_id for i in corpus.instances if field in set(i.fields)
        }
        for field in common
    }

    pairs: list[tuple[str, str]] = []
    seen: set[tuple[str, str]] = set()
    attempts = 0
    while len(pairs) < limit and attempts < limit * 200:
        attempts += 1
        left = common[int(rng.integers(len(common)))]
        right = common[int(rng.integers(len(common)))]
        if left == right:
            continue
        key = (left, right) if left < right else (right, left)
        if key in seen:
            continue
        seen.add(key)
        if not holders[left] & holders[right]:
            pairs.append(key)
    return pairs


def confusable_family_pairs(
    corpus: CorpusModel, *, limit: int,
) -> list[tuple[ReportFamily, ReportFamily, str]]:
    """Family pairs a user could plausibly mean either of.

    Same category and a shared title head-noun, differing on exactly one facet.
    That difference is what a clarifying question can be *about* -- a pair that
    differs on nothing gives the decision layer nothing real to ask.
    """
    by_category: dict[str, list[ReportFamily]] = defaultdict(list)
    for _, family in sorted(corpus.families.items()):
        first = corpus.instance(family.instance_ids[0])
        if first.category:
            by_category[first.category].append(family)

    out: list[tuple[ReportFamily, ReportFamily, str]] = []
    for _, families in sorted(by_category.items()):
        by_head: dict[str, list[ReportFamily]] = defaultdict(list)
        for family in families:
            tokens = [t for t in family.normalized_title.split() if len(t) > 3]
            if tokens:
                by_head[tokens[0]].append(family)

        for _, group in sorted(by_head.items()):
            for i, left in enumerate(group):
                for right in group[i + 1:]:
                    facet = _differing_facet(left, right, corpus)
                    if facet is not None:
                        out.append((left, right, facet))
                    if len(out) >= limit:
                        return out
    return out


def _differing_facet(
    left: ReportFamily, right: ReportFamily, corpus: CorpusModel,
) -> str | None:
    a, b = corpus.instance(left.instance_ids[0]), corpus.instance(right.instance_ids[0])
    for facet in ("data_source", "report_type"):
        if getattr(a, facet) and getattr(b, facet) and getattr(a, facet) != getattr(b, facet):
            return facet
    return None
