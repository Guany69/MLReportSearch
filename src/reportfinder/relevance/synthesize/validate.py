"""Fatal self-checks on the generated set.

A generator that quietly produces a slightly-wrong label is worse than one that
crashes: the labels look usable, the metrics look plausible, and the error is
only discoverable by re-deriving the whole set by hand. So every construction
claim an archetype makes is re-verified here against the corpus, independently
of the code that made it, and any violation aborts the run.

The one that matters most is `_check_siblings`: v1 contained zero queries grading
two instances of one family, which is why instance selection could not be
measured. That is asserted here as a hard floor so it cannot silently regress.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field

from ...corpus import CorpusModel
from . import surface


class ValidationError(RuntimeError):
    """A generated scenario contradicts the corpus it was built from."""


@dataclass
class ValidationReport:
    checks: dict[str, str] = field(default_factory=dict)
    counts: dict[str, int] = field(default_factory=dict)

    def record(self, name: str, detail: str) -> None:
        self.checks[name] = detail

    def as_dict(self) -> dict[str, object]:
        return {"checks": dict(sorted(self.checks.items())),
                "counts": dict(sorted(self.counts.items()))}


def _fail(message: str) -> None:
    raise ValidationError(message)


# How many scenarios must grade two instances of one family. v1 had zero, which
# is exactly why family expansion and instance selection could not be measured;
# this is the floor that stops that regressing silently. A parameter rather than
# a constant only so small fixtures can exercise the other checks.
MIN_SIBLING_GRADED = 600


def validate(
    scenarios,
    corpus: CorpusModel,
    lexicon,
    *,
    splits=None,
    min_sibling_graded: int = MIN_SIBLING_GRADED,
) -> ValidationReport:
    report = ValidationReport()
    report.counts = dict(sorted(Counter(s.archetype for s in scenarios).items()))
    report.counts["total"] = len(scenarios)

    _check_identity(scenarios, corpus, report)
    _check_siblings(scenarios, corpus, report, min_sibling_graded)
    _check_vague(scenarios, corpus, lexicon, report)
    _check_no_answer(scenarios, corpus, report)
    _check_ambiguity(scenarios, corpus, report)
    _check_misspellings(scenarios, corpus, report)
    _check_grade_vocabulary(scenarios, report)
    if splits is not None:
        _check_splits(scenarios, splits, report)
    return report


def _check_identity(scenarios, corpus, report) -> None:
    known = set(corpus.instance_ids)
    seen_ids: set[str] = set()
    for scenario in scenarios:
        if scenario.scenario_id in seen_ids:
            _fail(f"duplicate scenario id {scenario.scenario_id}")
        seen_ids.add(scenario.scenario_id)
        if not scenario.query.strip():
            _fail(f"{scenario.scenario_id} has an empty query")
        unknown = set(scenario.grades) - known
        if unknown:
            _fail(f"{scenario.scenario_id} grades unknown reports: {sorted(unknown)[:5]}")
    report.record("identity", f"{len(scenarios)} scenarios, all report keys resolve")


def _check_siblings(scenarios, corpus, report, minimum: int) -> None:
    sibling = [s for s in scenarios if s.archetype == "sibling_discriminating"]
    for scenario in sibling:
        family = corpus.families[scenario.target_family_id]
        graded = set(scenario.grades) & set(family.instance_ids)
        if graded != set(family.instance_ids):
            _fail(
                f"{scenario.scenario_id} must grade every sibling of "
                f"{family.family_id}; missing "
                f"{sorted(set(family.instance_ids) - graded)}"
            )
        if scenario.grades[scenario.target_instance_id] != 2:
            _fail(f"{scenario.scenario_id} carrier is not graded 2")

        kind = scenario.provenance["distinguishing_kind"]
        value = scenario.provenance["distinguishing_value"]
        from .catalog import _values

        carriers = [
            i for i in family.instance_ids
            if value in set(_values(corpus.instance(i), kind))
        ]
        if carriers != [scenario.target_instance_id]:
            _fail(
                f"{scenario.scenario_id} claims {value!r} is unique to "
                f"{scenario.target_instance_id}, but it is carried by {carriers}"
            )

    # The v1-zero fix, as a hard floor rather than an observation.
    graded_two_siblings = sum(
        1 for s in scenarios if _grades_two_siblings(s, corpus)
    )
    if graded_two_siblings < minimum:
        _fail(
            f"only {graded_two_siblings} scenarios grade two instances of one "
            "family; v1 had 0 and that is precisely what made family expansion "
            f"and instance selection unmeasurable. Expected at least {minimum}."
        )
    report.record(
        "siblings",
        f"{len(sibling)} sibling-discriminating scenarios; "
        f"{graded_two_siblings} scenarios grade >=2 siblings of one family "
        f"(v1: 0; floor {minimum})",
    )


def _grades_two_siblings(scenario, corpus) -> bool:
    per_family: Counter[str] = Counter()
    for instance_id in scenario.grades:
        per_family[corpus.instance(instance_id).family_id] += 1
    return any(count >= 2 for count in per_family.values())


def _check_vague(scenarios, corpus, lexicon, report) -> None:
    vague = [s for s in scenarios if s.archetype == "vague_outcome"]
    emissions = {rule.id: {e.canonical for e in rule.emits} for rule in lexicon.rules}

    for scenario in vague:
        family = corpus.families[scenario.target_family_id]
        title_tokens = surface.content_tokens(
            " ".join((family.canonical_title, *family.aliases))
        )
        overlap = surface.content_tokens(scenario.query) & title_tokens
        if overlap:
            _fail(
                f"{scenario.scenario_id} is not vague: it shares "
                f"{sorted(overlap)} with its target's title"
            )

        rule_id = scenario.provenance["lexicon_rule"]
        claimed = set(scenario.provenance["emitted_canonicals"])
        if not claimed <= emissions.get(rule_id, set()):
            _fail(
                f"{scenario.scenario_id} cites rule {rule_id} for canonicals "
                f"{sorted(claimed)}, which it does not emit (lexicon drift)"
            )
        target = corpus.instance(scenario.target_instance_id)
        if not claimed <= set(target.fields):
            _fail(
                f"{scenario.scenario_id} target {target.report_instance_id} does "
                f"not carry {sorted(claimed - set(target.fields))}"
            )
    report.record(
        "vague",
        f"{len(vague)} vague scenarios, all token-disjoint from their target title "
        "and grounded in a live lexicon rule",
    )


def _check_no_answer(scenarios, corpus, report) -> None:
    reserved = [s for s in scenarios if s.archetype == "no_answer_reserved"]
    combos = [s for s in scenarios if s.archetype == "no_answer_impossible_combo"]

    corpus_text = " ".join(
        view.text for views in corpus.views.values() for view in views.values()
    ).casefold()
    for word in surface.RESERVED_VOCABULARY:
        if word in corpus_text:
            _fail(
                f"reserved word {word!r} appears in the corpus; it can no longer "
                "serve as an out-of-domain negative"
            )

    for scenario in reserved + combos:
        if any(grade > 0 for grade in scenario.grades.values()):
            _fail(f"{scenario.scenario_id} is a no-answer case but grades a positive")

    for scenario in combos:
        left, right = scenario.provenance["fields"]
        both = [
            i.report_instance_id for i in corpus.instances
            if {left, right} <= set(i.fields)
        ]
        if both:
            _fail(
                f"{scenario.scenario_id} claims no report carries both {left!r} "
                f"and {right!r}, but {both[:3]} do"
            )
    report.record(
        "no_answer",
        f"{len(reserved)} reserved-vocabulary and {len(combos)} impossible-combination "
        "scenarios; no positives, all combinations verified co-absent",
    )


def _check_ambiguity(scenarios, corpus, report) -> None:
    ambiguous = [s for s in scenarios if s.archetype == "ambiguous_clarification"]
    for scenario in ambiguous:
        left_id, right_id = scenario.provenance["families"]
        facet = scenario.provenance["differing_facet"]
        a = corpus.instance(corpus.families[left_id].instance_ids[0])
        b = corpus.instance(corpus.families[right_id].instance_ids[0])
        if getattr(a, facet) == getattr(b, facet):
            _fail(
                f"{scenario.scenario_id} claims {left_id} and {right_id} differ on "
                f"{facet}, but both are {getattr(a, facet)!r}"
            )
        deciding = surface.content_tokens(f"{getattr(a, facet)} {getattr(b, facet)}")
        if surface.content_tokens(scenario.query) & deciding:
            _fail(
                f"{scenario.scenario_id} already names the deciding value, so it is "
                "not ambiguous"
            )
    report.record(
        "ambiguity",
        f"{len(ambiguous)} scenarios; each pair differs on a real facet the query "
        "deliberately omits",
    )


def _check_misspellings(scenarios, corpus, report) -> None:
    vocabulary = surface.corpus_vocabulary_tokens(corpus)
    misspelled = [s for s in scenarios if s.archetype == "misspelling"]
    for scenario in misspelled:
        for change in scenario.provenance["noise"]:
            _, corrupted = change.split("->")
            if corrupted.casefold() in vocabulary:
                _fail(
                    f"{scenario.scenario_id} corrupted a token into {corrupted!r}, "
                    "which is real corpus vocabulary -- that is a different query, "
                    "not a misspelling"
                )
    report.record("misspelling", f"{len(misspelled)} scenarios, no vocabulary collisions")


def _check_grade_vocabulary(scenarios, report) -> None:
    grades = {g for s in scenarios for g in s.grades.values()}
    if not grades <= {0, 1, 2}:
        _fail(f"unexpected grade values: {sorted(grades - {0, 1, 2})}")
    answerability = Counter(s.answerability for s in scenarios)
    if not set(answerability) <= {0, 1, 2}:
        _fail(f"unexpected answerability values: {sorted(set(answerability))}")
    report.record(
        "grades",
        f"grades in {sorted(grades)}; answerability distribution "
        f"{dict(sorted(answerability.items()))}",
    )


def _check_splits(scenarios, splits, report) -> None:
    all_ids = {s.scenario_id for s in scenarios}
    union: set[str] = set()
    for name, ids in sorted(splits.items()):
        if not ids:
            _fail(f"split {name!r} is empty")
        if union & set(ids):
            _fail(f"split {name!r} overlaps an earlier split")
        union |= set(ids)
    if union != all_ids:
        _fail(
            f"splits are not a partition: {len(all_ids - union)} scenarios unassigned, "
            f"{len(union - all_ids)} assigned but unknown"
        )

    by_split = {name: set(ids) for name, ids in splits.items()}
    archetype_of = {s.scenario_id: s.archetype for s in scenarios}
    for name in ("calibration", "validation", "test"):
        counts = Counter(archetype_of[i] for i in by_split[name])
        if counts.get("sibling_discriminating", 0) < 20:
            _fail(
                f"split {name!r} has only "
                f"{counts.get('sibling_discriminating', 0)} sibling-discriminating "
                "scenarios; instance selection would be unmeasurable there"
            )

    # Family groups must not straddle: the same family in train and validation
    # leaks its instances' identity across the boundary.
    group_of = {s.scenario_id: s.group_key for s in scenarios}
    for name, ids in sorted(by_split.items()):
        for other, other_ids in sorted(by_split.items()):
            if name >= other:
                continue
            shared = {group_of[i] for i in ids} & {group_of[i] for i in other_ids}
            if shared:
                _fail(
                    f"groups straddle {name}/{other}: {sorted(shared)[:3]}"
                )
    report.record(
        "splits",
        "; ".join(f"{name}={len(ids)}" for name, ids in sorted(by_split.items()))
        + " (disjoint partition, no group straddles, minimums met)",
    )
