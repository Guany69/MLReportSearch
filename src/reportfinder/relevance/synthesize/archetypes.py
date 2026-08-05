"""One constructor per query archetype.

Each builds a query *from* a chosen target's metadata and knows the grades before
any retrieval happens. The grade vocabulary matches the v1 bundle so the existing
loaders and the `BUNDLE_LABEL_TO_CLASS` remap keep working:

    2  the report that answers the request
    1  a report that partially answers it (a sibling without the asked-for facet)
    0  a plausible near-miss that does not answer it

The two archetypes that justify the whole exercise:

* `sibling_discriminating` grades *every* instance of a family, so selecting the
  wrong copy is finally visible. v1 contained zero such queries -- which is why
  family expansion could not be measured at all.
* `vague_outcome` enforces zero content-token overlap with the target's title, so
  it cannot be answered by matching words. That is the case the architecture
  exists for and the case v1's title-matching generator could least produce.
"""

from __future__ import annotations

from dataclasses import dataclass

from ...corpus import CorpusModel
from . import catalog, surface

# Bundle answerability convention (2 answerable / 1 clarify / 0 none), kept so
# `BUNDLE_LABEL_TO_CLASS` in train_decision.py applies to v2 unchanged.
ANSWERABLE, CLARIFY, NO_ANSWER = 2, 1, 0

ROLE_PRIMARY = "Synthetic primary positive"
ROLE_PARTIAL = "Partial bundle member"
ROLE_NEGATIVE = "Synthetic hard negative"


@dataclass(frozen=True)
class GeneratedScenario:
    """One query, its grades, and exactly how it was built."""

    scenario_id: str
    query: str
    archetype: str
    grades: dict[str, int]
    judgment_roles: dict[str, str]
    answerability: int
    group_key: str
    provenance: dict[str, object]
    target_family_id: str | None = None
    target_instance_id: str | None = None
    edge_case_tags: tuple[str, ...] = ()

    @property
    def positives(self) -> set[str]:
        return {r for r, g in self.grades.items() if g > 0}


@dataclass
class Budget:
    """How many of each archetype to attempt."""

    sibling_discriminating: int = 450
    vague_outcome: int = 400
    acronym: int = 150
    misspelling: int = 150
    negation: int = 100
    short_query: int = 120
    multi_intent_single: int = 60
    multi_intent_split: int = 60
    no_answer_reserved: int = 150
    no_answer_impossible_combo: int = 100
    ambiguous_clarification: int = 150

    def as_dict(self) -> dict[str, int]:
        return {f.name: getattr(self, f.name) for f in self.__dataclass_fields__.values()}


@dataclass
class _Ids:
    """Scenario ids in the Q2xxxx range, clear of the v1 bundle's Q0xxxx-Q1xxxx."""

    next_value: int = 20001

    def take(self) -> str:
        value = self.next_value
        self.next_value += 1
        return f"Q{value:05d}"


def _negatives(corpus: CorpusModel, target, *, exclude: set[str], limit: int, rng):
    """Plausible near-misses: same category, lacking what was asked for.

    Same category so they are genuinely confusable -- a negative drawn at random
    would be trivially separable and would flatter every retriever equally.
    """
    pool = [
        i.report_instance_id for i in corpus.instances
        if i.category == target.category
        and i.report_instance_id not in exclude
        and i.family_id != target.family_id
    ]
    if not pool:
        return []
    picks = rng.choice(len(pool), size=min(limit, len(pool)), replace=False)
    return [pool[int(p)] for p in sorted(picks)]


# --- 1. sibling discrimination ----------------------------------------------


def sibling_discriminating(corpus, rng, ids, budget, phrasings) -> list[GeneratedScenario]:
    """Queries that pick one copy of a report out of its family.

    The fix for the gap that made expansion unmeasurable: every sibling is
    graded, the carrier 2 and the rest 1. A system that returns the family but
    the wrong copy now scores strictly worse than one that returns the right
    copy, which under v1's labels it did not.
    """
    out: list[GeneratedScenario] = []
    for family in catalog.multi_instance_families(corpus):
        if len(out) >= budget:
            break
        options = catalog.sibling_distinguishers(family, corpus)
        if not options:
            continue

        distinguisher = options[int(rng.integers(len(options)))]
        carrier = corpus.instance(distinguisher.carrier_instance_id)
        topic = _family_topic(family, corpus, phrasings, rng)
        query = f"{topic} {distinguisher.phrase}"

        grades, roles = {}, {}
        for instance_id in family.instance_ids:
            is_carrier = instance_id == distinguisher.carrier_instance_id
            grades[instance_id] = ANSWERABLE if is_carrier else CLARIFY
            roles[instance_id] = ROLE_PRIMARY if is_carrier else ROLE_PARTIAL
        for negative in _negatives(
            corpus, carrier, exclude=set(family.instance_ids), limit=3, rng=rng,
        ):
            grades[negative], roles[negative] = 0, ROLE_NEGATIVE

        out.append(GeneratedScenario(
            scenario_id=ids.take(), query=query, archetype="sibling_discriminating",
            grades=grades, judgment_roles=roles, answerability=ANSWERABLE,
            group_key=family.family_id,
            target_family_id=family.family_id,
            target_instance_id=distinguisher.carrier_instance_id,
            edge_case_tags=("sibling_selection",),
            provenance={
                "distinguishing_kind": distinguisher.kind,
                "distinguishing_value": distinguisher.value,
                "sibling_count": len(family.instance_ids),
                "construction": "family topic + facet carried by exactly one sibling",
            },
        ))
    return out


def _family_topic(family, corpus, phrasings, rng) -> str:
    """How to refer to the family without naming the distinguishing facet."""
    shared = set(corpus.instance(family.instance_ids[0]).fields)
    for instance_id in family.instance_ids[1:]:
        shared &= set(corpus.instance(instance_id).fields)

    options = sorted(f for f in shared if f in phrasings)
    if options:
        chosen = options[int(rng.integers(len(options)))]
        rule_id, phrase = phrasings[chosen][int(rng.integers(len(phrasings[chosen])))]
        return phrase
    return surface.humanize(family.canonical_title)


# --- 2. vague business outcomes ---------------------------------------------


def vague_outcome(corpus, rng, ids, budget, phrasings, emissions) -> list[GeneratedScenario]:
    """Outcome-shaped requests that share no content word with the answer.

    Disjointness is enforced, not hoped for. A vague query that happens to share
    a token with its target title is answerable lexically, and including one
    would let BM25F score well on the very slice built to show where it cannot.
    """
    by_family: dict[str, list[str]] = {}
    for instance in corpus.instances:
        by_family.setdefault(instance.family_id, []).append(instance.report_instance_id)

    rules = sorted(emissions)
    out: list[GeneratedScenario] = []
    attempts = 0
    while len(out) < budget and attempts < budget * 60:
        attempts += 1
        rule_id = rules[int(rng.integers(len(rules)))]
        canonicals = emissions[rule_id]
        if not canonicals:
            continue

        phrase_options = [
            phrase for canonical in canonicals
            for rid, phrase in phrasings.get(canonical, ())
            if rid == rule_id
        ]
        if not phrase_options:
            continue
        phrase = sorted(set(phrase_options))[int(rng.integers(len(set(phrase_options))))]

        carriers = [
            i for i in corpus.instances if set(canonicals) <= set(i.fields)
        ]
        if not carriers:
            continue
        target = carriers[int(rng.integers(len(carriers)))]
        family = corpus.families[target.family_id]

        template = surface.OUTCOME_TEMPLATES[
            int(rng.integers(len(surface.OUTCOME_TEMPLATES)))
        ]
        query = template.format(phrase=phrase)

        title_tokens = surface.content_tokens(
            " ".join((family.canonical_title, *family.aliases))
        )
        if surface.content_tokens(query) & title_tokens:
            continue  # answerable by word overlap; not a vague case

        grades, roles = {}, {}
        for instance_id in by_family[target.family_id]:
            carries = set(canonicals) <= set(corpus.instance(instance_id).fields)
            grades[instance_id] = ANSWERABLE if carries else CLARIFY
            roles[instance_id] = ROLE_PRIMARY if carries else ROLE_PARTIAL
        for negative in _negatives(
            corpus, target, exclude=set(by_family[target.family_id]), limit=2, rng=rng,
        ):
            grades[negative], roles[negative] = 0, ROLE_NEGATIVE

        out.append(GeneratedScenario(
            scenario_id=ids.take(), query=query, archetype="vague_outcome",
            grades=grades, judgment_roles=roles, answerability=ANSWERABLE,
            group_key=target.family_id,
            target_family_id=target.family_id,
            target_instance_id=target.report_instance_id,
            edge_case_tags=("vague", "business_outcome", "zero_title_overlap"),
            provenance={
                "lexicon_rule": rule_id,
                "emitted_canonicals": list(canonicals),
                "phrase": phrase,
                "construction": "lexicon phrase -> emitted fields -> carrying report",
            },
        ))
    return out


# --- 3. surface noise on an otherwise plain query ----------------------------


def _plain_query(instance, phrasings, rng) -> str:
    options = sorted(f for f in instance.fields if f in phrasings)
    if options:
        chosen = options[int(rng.integers(len(options)))]
        _, phrase = phrasings[chosen][int(rng.integers(len(phrasings[chosen])))]
        return f"{phrase} by {surface.humanize(instance.category)}"
    return surface.humanize(instance.title)


def _noised(corpus, rng, ids, budget, phrasings, *, archetype, noise, tags):
    by_family: dict[str, list[str]] = {}
    for instance in corpus.instances:
        by_family.setdefault(instance.family_id, []).append(instance.report_instance_id)

    out: list[GeneratedScenario] = []
    attempts = 0
    while len(out) < budget and attempts < budget * 40:
        attempts += 1
        target = corpus.instances[int(rng.integers(len(corpus.instances)))]
        base = _plain_query(target, phrasings, rng)
        query, applied = noise(base, rng)
        if not applied:
            continue

        grades = {target.report_instance_id: ANSWERABLE}
        roles = {target.report_instance_id: ROLE_PRIMARY}
        for sibling in by_family[target.family_id]:
            grades.setdefault(sibling, CLARIFY)
            roles.setdefault(sibling, ROLE_PARTIAL)
        for negative in _negatives(
            corpus, target, exclude=set(by_family[target.family_id]), limit=2, rng=rng,
        ):
            grades[negative], roles[negative] = 0, ROLE_NEGATIVE

        out.append(GeneratedScenario(
            scenario_id=ids.take(), query=query, archetype=archetype,
            grades=grades, judgment_roles=roles, answerability=ANSWERABLE,
            group_key=target.family_id,
            target_family_id=target.family_id,
            target_instance_id=target.report_instance_id,
            edge_case_tags=tags,
            provenance={"base_query": base, "noise": applied,
                        "construction": f"plain query + {archetype} surface noise"},
        ))
    return out


def acronym(corpus, rng, ids, budget, phrasings):
    return _noised(
        corpus, rng, ids, budget, phrasings, archetype="acronym",
        noise=surface.apply_acronyms, tags=("acronym", "surface_noise"),
    )


def misspelling(corpus, rng, ids, budget, phrasings, *, vocabulary):
    def noise(text, generator):
        query, applied = surface.apply_typo(text, generator, vocabulary=vocabulary)
        return query, ([applied] if applied else [])

    return _noised(
        corpus, rng, ids, budget, phrasings, archetype="misspelling",
        noise=noise, tags=("misspelling", "surface_noise"),
    )


# --- 4. negation -------------------------------------------------------------


def negation(corpus, rng, ids, budget, phrasings) -> list[GeneratedScenario]:
    """"X but not Y", where a named near-miss really does carry Y.

    The excluded facet is drawn from an actual other report, so a system that
    ignores the negation has somewhere concrete to go wrong.
    """
    out: list[GeneratedScenario] = []
    attempts = 0
    while len(out) < budget and attempts < budget * 60:
        attempts += 1
        target = corpus.instances[int(rng.integers(len(corpus.instances)))]
        peers = [
            i for i in corpus.instances
            if i.category == target.category and i.family_id != target.family_id
        ]
        if not peers:
            continue
        peer = peers[int(rng.integers(len(peers)))]

        excluded = sorted(set(peer.fields) - set(target.fields))
        if not excluded:
            continue
        excluded_field = excluded[int(rng.integers(len(excluded)))]

        base = _plain_query(target, phrasings, rng)
        query = f"{base} but not {surface.humanize(excluded_field)}"

        grades = {target.report_instance_id: ANSWERABLE,
                  peer.report_instance_id: 0}
        roles = {target.report_instance_id: ROLE_PRIMARY,
                 peer.report_instance_id: ROLE_NEGATIVE}

        out.append(GeneratedScenario(
            scenario_id=ids.take(), query=query, archetype="negation",
            grades=grades, judgment_roles=roles, answerability=ANSWERABLE,
            group_key=target.family_id,
            target_family_id=target.family_id,
            target_instance_id=target.report_instance_id,
            edge_case_tags=("negation",),
            provenance={"excluded_field": excluded_field,
                        "excluded_carrier": peer.report_instance_id,
                        "construction": "target query + a facet only the near-miss has"},
        ))
    return out


# --- 5. very short queries ---------------------------------------------------


def short_query(corpus, rng, ids, budget) -> list[GeneratedScenario]:
    """One to three tokens naming a field only one family carries."""
    out: list[GeneratedScenario] = []
    for field_name, carriers in sorted(catalog.rare_fields(corpus).items()):
        if len(out) >= budget:
            break
        tokens = surface.humanize(field_name).split()
        if not 1 <= len(tokens) <= 3:
            continue

        family_id = corpus.instance(carriers[0]).family_id
        grades = {i: ANSWERABLE for i in carriers}
        roles = {i: ROLE_PRIMARY for i in carriers}
        for sibling in corpus.families[family_id].instance_ids:
            grades.setdefault(sibling, CLARIFY)
            roles.setdefault(sibling, ROLE_PARTIAL)

        out.append(GeneratedScenario(
            scenario_id=ids.take(), query=" ".join(tokens), archetype="short_query",
            grades=grades, judgment_roles=roles, answerability=ANSWERABLE,
            group_key=family_id, target_family_id=family_id,
            target_instance_id=carriers[0],
            edge_case_tags=("short_query",),
            provenance={"rare_field": field_name, "carrier_count": len(carriers),
                        "construction": "field with df<=3 confined to one family"},
        ))
    return out


# --- 6. multi-intent ---------------------------------------------------------


def multi_intent(corpus, rng, ids, budget_single, budget_split, phrasings):
    """Two intents, either satisfiable by one report or by none.

    The split case is the interesting one: no single report covers both, so the
    honest answer is a clarification rather than a confident pick.
    """
    single, split = [], []
    attempts = 0
    while (len(single) < budget_single or len(split) < budget_split) \
            and attempts < (budget_single + budget_split) * 60:
        attempts += 1
        fields = sorted(phrasings)
        left = fields[int(rng.integers(len(fields)))]
        right = fields[int(rng.integers(len(fields)))]
        if left == right:
            continue

        _, left_phrase = phrasings[left][int(rng.integers(len(phrasings[left])))]
        _, right_phrase = phrasings[right][int(rng.integers(len(phrasings[right])))]
        query = f"{left_phrase} and also {right_phrase}"

        both = [i for i in corpus.instances if {left, right} <= set(i.fields)]
        if both and len(single) < budget_single:
            target = both[int(rng.integers(len(both)))]
            grades = {target.report_instance_id: ANSWERABLE}
            roles = {target.report_instance_id: ROLE_PRIMARY}
            single.append(GeneratedScenario(
                scenario_id=ids.take(), query=query, archetype="multi_intent_single",
                grades=grades, judgment_roles=roles, answerability=ANSWERABLE,
                group_key=target.family_id, target_family_id=target.family_id,
                target_instance_id=target.report_instance_id,
                edge_case_tags=("multi_intent",),
                provenance={"fields": [left, right],
                            "construction": "two intents, one report carries both"},
            ))
            continue

        if both or len(split) >= budget_split:
            continue
        left_only = [i for i in corpus.instances if left in set(i.fields)]
        right_only = [i for i in corpus.instances if right in set(i.fields)]
        if not left_only or not right_only:
            continue

        primary = left_only[int(rng.integers(len(left_only)))]
        secondary = right_only[int(rng.integers(len(right_only)))]
        grades = {primary.report_instance_id: ANSWERABLE,
                  secondary.report_instance_id: CLARIFY}
        roles = {primary.report_instance_id: ROLE_PRIMARY,
                 secondary.report_instance_id: ROLE_PARTIAL}
        split.append(GeneratedScenario(
            scenario_id=ids.take(), query=query, archetype="multi_intent_split",
            grades=grades, judgment_roles=roles, answerability=CLARIFY,
            group_key=primary.family_id, target_family_id=primary.family_id,
            target_instance_id=primary.report_instance_id,
            edge_case_tags=("multi_intent", "needs_clarification"),
            provenance={"fields": [left, right],
                        "construction": "two intents, no single report carries both"},
        ))
    return single, split


# --- 7. no-answer ------------------------------------------------------------


def no_answer_reserved(corpus, rng, ids, budget) -> list[GeneratedScenario]:
    """Out-of-domain by construction.

    `query/expansion/rules.py` refuses to load a lexicon that grounds these
    words, and they appear nowhere in the estate -- so no evidence can exist and
    the only correct response is to decline.
    """
    out: list[GeneratedScenario] = []
    for index in range(budget):
        words = list(surface.RESERVED_VOCABULARY)
        rng.shuffle(words)
        template = surface.NO_ANSWER_TEMPLATES[
            int(rng.integers(len(surface.NO_ANSWER_TEMPLATES)))
        ]
        query = template.format(a=words[0], b=words[1], c=words[2])

        # A couple of graded-0 rows so the scenario survives the judgement join,
        # which keys on judgement lines rather than on the scenario list.
        near = _negatives(
            corpus, corpus.instances[int(rng.integers(len(corpus.instances)))],
            exclude=set(), limit=2, rng=rng,
        )
        out.append(GeneratedScenario(
            scenario_id=ids.take(), query=query, archetype="no_answer_reserved",
            grades=dict.fromkeys(near, 0),
            judgment_roles=dict.fromkeys(near, ROLE_NEGATIVE),
            answerability=NO_ANSWER, group_key=f"reserved-{index % 8}",
            edge_case_tags=("no_answer", "out_of_domain"),
            provenance={"reserved_vocabulary": words,
                        "construction": "reserved out-of-domain vocabulary"},
        ))
    return out


def no_answer_impossible_combo(corpus, rng, ids, budget, phrasings):
    """In-domain but unsatisfiable: two real fields no report carries together.

    Harder than the reserved case, and the one that exercises the "no single
    report carries all requested fields" gate rather than plain absence.
    """
    pairs = catalog.co_absent_field_pairs(corpus, limit=budget, rng=rng)
    out: list[GeneratedScenario] = []
    for index, (left, right) in enumerate(pairs):
        query = (
            f"a report with both {surface.humanize(left)} "
            f"and {surface.humanize(right)}"
        )
        near = [
            i.report_instance_id for i in corpus.instances
            if left in set(i.fields)
        ][:1] + [
            i.report_instance_id for i in corpus.instances
            if right in set(i.fields)
        ][:1]

        out.append(GeneratedScenario(
            scenario_id=ids.take(), query=query,
            archetype="no_answer_impossible_combo",
            grades=dict.fromkeys(near, 0),
            judgment_roles=dict.fromkeys(near, ROLE_NEGATIVE),
            answerability=NO_ANSWER, group_key=f"impossible-{index % 8}",
            edge_case_tags=("no_answer", "impossible_combination"),
            provenance={"fields": [left, right],
                        "construction": "two real fields no single report carries"},
        ))
    return out


# --- 8. ambiguity ------------------------------------------------------------


def ambiguous_clarification(corpus, rng, ids, budget) -> list[GeneratedScenario]:
    """Two families a user could equally mean, with the deciding facet omitted.

    Constructed so a clarifying question has something real to ask about: the
    two families genuinely differ on a facet, and the query deliberately does not
    say which one is wanted.
    """
    out: list[GeneratedScenario] = []
    for left, right, facet in catalog.confusable_family_pairs(corpus, limit=budget * 3):
        if len(out) >= budget:
            break
        a = corpus.instance(left.instance_ids[0])
        b = corpus.instance(right.instance_ids[0])

        shared = surface.content_tokens(left.normalized_title) & surface.content_tokens(
            right.normalized_title
        )
        if not shared:
            continue
        query = " ".join(sorted(shared)) + " report"

        # The query must not already name the deciding values, or it would not be
        # ambiguous at all.
        deciding = surface.content_tokens(
            f"{getattr(a, facet)} {getattr(b, facet)}"
        )
        if surface.content_tokens(query) & deciding:
            continue

        grades, roles = {}, {}
        for family in (left, right):
            for index, instance_id in enumerate(family.instance_ids):
                grades[instance_id] = ANSWERABLE if index == 0 else CLARIFY
                roles[instance_id] = ROLE_PRIMARY if index == 0 else ROLE_PARTIAL

        out.append(GeneratedScenario(
            scenario_id=ids.take(), query=query,
            archetype="ambiguous_clarification",
            grades=grades, judgment_roles=roles, answerability=CLARIFY,
            group_key=f"{left.family_id}|{right.family_id}",
            target_family_id=left.family_id,
            target_instance_id=left.instance_ids[0],
            edge_case_tags=("ambiguous", "needs_clarification"),
            provenance={
                "families": [left.family_id, right.family_id],
                "differing_facet": facet,
                "values": [getattr(a, facet), getattr(b, facet)],
                "construction": "two confusable families, deciding facet omitted",
            },
        ))
    return out
