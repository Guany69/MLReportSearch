"""The v2 label generator's construction claims, re-verified independently.

The generator already self-checks (`validate.py`, fatal). These tests exist for
the claims that must hold *for the generator to be worth having at all*, so that
weakening a self-check is caught rather than being the same edit twice:

* every sibling of a family is graded, and the carrier uniquely carries the facet
  -- v1 graded zero sibling pairs, which is why instance selection could not be
  measured;
* vague queries share no content token with their target's title -- otherwise
  they are answerable lexically and measure nothing new;
* no-answer cases really have no answer;
* the same seed produces the same bytes.

Most run against a small in-code corpus so they are fast and independent of the
estate. The two that need the real workbook are guarded.

Run: uv run pytest tests/test_synthesize_v2.py -v
"""

from __future__ import annotations

import json
from collections import Counter

import numpy as np
import pandas as pd
import pytest

from reportfinder.corpus import build_corpus_model
from reportfinder.query.expansion.rules import load_lexicon
from reportfinder.query.expansion.vocabulary import CorpusVocabulary
from reportfinder.relevance.synthesize import Budget, build_scenarios, catalog, emit, surface
from reportfinder.relevance.synthesize.validate import ValidationError
from reportfinder.relevance.synthesize.validate import validate as run_validation

from .conftest import requires_real_estate

# A corpus with the shapes every archetype needs: multi-instance families whose
# siblings differ on exactly one facet, rare fields confined to one family, and
# two confusable families in one category.
ROWS = [
    # family "worker roster" -- three siblings, each with a unique field
    ("R0001", "Worker Roster", ["Worker", "Location", "Hire Date"], "Worker Data",
     "All Workers", "Advanced", ["Effective Date"]),
    ("R0002", "Worker Roster", ["Worker", "Location", "Termination Date"], "Worker Data",
     "All Workers", "Advanced", ["Effective Date"]),
    ("R0003", "Worker Roster", ["Worker", "Location", "Cost Center"], "Worker Data",
     "All Workers", "Matrix", ["Effective Date"]),
    # family "headcount summary" -- two siblings differing on data source
    ("R0004", "Headcount Summary", ["Headcount", "Supervisory Organization"],
     "Worker Data", "All Workers", "Matrix", ["As Of Date"]),
    ("R0005", "Headcount Summary", ["Headcount", "Supervisory Organization",
     "Average Headcount"], "Worker Data", "All Active and Terminated", "Matrix",
     ["As Of Date"]),
    # confusable pair in one category, differing on data source
    ("R0006", "Payroll Detail Report", ["Net Pay", "Pay Period"], "Payroll",
     "All Workers", "Advanced", ["Pay Period"]),
    ("R0007", "Payroll Detail Extract", ["Net Pay", "Gross Pay"], "Payroll",
     "All Active and Terminated", "Advanced", ["Pay Period"]),
    # a rare field confined to one family
    ("R0008", "Requisition Aging", ["Time to Fill", "Requisition ID"], "Recruiting",
     "All Workers", "Advanced", []),
    ("R0009", "Termination Analysis", ["Termination Reason", "Termination Date"],
     "Talent", "All Workers", "Advanced", []),
    ("R0010", "Compensation Review", ["Salary Range", "Job Profile"], "Compensation",
     "All Workers", "Advanced", []),
]


def _frame():
    return pd.DataFrame([
        {
            "report_key": key, "title_key": title.casefold(), "source_row": 100 + i,
            "title": title, "description": "", "category": category,
            "data_source": source, "report_type": report_type, "prompts": prompts,
            "fields": fields, "tags": "", "area_where_used": "", "worklet": "Standard",
            "chart_type": "", "landing_page": "", "worklet_landing_pages": "",
            "shared": "Yes", "field_meta": [],
        }
        for i, (key, title, fields, category, source, report_type, prompts)
        in enumerate(ROWS)
    ])


@pytest.fixture(scope="module")
def corpus():
    return build_corpus_model(
        _frame(), ingest_mode="legacy_single_file", source_file="fixture"
    )


@pytest.fixture(scope="module")
def lexicon():
    return load_lexicon(CorpusVocabulary.from_frame(_frame()))


@pytest.fixture(scope="module")
def scenarios(corpus, lexicon):
    return build_scenarios(corpus, lexicon, seed=7, budget=Budget(
        sibling_discriminating=10, vague_outcome=10, acronym=5, misspelling=5,
        negation=5, short_query=5, multi_intent_single=3, multi_intent_split=3,
        no_answer_reserved=5, no_answer_impossible_combo=3,
        ambiguous_clarification=5,
    ))


# --- the claim that justifies the whole exercise -----------------------------


def test_sibling_queries_grade_every_sibling(corpus, scenarios):
    """v1 graded zero sibling pairs, so returning the wrong copy of the right
    report cost nothing. Every sibling must carry a grade for that to be visible."""
    sibling = [s for s in scenarios if s.archetype == "sibling_discriminating"]
    assert sibling, "fixture must produce sibling-discriminating scenarios"

    for scenario in sibling:
        family = corpus.families[scenario.target_family_id]
        assert set(family.instance_ids) <= set(scenario.grades)
        assert scenario.grades[scenario.target_instance_id] == 2
        others = set(family.instance_ids) - {scenario.target_instance_id}
        assert all(scenario.grades[i] == 1 for i in others)


def test_the_distinguishing_facet_is_unique_to_the_carrier(corpus, scenarios):
    """If two siblings carried it, the query would have two right answers and
    grading one of them 1 would invent a preference the catalog never states."""
    for scenario in scenarios:
        if scenario.archetype != "sibling_discriminating":
            continue
        kind = scenario.provenance["distinguishing_kind"]
        value = scenario.provenance["distinguishing_value"]
        family = corpus.families[scenario.target_family_id]

        carriers = [
            i for i in family.instance_ids
            if value in set(catalog._values(corpus.instance(i), kind))
        ]
        assert carriers == [scenario.target_instance_id]


def test_many_scenarios_grade_more_than_one_sibling(corpus, scenarios):
    for scenario in scenarios:
        per_family: Counter[str] = Counter()
        for instance_id in scenario.grades:
            per_family[corpus.instance(instance_id).family_id] += 1
        if scenario.archetype == "sibling_discriminating":
            assert max(per_family.values()) >= 2


# --- vague queries must actually be vague ------------------------------------


def test_vague_queries_share_no_content_token_with_the_target_title(corpus, scenarios):
    """A vague query that shares a word with its answer is answerable lexically,
    and including one would flatter BM25F on the very slice built to expose it."""
    vague = [s for s in scenarios if s.archetype == "vague_outcome"]
    assert vague

    for scenario in vague:
        family = corpus.families[scenario.target_family_id]
        title_tokens = surface.content_tokens(
            " ".join((family.canonical_title, *family.aliases))
        )
        assert not (surface.content_tokens(scenario.query) & title_tokens)


def test_vague_targets_carry_the_canonicals_the_cited_rule_emits(
    corpus, lexicon, scenarios
):
    """Grounds the grade in a live lexicon rule, so lexicon drift breaks
    generation loudly instead of producing quietly-wrong labels."""
    emissions = {rule.id: {e.canonical for e in rule.emits} for rule in lexicon.rules}
    for scenario in scenarios:
        if scenario.archetype != "vague_outcome":
            continue
        rule_id = scenario.provenance["lexicon_rule"]
        claimed = set(scenario.provenance["emitted_canonicals"])
        assert claimed <= emissions[rule_id]
        assert claimed <= set(corpus.instance(scenario.target_instance_id).fields)


# --- no-answer cases really have no answer -----------------------------------


def test_reserved_vocabulary_appears_nowhere_in_the_corpus(corpus):
    text = " ".join(
        view.text for views in corpus.views.values() for view in views.values()
    ).casefold()
    for word in surface.RESERVED_VOCABULARY:
        assert word not in text


def test_no_answer_scenarios_grade_no_positives(scenarios):
    for scenario in scenarios:
        if scenario.archetype.startswith("no_answer"):
            assert all(grade == 0 for grade in scenario.grades.values())
            assert scenario.answerability == 0


def test_impossible_combinations_have_no_carrying_report(corpus, scenarios):
    """In-domain and unsatisfiable: both fields are real, so only the
    'no single report carries all requested fields' gate can catch it."""
    combos = [s for s in scenarios if s.archetype == "no_answer_impossible_combo"]
    for scenario in combos:
        left, right = scenario.provenance["fields"]
        assert not [
            i for i in corpus.instances if {left, right} <= set(i.fields)
        ]


# --- ambiguity has something real to ask about -------------------------------


def test_ambiguous_pairs_differ_on_a_facet_the_query_omits(corpus, scenarios):
    for scenario in scenarios:
        if scenario.archetype != "ambiguous_clarification":
            continue
        left_id, right_id = scenario.provenance["families"]
        facet = scenario.provenance["differing_facet"]
        a = corpus.instance(corpus.families[left_id].instance_ids[0])
        b = corpus.instance(corpus.families[right_id].instance_ids[0])

        assert getattr(a, facet) != getattr(b, facet)
        deciding = surface.content_tokens(f"{getattr(a, facet)} {getattr(b, facet)}")
        assert not (surface.content_tokens(scenario.query) & deciding)


# --- surface noise -----------------------------------------------------------


def test_misspellings_never_collide_with_real_vocabulary(corpus, scenarios):
    """A corruption that spells another catalog word is a different question,
    not a misspelling, and would have a different right answer."""
    vocabulary = surface.corpus_vocabulary_tokens(corpus)
    for scenario in scenarios:
        if scenario.archetype != "misspelling":
            continue
        for change in scenario.provenance["noise"]:
            _, corrupted = change.split("->")
            assert corrupted.casefold() not in vocabulary


def test_short_queries_name_a_field_confined_to_one_family(corpus, scenarios):
    for scenario in scenarios:
        if scenario.archetype != "short_query":
            continue
        assert 1 <= len(scenario.query.split()) <= 3
        field_name = scenario.provenance["rare_field"]
        families = {
            i.family_id for i in corpus.instances if field_name in set(i.fields)
        }
        assert len(families) == 1


# --- determinism and splits --------------------------------------------------


def test_the_same_seed_produces_the_same_scenarios(corpus, lexicon):
    budget = Budget(
        sibling_discriminating=8, vague_outcome=8, acronym=3, misspelling=3,
        negation=3, short_query=3, multi_intent_single=2, multi_intent_split=2,
        no_answer_reserved=3, no_answer_impossible_combo=2,
        ambiguous_clarification=3,
    )
    first = build_scenarios(corpus, lexicon, seed=11, budget=budget)
    second = build_scenarios(corpus, lexicon, seed=11, budget=budget)

    assert [s.scenario_id for s in first] == [s.scenario_id for s in second]
    assert [s.query for s in first] == [s.query for s in second]
    assert [s.grades for s in first] == [s.grades for s in second]


def test_splits_are_a_partition_and_include_calibration(scenarios):
    splits = emit.assign_splits(scenarios, np.random.default_rng(3))

    assert set(splits) == {"train", "calibration", "validation", "test"}
    union: set[str] = set()
    for ids in splits.values():
        assert not union & set(ids), "splits must be disjoint"
        union |= set(ids)
    assert union == {s.scenario_id for s in scenarios}


def test_no_family_group_straddles_a_split(scenarios):
    """The same report being the answer in train and validation makes validation
    measure memorisation."""
    splits = emit.assign_splits(scenarios, np.random.default_rng(3))
    group_of = {s.scenario_id: s.group_key for s in scenarios}

    groups = {name: {group_of[i] for i in ids} for name, ids in splits.items()}
    names = sorted(groups)
    for i, left in enumerate(names):
        for right in names[i + 1:]:
            assert not (groups[left] & groups[right])


# --- the validator has teeth -------------------------------------------------


def test_validation_rejects_a_vague_query_that_overlaps_its_target(corpus, lexicon):
    """The self-check must fail on a hand-broken scenario, or it is decoration."""
    import dataclasses

    scenarios = build_scenarios(corpus, lexicon, seed=5, budget=Budget(
        sibling_discriminating=6, vague_outcome=6, acronym=0, misspelling=0,
        negation=0, short_query=0, multi_intent_single=0, multi_intent_split=0,
        no_answer_reserved=0, no_answer_impossible_combo=0,
        ambiguous_clarification=0,
    ))
    vague = next(s for s in scenarios if s.archetype == "vague_outcome")
    title = corpus.families[vague.target_family_id].canonical_title
    broken = [
        dataclasses.replace(vague, query=f"{vague.query} {title}") if s is vague else s
        for s in scenarios
    ]

    with pytest.raises(ValidationError, match="not vague"):
        run_validation(broken, corpus, lexicon, min_sibling_graded=1)


def test_validation_rejects_a_no_answer_scenario_with_a_positive(corpus, lexicon):
    import dataclasses

    scenarios = build_scenarios(corpus, lexicon, seed=5, budget=Budget(
        sibling_discriminating=6, vague_outcome=0, acronym=0, misspelling=0,
        negation=0, short_query=0, multi_intent_single=0, multi_intent_split=0,
        no_answer_reserved=4, no_answer_impossible_combo=0,
        ambiguous_clarification=0,
    ))
    target = next(s for s in scenarios if s.archetype == "no_answer_reserved")
    broken = [
        dataclasses.replace(target, grades={**target.grades, "R0001": 2})
        if s is target else s
        for s in scenarios
    ]

    with pytest.raises(ValidationError, match="grades a positive"):
        run_validation(broken, corpus, lexicon, min_sibling_graded=1)


# --- the real estate ---------------------------------------------------------


V2_ROOT = None


@requires_real_estate
def test_generating_against_the_real_estate_passes_every_check(tmp_path):
    """The end-to-end run, including the >=600 sibling-graded floor."""
    from reportfinder.relevance.synthesize import generate

    manifest = generate(tmp_path / "v2", seed=20260804)

    assert manifest["label_source"] == "synthetic_v2_generative"
    assert manifest["human_validated"] is False
    assert manifest["scenario_count"] > 1500
    assert set(manifest["split_sizes"]) == {
        "train", "calibration", "validation", "test",
    }
    report = json.loads((tmp_path / "v2" / "validation_report.json").read_text())
    assert "v1: 0" in report["checks"]["siblings"]


@requires_real_estate
def test_the_real_v2_root_loads_through_the_existing_loader(tmp_path):
    """A v2 root must be readable by `load_labelled_queries` unchanged, or
    switching label sets becomes a migration instead of a flag."""
    from reportfinder.relevance.synthesize import generate
    from reportfinder.training.datasets import label_provenance, load_labelled_queries

    generate(tmp_path / "v2", seed=20260804)
    root = tmp_path / "v2"

    for split in ("train", "calibration", "validation", "test"):
        queries = load_labelled_queries(root, split=split)
        assert queries, f"{split} split loaded no queries"
        assert all(q.query for q in queries), "every scenario must have query text"

    assert label_provenance(root)["training_label_source"] == "synthetic_v2_generative"
    assert label_provenance(root)["approved"] is False
