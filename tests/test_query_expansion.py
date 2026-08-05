from __future__ import annotations

import numpy as np
import pandas as pd

from reportfinder.confidence.answerability import AnswerabilityOutcome, decide_answerability
from reportfinder.confidence.support import SupportAccounting, TermSupport, compute_support
from reportfinder.query.intent import IntentParser
from reportfinder.query.normalize import tokenize
from reportfinder.query.trie import PhraseTrie
from reportfinder.ranking.features import mandatory_gate
from reportfinder.retrieval.forms import aggregate_top_two


def frame():
    rows = []
    fields = [
        "Manager", "Previous Manager", "Termination Date", "Termination Reason",
        "Termination Type", "Net Pay", "Pay Period", "Headcount", "Earnings",
    ]
    for index in range(3):
        rows.append({
            "title": f"Report {index}", "fields": fields, "category": "Worker",
            "data_source": "Workers", "field_meta": [],
        })
    return pd.DataFrame(rows)


def test_tokenization_preserves_original_offsets():
    query = "Show take-home pay"
    tokens = tokenize(query)
    assert [token.text for token in tokens] == ["show", "take", "home", "pay"]
    assert query[tokens[1].char_start:tokens[3].char_end] == "take-home pay"


def test_phrase_trie_returns_nested_matches():
    trie = PhraseTrie()
    trie.add(("manager",), "generic")
    trie.add(("previous", "manager"), "specific")
    matches = trie.find_all(tokenize("previous manager"))
    assert {(match.tok_start, match.tok_end, match.payload) for match in matches} == {
        (0, 2, "specific"), (1, 2, "generic"),
    }


def test_specificity_suppression_and_literal_field_is_mandatory():
    intent = IntentParser(frame()).parse("show previous manager")
    assert [field.value for field in intent.fields] == ["Previous Manager"]
    assert intent.mandatory_fields == ["Previous Manager"]
    assert [item.canonical for item in intent.expansion.suppressed] == ["Manager"]


def test_requirement_marker_distributes_across_coordinator():
    intent = IntentParser(frame()).parse("need Termination Date and Manager")
    assert set(intent.mandatory_fields) == {"Termination Date", "Manager"}


def test_plain_language_rules_and_support_accounting():
    intent = IntentParser(frame()).parse("voluntary exits")
    assert {"Termination Reason", "Termination Type"} <= {field.value for field in intent.fields}
    support = compute_support(intent.expansion, {"termination"}, top_fields=frame().iloc[0]["fields"])
    assert support.ratio >= .80
    assert not support.unsupported_terms


def test_typo_correction_is_conservative_and_auditable():
    parser = IntentParser(frame())
    typo = parser.parse("by manger")
    assert typo.expansion.typos[0].corrected == "manager"
    assert [field.value for field in typo.fields] == ["Manager"]
    assert typo.mandatory_fields == []
    assert parser.engine.corrections.correct("earnings", 2) is None


def test_top_two_form_aggregation_bounds_expansion_influence():
    scores = np.asarray([[1.0, 0.0], [1.0, 1.0]])
    result = aggregate_top_two(scores, [1.0, .85], .30)
    np.testing.assert_allclose(result.combined[1], .85 / 1.30, atol=1e-12)


def test_mandatory_gate_is_bounded_and_monotone():
    gates = [
        mandatory_gate({"has_mandatory_requirements": 1, "mandatory_coverage": value})
        for value in (0.0, .5, 1.0)
    ]
    assert .14 < gates[0] < gates[1] < gates[2] == 1.0


def test_support_exactly_at_hard_floor_abstains():
    support = SupportAccounting(
        (
            TermSupport("known", True, "typo", .8),
            TermSupport("x", False, "none", 0.0),
            TermSupport("y", False, "none", 0.0),
            TermSupport("z", False, "none", 0.0),
        ),
        .20, ("x", "y", "z"), (), 0,
    )
    decision = decide_answerability(
        top_score=.8, margin=.2, field_coverage=0, retriever_agreement=.5,
        parser_confidence=.3, support=support,
    )
    assert decision.outcome is AnswerabilityOutcome.NO_SATISFACTORY_REPORT
