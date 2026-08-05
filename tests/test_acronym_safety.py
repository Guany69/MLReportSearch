from __future__ import annotations

import pandas as pd

from reportfinder.query.intent import IntentParser


def _parser():
    return IntentParser(pd.DataFrame([{
        "title": "HR Detail",
        "fields": ["Headcount", "Recruiter", "Business Unit", "Cost Center"],
        "category": "Worker Data", "data_source": "Workers", "field_meta": [],
    }]))


def _values(query):
    intent = _parser().parse(query)
    return {concept.canonical for concept in intent.expansion.concepts}


def test_lowercase_common_acronyms_do_not_expand_without_context():
    assert "Business Unit" not in _values("bu report")
    assert "Cost Center" not in _values("cc report")
    assert "Time Away" not in _values("ta report")


def test_uppercase_and_safe_acronyms_expand():
    assert "Business Unit" in _values("BU report")
    assert "Headcount" in _values("hc report")


def test_ta_partner_means_recruiter_not_time_away():
    values = _values("ta partner activity")
    assert "Recruiter" in values
    assert "Time Away" not in values
