from __future__ import annotations

import pandas as pd

from reportfinder.query.intent import IntentParser


def _parser():
    row = {
        "title": "HR Detail",
        "fields": [
            "Attrition", "Manager", "Headcount", "Deductions", "Pay Period",
            "Net Pay", "Learning Assignment",
        ],
        "category": "Worker Data", "data_source": "Workers", "field_meta": [],
    }
    return IntentParser(pd.DataFrame([row.copy() for _ in range(3)]))


def test_expected_domain_typos_are_corrected():
    parser = _parser()
    expected = {
        "attriton": "attrition", "manger": "manager", "headcout": "headcount",
        "deductons": "deductions", "perod": "period", "hom": "home",
    }
    for surface, corrected in expected.items():
        result = parser.parse(surface if surface != "hom" else "take hom pay")
        assert any(item.surface.casefold() == surface and item.corrected == corrected
                   for item in result.expansion.typos)


def test_valid_words_do_not_drift():
    parser = _parser()
    assert not parser.parse("training").expansion.typos
    assert not parser.parse("earnings").expansion.typos
