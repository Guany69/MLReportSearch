from __future__ import annotations

import pandas as pd

from reportfinder.query.intent import IntentParser


def _frame():
    fields = [
        "Organization", "Supervisory Organization", "Start Date", "Hire Date",
        "Manager", "Previous Manager", "Period", "Pay Period", "Worker",
    ]
    return pd.DataFrame([{
        "title": "Worker Detail", "fields": fields, "category": "Worker Data",
        "data_source": "Workers", "field_meta": [],
    }])


def _fields(query):
    intent = IntentParser(_frame()).parse(query)
    return {field.value for field in intent.fields}, {
        field.canonical for field in intent.expansion.suppressed
    }


def test_grouping_organization_uses_supervisory_organization():
    fields, suppressed = _fields("grouped by organization headcount")
    assert "Supervisory Organization" in fields
    assert "Organization" not in fields
    assert "Organization" in suppressed


def test_hire_context_replaces_start_date():
    fields, suppressed = _fields("recent starters with start date")
    assert "Hire Date" in fields
    assert "Start Date" not in fields
    assert "Start Date" in suppressed


def test_previous_boss_replaces_manager():
    fields, suppressed = _fields("prior boss for each worker")
    assert "Previous Manager" in fields
    assert "Manager" not in fields
    assert "Manager" in suppressed


def test_pay_period_structurally_suppresses_period():
    fields, suppressed = _fields("show Pay Period")
    assert "Pay Period" in fields
    assert "Period" not in fields
    assert "Period" in suppressed
