from pathlib import Path

import pandas as pd

from reportfinder.evaluation.qrels import Qrel, load_qrels, write_qrels


def test_predicate_qrel_resolves_multiple_reports_and_explicit_override():
    frame = pd.DataFrame([
        {"report_id": 2, "title": "Worker Termination Detail", "fields": ["Termination Reason"]},
        {"report_id": 1, "title": "Termination Audit", "fields": ["Termination Reason"]},
        {"report_id": 3, "title": "Worker Roster", "fields": ["Worker"]},
    ])
    qrel = Qrel(
        "q", "why did people leave", {"2": 3},
        required_fields=("Termination Reason",),
        title_terms_any=("termination",),
        predicate_grade=2,
    )
    assert qrel.resolve_relevant(frame) == {"1": 2, "2": 3}


def test_predicate_qrel_resolution_is_order_independent():
    frame = pd.DataFrame([
        {"report_id": "a", "title": "Headcount by Company", "fields": ["Headcount"]},
        {"report_id": "b", "title": "Headcount by Manager", "fields": ["Headcount"]},
    ])
    qrel = Qrel("q", "team size", required_fields=("Headcount",))
    assert qrel.resolve_relevant(frame) == qrel.resolve_relevant(frame.iloc[::-1])


def test_category_and_excluded_field_predicates():
    frame = pd.DataFrame([
        {"report_id": "a", "title": "Pay Detail", "category": "Payroll",
         "fields": ["Gross Pay"]},
        {"report_id": "b", "title": "Pay Detail", "category": "Payroll",
         "fields": ["Gross Pay", "Termination Date"]},
        {"report_id": "c", "title": "Pay Detail", "category": "Benefits",
         "fields": ["Gross Pay"]},
    ])
    qrel = Qrel(
        "q", "payroll", categories=("Payroll",),
        excluded_fields=("Termination Date",),
    )
    assert qrel.resolve_relevant(frame) == {"a": 2}


def test_qrel_json_round_trip(tmp_path: Path):
    source = [Qrel(
        "q", "query", {"a": 3}, expected_status="ANSWERABLE",
        label_source="author_synthetic", predicate_grade=1,
        categories=("Payroll",), excluded_fields=("Termination Date",),
    )]
    path = tmp_path / "qrels.jsonl"
    write_qrels(source, path)
    assert load_qrels(path) == source
