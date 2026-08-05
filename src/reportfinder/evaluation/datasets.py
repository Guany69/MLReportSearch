"""Deterministic weak-query generation from report metadata."""
from __future__ import annotations

from collections import defaultdict

from reportfinder.relevance.loaders import RelevanceDataLoader
from reportfinder.relevance.splits import SplitGuard

from .qrels import Qrel


def generate(frame, limit: int = 100) -> list[Qrel]:
    """Weak qrels keyed on the most stable identifier the frame offers.

    `report_key` (R####) is the physical catalog row and is stable across corpus
    builds; `report_id` is a positional index that silently points at a different
    report if granularity or ordering changes -- which made a stored qrels file
    quietly wrong rather than obviously broken.
    """
    head = frame.head(limit)

    # A title query is answered by *every* instance carrying that title. 533 titles
    # on this estate appear on more than one row, so keying a title qrel to a
    # single instance marks its own duplicates wrong -- the ranker is then
    # penalised for returning an equally correct answer.
    by_title: dict[str, list[str]] = defaultdict(list)
    for i, row in head.iterrows():
        rid = str(row.get("report_key") or row.get("report_id", i))
        by_title[str(row["title"]).casefold()].append(rid)

    out = []
    for i, row in head.iterrows():
        rid = str(row.get("report_key") or row.get("report_id", i))
        title = str(row["title"])
        siblings = by_title[title.casefold()]
        out.append(Qrel(
            f"title-{i}", title, dict.fromkeys(siblings, 2), True, "title", "weak_title",
        ))
        if row["fields"]:
            out.append(Qrel(f"field-{i}", f"report with {row['fields'][0]}", {rid: 1}, True, "field_name", "weak_field"))
    out.extend([Qrel("noanswer-1", "quantum submarine telemetry", {}, False, "no_answer", "synthetic_negative")])
    return out


def load_weak_qrels(root, split: str) -> list[Qrel]:
    """Bridge the validated R#### weak-label bundle into benchmark qrels."""
    if split not in {"train", "validation", "test"}:
        raise ValueError(f"Unknown weak-label split: {split}")
    data = RelevanceDataLoader(root).load(validate_processed=False)
    SplitGuard(data.splits)  # validates disjoint split membership eagerly
    scenarios = {scenario.scenario_id: scenario for scenario in data.scenarios}
    judgments: defaultdict[str, dict[str, int]] = defaultdict(dict)
    for judgment in data.judgments:
        grade = judgment.synthetic_grade or 0
        if judgment.scenario_id in data.splits[split] and grade > 0:
            judgments[judgment.scenario_id][judgment.report_key] = grade
    result = []
    for scenario_id in sorted(data.splits[split]):
        scenario = scenarios[scenario_id]
        answerable = scenario.answerability not in {"No answer"}
        result.append(Qrel(
            query_id=scenario_id, query=scenario.query_text,
            relevant=dict(sorted(judgments.get(scenario_id, {}).items())),
            answerable=answerable, slice=scenario.scenario_family,
            label_source="weak_synthetic_bundle",
            expected_status=(
                "NO_SATISFACTORY_REPORT" if not answerable
                else "NEEDS_CLARIFICATION" if "clarification" in scenario.answerability.casefold()
                else None
            ),
        ))
    return result
