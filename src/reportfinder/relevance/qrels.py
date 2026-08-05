from __future__ import annotations

from dataclasses import dataclass

from .labels import resolve_label


@dataclass(frozen=True)
class RelevanceQrel:
    scenario_id: str
    report_key: str
    grade: int
    label_source: str
    label_weight: float
    scenario_family: str = ""
    difficulty: str = ""
    intent_count: int = 1
    answerability: str = ""
    edge_case_tags: tuple[str, ...] = ()


def build_qrels(scenarios, judgments, splits=None) -> tuple[RelevanceQrel, ...]:
    scenario_map = {s.scenario_id: s for s in scenarios}; output = []
    for judgment in judgments:
        label = resolve_label(adjudicated=judgment.adjudicated_grade,
                              human=judgment.human_grade,
                              synthetic_seed=judgment.synthetic_grade)
        if label.grade is None:
            continue
        scenario = scenario_map[judgment.scenario_id]
        output.append(RelevanceQrel(judgment.scenario_id, judgment.report_key, label.grade,
            label.source, label.weight, scenario.scenario_family, scenario.difficulty,
            scenario.intent_count, scenario.answerability, scenario.edge_case_tags))
    return tuple(sorted(output, key=lambda q: (q.scenario_id, q.report_key)))
