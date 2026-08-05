from __future__ import annotations

import re
from dataclasses import dataclass

SCHEMA_VERSION = "1"
FEATURE_VERSION = "1.0.0-baseline"
SCENARIO_RE = re.compile(r"^Q\d{5}$")
REPORT_KEY_RE = re.compile(r"^R\d{4}$")
ALLOWED_GRADES = frozenset({0, 1, 2, 3})
ALLOWED_ANSWERABILITY = frozenset({"Answerable", "Needs clarification", "No answer",
    "Partial / needs clarification", "Partial / missing conversation context"})

SCENARIO_COLUMNS = (
    "Scenario ID", "Employee Search Query", "Scenario Family", "Difficulty",
    "Answerability", "Intent Count",
)
SEED_COLUMNS = (
    "Scenario ID", "Report Key", "Report Title", "Synthetic Relevance Grade",
)
CANDIDATE_COLUMNS = ("scenario_id", "report_key")


class DataContractError(ValueError):
    pass


@dataclass(frozen=True)
class Scenario:
    scenario_id: str
    query_text: str
    scenario_family: str
    difficulty: str
    answerability: str
    intent_count: int
    edge_case_tags: tuple[str, ...] = ()


@dataclass(frozen=True)
class Judgment:
    scenario_id: str
    report_key: str
    report_title: str
    synthetic_grade: int | None = None
    human_grade: int | None = None
    adjudicated_grade: int | None = None
