"""The corpus that reproduces the original recall failure.

Twelve instances, held in code rather than in a workbook so the regression runs in
well under a second and needs no model download.

The shape that matters:

* **R0101** is the target. Its title -- "Attrition and Replacement Lag Analysis" --
  shares no content token with the query "why are we losing people faster than we
  can backfill". What makes it the right answer lives in its *fields* and its
  *description*, which is exactly the evidence a lexical retriever cannot use.
* **R0103** "Backfill Request Log" is the lexical decoy. It matches the query's one
  distinctive word literally, ranks highly on BM25F, and is the wrong answer.
* Filler instances span five families, two of which have several instances, so
  family aggregation and best-instance selection are genuinely exercised.
"""

from __future__ import annotations

import pandas as pd

TARGET = "R0101"
LEXICAL_DECOY = "R0103"
QUERY = "why are we losing people faster than we can backfill"
EXACT_TITLE_QUERY = "Payroll Earnings Detail"

# (instance_id, title, fields, description, category)
INSTANCES = [
    # Deliberately disjoint from the query's vocabulary *and* from what the
    # lexicon expands it to. The expansion engine turns "people" into the canonical
    # field "Worker", so the target must not carry that word in any zone either --
    # otherwise BM25F reaches it through the description and the premise of this
    # regression quietly stops holding.
    (TARGET, "Attrition and Replacement Lag Analysis",
     ["Termination Reason", "Time to Fill", "Headcount", "Hire Date"],
     "Measures the interval between an exit event and the arrival of a successor.",
     "Talent"),
    ("R0102", "People Directory Export",
     ["Worker", "Email", "Location"],
     "Directory listing of workers.", "Worker Data"),
    (LEXICAL_DECOY, "Backfill Request Log",
     ["Requisition Status", "Requisition ID"],
     "Log of open backfill requisitions.", "Recruiting"),
    ("R0104", "Fast Track Promotion List",
     ["Promotion Date", "Job Profile"],
     "Workers on accelerated promotion tracks.", "Talent"),
    ("R0105", "Payroll Earnings Detail",
     ["Net Pay", "Pay Period", "Gross Pay"],
     "Earnings by pay period.", "Payroll"),
    ("R0106", "Payroll Earnings Detail",
     ["Net Pay", "Pay Period", "Deductions", "Cost Center"],
     "Earnings by pay period with cost centre detail.", "Payroll"),
    ("R0107", "Learning Course Completion",
     ["Learning Assignment", "Due Date"],
     "Course completion status.", "Learning"),
    ("R0108", "Worker Location Roster",
     ["Location", "Worker", "Company"],
     "Where each worker is based.", "Worker Data"),
    ("R0109", "Compensation Band Review",
     ["Salary Range", "Job Profile"],
     "Salary ranges by job profile.", "Compensation"),
    ("R0110", "Worker Location Roster",
     ["Location", "Worker", "Region"],
     "Where each worker is based, by region.", "Worker Data"),
    ("R0111", "Succession Readiness Summary",
     ["Succession Candidate", "Succession Readiness"],
     "Succession pipeline readiness.", "Talent"),
    ("R0112", "Time Off Balance Detail",
     ["Time Off Balance", "Worker"],
     "Remaining time off by worker.", "Absence"),
]


def frame() -> pd.DataFrame:
    """The fixture as a row-level corpus frame."""
    return pd.DataFrame([
        {
            "report_key": instance_id,
            "title_key": title.casefold(),
            "source_row": 100 + position,
            "title": title,
            "description": description,
            "category": category,
            "data_source": "All Workers",
            "report_type": "Advanced",
            "prompts": ["Effective Date"],
            "fields": fields,
            "tags": "",
            "area_where_used": "",
            "worklet": "Standard",
            "chart_type": "Bar",
            "landing_page": "People",
            "worklet_landing_pages": "",
            "shared": "Yes",
            "field_meta": [],
        }
        for position, (instance_id, title, fields, description, category)
        in enumerate(INSTANCES)
    ])


def decoy_ids(count: int) -> list[str]:
    """Synthetic ids used to bury the target under better-fused candidates."""
    return [f"D{i:03d}" for i in range(count)]
