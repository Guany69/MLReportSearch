"""Blinded annotation export/import helpers; source templates are never modified."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

HIDDEN_COLUMNS = frozenset({"candidate_source", "model_score", "rank",
                            "synthetic_label_status", "system_recommendation"})


def create_blinded_assignments(candidate_pool: pd.DataFrame, reviewers: list[str]) -> pd.DataFrame:
    if not reviewers: raise ValueError("at least one reviewer is required")
    required = {"scenario_id", "report_key"}
    if not required <= set(candidate_pool): raise ValueError(f"candidate pool missing {sorted(required-set(candidate_pool))}")
    visible = [c for c in candidate_pool.columns if c.casefold() not in HIDDEN_COLUMNS]
    chunks = []
    for offset, (_, row) in enumerate(candidate_pool.sort_values(["scenario_id", "report_key"])[visible].iterrows()):
        item = row.to_dict(); item.update({"reviewer": reviewers[offset % len(reviewers)],
                                          "relevance_grade": pd.NA, "reviewer_notes": ""})
        chunks.append(item)
    return pd.DataFrame(chunks)


def write_revision(frame: pd.DataFrame, root: str | Path, version: str | None = None) -> Path:
    version = version or datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    directory = Path(root) / "annotation/revisions" / version
    directory.mkdir(parents=True, exist_ok=False)
    path = directory / "judgments.csv"; frame.to_csv(path, index=False)
    return path
