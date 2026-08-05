"""Inventory label files without treating blank templates as judgments."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import pandas as pd


def csv_inventory(path: Path) -> dict:
    frame = pd.read_csv(path)
    label_columns = [
        column for column in frame.columns
        if "grade" in column.casefold() or "relevance" in column.casefold()
    ]
    nonempty = {
        column: int(frame[column].notna().sum())
        for column in label_columns
    }
    return {
        "rows": len(frame), "columns": list(frame.columns),
        "nonempty_label_cells": nonempty,
    }


def jsonl_inventory(path: Path) -> dict:
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    grades = Counter()
    sources = Counter()
    queries = set()
    for row in rows:
        query_identifier = (
            row.get("query_id") or row.get("Scenario ID")
            or row.get("scenario_id") or row.get("query", "")
        )
        queries.add(str(query_identifier))
        source = (
            row.get("label_source") or row.get("Synthetic Judgment Role")
            or row.get("Review Status") or "unspecified"
        )
        sources[str(source)] += 1
        for grade in (row.get("relevant") or {}).values():
            grades[str(grade)] += 1
        if "grade" in row:
            grades[str(row["grade"])] += 1
        for key in ("Synthetic Relevance Grade", "Human Relevance Grade"):
            if key in row and str(row[key]).strip():
                grades[f"{key}:{row[key]}"] += 1
    return {
        "rows": len(rows), "unique_queries": len(queries),
        "grade_distribution": dict(grades), "label_sources": dict(sources),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="artifacts/label_inventory.json")
    args = parser.parse_args()
    paths = [
        Path("evaluation/qrels.jsonl"),
        Path("data/relevance/raw/axis_report_search_seed_judgments.jsonl"),
        Path("data/relevance/annotation/human_judgments.csv"),
        Path("data/relevance/annotation/adjudicated_judgments.csv"),
        Path("data/relevance/annotation/candidate_pool.csv"),
    ]
    inventory = {}
    for path in paths:
        if not path.exists():
            inventory[str(path)] = {"missing": True}
        elif path.suffix == ".csv":
            inventory[str(path)] = csv_inventory(path)
        else:
            inventory[str(path)] = jsonl_inventory(path)
    inventory["integrity_statement"] = {
        "human_validated": False,
        "production_accuracy_claimed": False,
        "note": "Human and adjudication files are templates unless non-empty grades are reported above.",
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(inventory, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
