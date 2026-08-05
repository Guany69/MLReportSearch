from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import pandas as pd

from .schema import (
    ALLOWED_ANSWERABILITY,
    ALLOWED_GRADES,
    CANDIDATE_COLUMNS,
    FEATURE_VERSION,
    REPORT_KEY_RE,
    SCENARIO_COLUMNS,
    SCENARIO_RE,
    SEED_COLUMNS,
    DataContractError,
    Judgment,
    Scenario,
)


def _required(frame: pd.DataFrame, columns: Iterable[str], path: Path) -> None:
    missing = [name for name in columns if name not in frame.columns]
    if missing:
        raise DataContractError(f"{path}: missing required columns {missing}")


def _nonnull(value, *, path: Path, row: int, field: str):
    if pd.isna(value) or str(value).strip() == "":
        raise DataContractError(f"{path}: row {row}, field {field}: value is required")
    return value


@dataclass(frozen=True)
class RelevanceData:
    scenarios: tuple[Scenario, ...]
    judgments: tuple[Judgment, ...]
    splits: dict[str, frozenset[str]]


class RelevanceDataLoader:
    def __init__(self, root: str | Path):
        self.root = Path(root)

    def load_scenarios(self, path: str | Path | None = None) -> tuple[Scenario, ...]:
        path = Path(path) if path else self.root / "raw/axis_report_search_scenarios_10000.csv"
        frame = pd.read_csv(path, keep_default_na=True)
        _required(frame, SCENARIO_COLUMNS, path)
        seen: set[str] = set(); result = []
        for offset, row in frame.iterrows():
            line = int(offset) + 2
            sid = str(_nonnull(row["Scenario ID"], path=path, row=line, field="Scenario ID"))
            if not SCENARIO_RE.fullmatch(sid):
                raise DataContractError(f"{path}: row {line}, field Scenario ID: malformed {sid!r}")
            if sid in seen:
                raise DataContractError(f"{path}: row {line}: duplicate scenario ID {sid}")
            seen.add(sid)
            answer = str(_nonnull(row["Answerability"], path=path, row=line, field="Answerability"))
            if answer not in ALLOWED_ANSWERABILITY:
                raise DataContractError(f"{path}: row {line}, field Answerability: {answer!r}")
            result.append(Scenario(sid, str(row["Employee Search Query"]),
                str(row["Scenario Family"]), str(row["Difficulty"]), answer,
                int(row["Intent Count"]), tuple(str(row.get("Edge Case Tags", "")).split(";"))))
        return tuple(sorted(result, key=lambda item: item.scenario_id))

    def load_seed_judgments(self, path: str | Path | None = None,
                            catalog=None) -> tuple[Judgment, ...]:
        path = Path(path) if path else self.root / "raw/axis_report_search_seed_judgments.jsonl"
        rows = []
        with path.open(encoding="utf-8") as handle:
            for line, text in enumerate(handle, 1):
                try: rows.append(json.loads(text))
                except Exception as exc: raise DataContractError(f"{path}: row {line}: invalid JSON: {exc}") from exc
        frame = pd.DataFrame(rows); _required(frame, SEED_COLUMNS, path)
        catalog_map = None
        if catalog is not None:
            iterable = catalog.to_dict("records") if hasattr(catalog, "to_dict") else catalog
            catalog_map = {str(r.get("report_key") if isinstance(r, dict) else r.report_key):
                           str(r.get("title") if isinstance(r, dict) else r.report_name) for r in iterable}
        seen = set(); result = []
        for offset, row in frame.iterrows():
            line = int(offset) + 1; sid = str(row["Scenario ID"]); key = str(row["Report Key"])
            if not SCENARIO_RE.fullmatch(sid): raise DataContractError(f"{path}: row {line}: malformed scenario {sid!r}")
            if not REPORT_KEY_RE.fullmatch(key): raise DataContractError(f"{path}: row {line}: malformed report key {key!r}")
            if (sid, key) in seen: raise DataContractError(f"{path}: row {line}: duplicate candidate pair {(sid, key)}")
            seen.add((sid, key)); grade = int(row["Synthetic Relevance Grade"])
            if grade not in ALLOWED_GRADES: raise DataContractError(f"{path}: row {line}: invalid grade {grade}")
            title = str(row["Report Title"])
            if catalog_map is not None:
                if key not in catalog_map: raise DataContractError(f"{path}: row {line}: unknown report key {key}")
                if catalog_map[key].strip() != title.strip(): raise DataContractError(f"{path}: row {line}: title mismatch for {key}: {title!r} != {catalog_map[key]!r}")
            result.append(Judgment(sid, key, title, synthetic_grade=grade))
        return tuple(sorted(result, key=lambda item: (item.scenario_id, item.report_key)))

    def load_splits(self) -> dict[str, frozenset[str]]:
        splits = {name: frozenset(line.strip() for line in (self.root / f"splits/{name}_queries.txt").read_text().splitlines() if line.strip()) for name in ("train", "validation", "test")}
        for name, ids in splits.items():
            malformed = sorted(s for s in ids if not SCENARIO_RE.fullmatch(s))
            if malformed: raise DataContractError(f"{name} split contains malformed IDs: {malformed[:10]}")
        names = list(splits)
        for i, left in enumerate(names):
            for right in names[i + 1:]:
                overlap = splits[left] & splits[right]
                if overlap: raise DataContractError(f"split overlap {left}/{right}: {sorted(overlap)[:10]}")
        return splits

    def validate_processed(self) -> None:
        for name in ("ranking_features.parquet", "answerability_features.parquet"):
            path = self.root / "processed" / name; frame = pd.read_parquet(path)
            _required(frame, (*CANDIDATE_COLUMNS, "feature_version") if name.startswith("ranking") else ("scenario_id", "feature_version"), path)
            versions = set(frame["feature_version"].dropna().astype(str))
            if versions != {FEATURE_VERSION}: raise DataContractError(f"{path}: feature_version mismatch: {sorted(versions)}")

    def load(self, *, catalog=None, validate_processed: bool = True) -> RelevanceData:
        scenarios = self.load_scenarios(); judgments = self.load_seed_judgments(catalog=catalog); splits = self.load_splits()
        scenario_ids = {s.scenario_id for s in scenarios}
        if set().union(*splits.values()) != scenario_ids: raise DataContractError("split union does not equal scenario IDs")
        unknown = sorted({j.scenario_id for j in judgments} - scenario_ids)
        if unknown: raise DataContractError(f"judgments contain unknown scenarios: {unknown[:10]}")
        if validate_processed: self.validate_processed()
        return RelevanceData(scenarios, judgments, splits)
