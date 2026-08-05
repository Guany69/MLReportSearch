"""Human-judgment workflow and conservative production acceptance gates."""
from __future__ import annotations

import hashlib
import json
import random
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Judgment:
    query_id: str
    query: str
    report_id: str
    relevance: int                 # 0=not relevant, 1=useful, 2=best answer
    answerable: bool
    annotator_id: str
    query_timestamp: str
    source: str = "production_query"
    notes: str = ""

def load_judgments(path) -> list[Judgment]:
    with Path(path).open() as stream:
        return [Judgment(**json.loads(line)) for line in stream if line.strip()]

def validate_judgments(items: list[Judgment]) -> None:
    if not items: raise ValueError("No human judgments supplied")
    for item in items:
        if item.relevance not in (0, 1, 2): raise ValueError(f"Invalid relevance for {item.query_id}")
        if not item.annotator_id.strip(): raise ValueError("annotator_id is required")
        if item.source != "production_query": raise ValueError("Production gates accept only production_query judgments")

def query_split(query_id: str, *, seed: str = "reportfinder-v1") -> str:
    """Stable query-level split: related candidate judgments cannot leak."""
    bucket = int(hashlib.sha256(f"{seed}:{query_id}".encode()).hexdigest()[:8], 16) % 100
    return "train" if bucket < 70 else "validation" if bucket < 85 else "test"

def agreement(items: list[Judgment]) -> float:
    """Pairwise exact agreement on relevance for double-annotated examples."""
    groups: dict[tuple[str, str], list[int]] = {}
    for j in items: groups.setdefault((j.query_id, j.report_id), []).append(j.relevance)
    pairs = [(a,b) for values in groups.values() for i,a in enumerate(values) for b in values[i+1:]]
    return sum(a == b for a,b in pairs)/len(pairs) if pairs else 0.0

def acceptance_gate(metrics: dict, items: list[Judgment], *, min_queries: int = 500,
                    min_agreement: float = .75) -> tuple[bool, list[str]]:
    """Block a production claim unless independent evidence clears every gate."""
    validate_judgments(items)
    reasons=[]; query_count=len({j.query_id for j in items})
    if query_count < min_queries: reasons.append(f"need {min_queries} human queries; have {query_count}")
    measured_agreement=agreement(items)
    if measured_agreement < min_agreement: reasons.append(f"annotator agreement {measured_agreement:.3f} < {min_agreement:.3f}")
    thresholds={"ndcg@10":.75,"mrr@10":.70,"no_answer_precision":.90,"no_answer_recall":.80,"ece":.08}
    for name, threshold in thresholds.items():
        value=metrics.get(name)
        if value is None: reasons.append(f"missing metric {name}")
        elif name == "ece" and value > threshold: reasons.append(f"{name} {value:.3f} > {threshold:.3f}")
        elif name != "ece" and value < threshold: reasons.append(f"{name} {value:.3f} < {threshold:.3f}")
    return not reasons, reasons

def export_annotation_pool(finder, queries: list[dict], path, *, k: int = 10) -> None:
    """Create blinded candidate rows for two-pass annotation."""
    rows=[]
    for item in queries:
        result=finder.query(item["query"], top_k=k)
        for rank,candidate in enumerate(result.candidates,1):
            rows.append({"query_id":item["query_id"],"query":item["query"],
                         "report_id":str(candidate.row.get("report_id",candidate.index)),
                         "title":candidate.row["title"],"fields":list(candidate.row["fields"]),
                         "candidate_rank":rank,"relevance":None,"answerable":None,
                         "annotator_id":"","query_timestamp":item["query_timestamp"],
                         "source":"production_query","notes":""})
    random.Random(0).shuffle(rows)
    with Path(path).open("w") as stream:
        for row in rows: stream.write(json.dumps(row,sort_keys=True)+"\n")
