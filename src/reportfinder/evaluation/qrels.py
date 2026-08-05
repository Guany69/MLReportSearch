from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class Qrel:
    query_id: str
    query: str
    relevant: dict[str, int] = field(default_factory=dict)
    answerable: bool = True
    slice: str = "unspecified"
    label_source: str = "weak_synthetic"
    required_fields: tuple[str, ...] = ()
    title_terms_any: tuple[str, ...] = ()
    title_terms_all: tuple[str, ...] = ()
    categories: tuple[str, ...] = ()
    excluded_fields: tuple[str, ...] = ()
    predicate_grade: int = 2
    expected_status: str | None = None
    version: int = 1
    coverage_gap_justification: str = ""

    def resolve_relevant(self, report_frame) -> dict[str, int]:
        """Resolve predicate relevance against the current corpus.

        Explicit judgments win over generated predicate grades. Resolution is
        deterministic and independent of frame ordering.
        """
        resolved: dict[str, int] = {}
        required = {field.casefold() for field in self.required_fields}
        any_terms = {term.casefold() for term in self.title_terms_any}
        all_terms = {term.casefold() for term in self.title_terms_all}
        categories = {category.casefold() for category in self.categories}
        excluded = {field.casefold() for field in self.excluded_fields}
        for index, row in report_frame.iterrows():
            title = str(row.get("title", "")).casefold()
            fields = {str(field).casefold() for field in row.get("fields", ())}
            if required and not required <= fields:
                continue
            if any_terms and not any(term in title for term in any_terms):
                continue
            if all_terms and not all(term in title for term in all_terms):
                continue
            if categories and str(row.get("category", "")).casefold() not in categories:
                continue
            if excluded and excluded & fields:
                continue
            if required or any_terms or all_terms or categories or excluded:
                report_id = str(row.get("report_id", index))
                resolved[report_id] = self.predicate_grade
        resolved.update({str(key): int(grade) for key, grade in self.relevant.items()})
        return dict(sorted(resolved.items()))

def load_qrels(path: str | Path) -> list[Qrel]:
    out = []
    with Path(path).open() as stream:
        for line in stream:
            if line.strip():
                raw = json.loads(line)
                for key in (
                    "required_fields", "title_terms_any", "title_terms_all",
                    "categories", "excluded_fields",
                ):
                    if key in raw:
                        raw[key] = tuple(raw[key])
                out.append(Qrel(**raw))
    return out

def write_qrels(qrels: list[Qrel], path: str | Path) -> None:
    with Path(path).open("w") as stream:
        for item in qrels:
            stream.write(json.dumps(item.__dict__, sort_keys=True) + "\n")
