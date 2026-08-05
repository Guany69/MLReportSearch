from __future__ import annotations

from dataclasses import dataclass

DEFAULT_LABEL_WEIGHTS = {"adjudicated": 1.0, "human": 1.0, "synthetic-seed": 0.6,
                         "unjudged": 0.0}


@dataclass(frozen=True)
class ResolvedLabel:
    grade: int | None
    source: str
    weight: float


def resolve_label(*, adjudicated: int | None = None, human: int | None = None,
                  synthetic_seed: int | None = None,
                  weights: dict[str, float] | None = None) -> ResolvedLabel:
    configured = DEFAULT_LABEL_WEIGHTS | (weights or {})
    for source, value in (("adjudicated", adjudicated), ("human", human),
                          ("synthetic-seed", synthetic_seed)):
        if value is not None:
            if isinstance(value, bool) or int(value) not in {0, 1, 2, 3}:
                raise ValueError(f"Invalid relevance grade for {source}: {value!r}")
            return ResolvedLabel(int(value), source, float(configured[source]))
    return ResolvedLabel(None, "unjudged", float(configured["unjudged"]))
