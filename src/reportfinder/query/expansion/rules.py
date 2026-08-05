"""Typed YAML rule loader with deterministic provenance hashing."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import yaml

from .types import CONCEPT_KINDS, ConceptKind


def _concept_kind(value: object, *, rule_id: str, path: Path) -> ConceptKind:
    """Validate a YAML `kind` against the five the pipeline understands.

    Previously this was `str(...)` and trusted. A misspelled kind then reached
    `ExpansionConcept.kind`, which is a Literal, and produced a concept that no
    vocabulary lookup could ever resolve -- a silent dead rule rather than an
    error at load time.
    """
    kind = str(value)
    if kind not in CONCEPT_KINDS:
        raise ValueError(
            f"lexicon rule {rule_id!r} in {path.name} emits unknown kind {kind!r}; "
            f"expected one of {sorted(CONCEPT_KINDS)}"
        )
    return cast(ConceptKind, kind)


@dataclass(frozen=True)
class Emission:
    canonical: str
    kind: ConceptKind
    confidence: float


@dataclass(frozen=True)
class Rule:
    id: str
    phrases: tuple[str, ...]
    emits: tuple[Emission, ...]
    priority: int = 0
    requires_any: tuple[str, ...] = ()
    requires_none: tuple[str, ...] = ()
    window: int = 0
    suppresses: tuple[str, ...] = ()


@dataclass(frozen=True)
class Lexicon:
    rules: tuple[Rule, ...]
    hash: str
    dead_rules: tuple[str, ...]


def lexicon_content_hash(directory: Path | None = None) -> str:
    root = directory or Path(__file__).resolve().parents[1] / "lexicon"
    digest = hashlib.sha256()
    for path in sorted(item for item in root.glob("*.yaml") if item.name != "_schema.yaml"):
        digest.update(path.name.encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()[:16]


def load_lexicon(vocabulary, directory: Path | None = None) -> Lexicon:
    root = directory or Path(__file__).resolve().parents[1] / "lexicon"
    files = sorted(path for path in root.glob("*.yaml") if path.name != "_schema.yaml")
    digest = hashlib.sha256()
    rules = []
    dead = []
    ids = set()
    for path in files:
        content = path.read_bytes()
        digest.update(path.name.encode())
        digest.update(content)
        doc = yaml.safe_load(content) or {}
        if doc.get("schema", 1) != 1:
            raise ValueError(f"Unsupported lexicon schema in {path}")
        for raw in doc.get("rules", ()):
            rule_id = str(raw["id"])
            if rule_id in ids:
                raise ValueError(f"Duplicate lexicon rule id: {rule_id}")
            ids.add(rule_id)
            emissions = []
            for emission in raw.get("emits", ()):
                kind = _concept_kind(emission["kind"], rule_id=rule_id, path=path)
                canonical = vocabulary.canonical(kind, str(emission["canonical"]))
                if canonical is None:
                    dead.append(f"{rule_id}:{emission['canonical']}")
                    continue
                emissions.append(Emission(canonical, kind, float(emission.get("confidence", .8))))
            if not emissions:
                continue
            context = raw.get("context") or {}
            rules.append(Rule(
                rule_id, tuple(map(str, raw.get("phrases", ()))),
                tuple(emissions), int(raw.get("priority", 0)),
                tuple(map(str.casefold, context.get("requires_any", ()))),
                tuple(map(str.casefold, context.get("requires_none", ()))),
                int(context.get("window", 0) or 0),
                tuple(map(str, raw.get("suppresses", ()))),
            ))
    forbidden = {"quantum", "submarine", "telemetry"}
    if any(forbidden & set(phrase.casefold().split()) for rule in rules for phrase in rule.phrases):
        raise ValueError("Lexicon may not ground the reserved synthetic-negative vocabulary")
    conflicts: dict[tuple[str, int], tuple[str, ...]] = {}
    for rule in rules:
        for phrase in rule.phrases:
            key = (phrase.casefold(), rule.priority)
            targets = tuple(sorted(value.casefold() for value in rule.suppresses))
            previous = conflicts.setdefault(key, targets)
            if previous != targets:
                raise ValueError(
                    f"Phrase {phrase!r} has conflicting suppressions at priority {rule.priority}"
                )
    graph = {
        emission.canonical.casefold(): {target.casefold() for target in rule.suppresses}
        for rule in rules for emission in rule.emits if rule.suppresses
    }
    visiting: set[str] = set()
    visited: set[str] = set()

    def visit(node: str) -> None:
        if node in visiting:
            raise ValueError(f"Cycle in lexicon suppression graph at {node!r}")
        if node in visited:
            return
        visiting.add(node)
        for child in graph.get(node, ()):
            visit(child)
        visiting.remove(node)
        visited.add(node)

    for node in graph:
        visit(node)
    return Lexicon(tuple(rules), digest.hexdigest()[:16], tuple(dead))
