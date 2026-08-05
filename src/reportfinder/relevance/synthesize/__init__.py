"""Generative synthetic label generation (v2).

v1's labels were made by *matching* generated queries against report text -- the
same operation BM25F performs -- so measuring lexical retrieval against them was
circular, and zero of its queries graded two instances of one family, which made
family expansion and instance selection unmeasurable.

v2 builds each query *from* a chosen target's metadata instead, so the grade is
true by construction and no retriever was consulted to decide it.

    uv run python scripts/generate_labels_v2.py --out data/relevance_v2 -v
"""

from __future__ import annotations

from .archetypes import Budget, GeneratedScenario
from .generate import GENERATOR_VERSION, build_scenarios, generate, main

# `validate` the *function* is deliberately not re-exported: it would shadow
# `synthesize.validate` the module, so `from ... import validate` would silently
# hand you a function where a module was meant.
from .validate import ValidationError, ValidationReport
from .validate import validate as validate_scenarios

__all__ = [
    "GENERATOR_VERSION",
    "Budget",
    "GeneratedScenario",
    "ValidationError",
    "ValidationReport",
    "build_scenarios",
    "generate",
    "main",
    "validate_scenarios",
]
