"""Independent candidate generators. None of them gates any of the others."""

from __future__ import annotations

from .base import (
    CandidateGenerator,
    CandidateHit,
    GeneratorResult,
    merge_variant_runs,
)
from .dense import VIEW_VARIANTS, DenseViewGenerator
from .late_interaction import (
    DisabledLateInteractionGenerator,
    LateInteractionGenerator,
)
from .lexical import BM25F_VARIANTS, Bm25fGenerator, zone_weights
from .prototype import PrototypeGenerator
from .sparse import SpladeGenerator

__all__ = [
    "BM25F_VARIANTS",
    "VIEW_VARIANTS",
    "Bm25fGenerator",
    "CandidateGenerator",
    "CandidateHit",
    "DenseViewGenerator",
    "DisabledLateInteractionGenerator",
    "GeneratorResult",
    "LateInteractionGenerator",
    "PrototypeGenerator",
    "SpladeGenerator",
    "merge_variant_runs",
    "zone_weights",
]
