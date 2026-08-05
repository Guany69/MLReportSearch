"""Typed, validated access to the versioned relevance dataset."""

from .labels import ResolvedLabel, resolve_label
from .loaders import RelevanceData, RelevanceDataLoader
from .splits import SplitGuard, SplitLeakageError

__all__ = ["RelevanceData", "RelevanceDataLoader", "ResolvedLabel", "SplitGuard",
           "SplitLeakageError", "resolve_label"]
