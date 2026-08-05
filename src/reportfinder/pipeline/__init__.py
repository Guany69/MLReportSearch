"""The search pipeline: preparation, generation, union, risk, shortlist, rerank,
fusion, aggregation and the three-way decision."""

from __future__ import annotations

from .aggregate import (
    ExpansionResult,
    FamilyResult,
    ScoredInstance,
    aggregate_families,
    attach_expanded,
    expand_families,
    instance_suitability,
    score_instances,
)
from .decide import Clarification, Decision, DecisionHead, DecisionResult, decide
from .fuse import (
    FUSION_FEATURE_HASH,
    FUSION_FEATURE_NAMES,
    FusionResult,
    NeuralFuser,
    RrfFuser,
    build_features,
    build_fuser,
)
from .orchestrator import SearchOutcome, SearchPipeline
from .prepare import FacetState, QueryPlan, QueryVariant, build_query_plan
from .rerank import PairScorer, RerankResult, TorchCrossEncoder, rerank_shortlist
from .risk import RecallRiskResult, assess_recall_risk
from .shortlist import Shortlist, ShortlistEntry, select_shortlist
from .telemetry import STAGES, SearchTelemetry
from .union import CandidateUnion, UnionRecord, build_union

__all__ = [
    "FUSION_FEATURE_HASH",
    "FUSION_FEATURE_NAMES",
    "STAGES",
    "CandidateUnion",
    "Clarification",
    "Decision",
    "DecisionHead",
    "DecisionResult",
    "FacetState",
    "FamilyResult",
    "FusionResult",
    "NeuralFuser",
    "PairScorer",
    "QueryPlan",
    "QueryVariant",
    "RecallRiskResult",
    "RerankResult",
    "RrfFuser",
    "ScoredInstance",
    "SearchOutcome",
    "SearchPipeline",
    "SearchTelemetry",
    "Shortlist",
    "ShortlistEntry",
    "TorchCrossEncoder",
    "UnionRecord",
    "ExpansionResult",
    "aggregate_families",
    "attach_expanded",
    "expand_families",
    "instance_suitability",
    "assess_recall_risk",
    "build_features",
    "build_fuser",
    "build_query_plan",
    "build_union",
    "decide",
    "rerank_shortlist",
    "score_instances",
    "select_shortlist",
]
