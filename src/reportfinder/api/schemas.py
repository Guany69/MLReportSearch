"""Wire contracts for the HTTP service.

Pydantic lives only in this package. Everything below the API boundary uses stdlib
frozen dataclasses, which is the repository's convention; models here exist because
FastAPI needs them for validation and OpenAPI, not as a domain representation.

Two rules the shapes enforce structurally rather than by filtering:

**Sensitive columns are absent, not removed.** `owner`, `created_by`, `last_run_by`
and the rest of `FORBIDDEN_VIEW_SOURCES` have no field here at all. A response model
that *could* carry them and relies on a filter to strip them is one refactor away
from leaking; a model with no such field cannot.

**No score is called a probability.** The reranker emits raw logits, no calibration
artifact ships, and the decision fallback is a deterministic policy. So the score
field is named `unnormalized_ranking_score` and `decision_calibrated` is always
reported alongside it.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

Decision = Literal["RETURN_RESULTS", "ASK_CLARIFICATION", "NO_CONFIDENT_MATCH"]


class SearchRequestBody(BaseModel):
    """What a client may ask for.

    Note what is *not* here: any identity field. The principal is resolved
    server-side from the transport credential. Accepting a caller-supplied user id
    would make every entitlement check advisory.
    """

    query: str = Field(min_length=1, max_length=2000)
    top_k: int | None = Field(default=None, ge=1, le=50)
    clarification_context: str | None = Field(default=None, max_length=2000)


class GroundedEvidence(BaseModel):
    """Why this report was returned, in catalog terms a user can check."""

    matched_title_concepts: list[str] = []
    matched_aliases: list[str] = []
    matched_fields: list[str] = []
    matched_prompts: list[str] = []
    category: str = ""
    data_source: str = ""
    report_type: str = ""
    retriever_support: dict[str, int] = {}
    admitted_via: str = ""
    # True when this instance reached the answer through its family rather than
    # through any retriever.
    via_family_expansion: bool = False


class SelectedInstance(BaseModel):
    """The concrete row to run."""

    report_instance_id: str
    title: str
    fields: list[str] = []
    prompts: list[str] = []
    data_source: str = ""
    report_type: str = ""
    # How many authorized instances this family has, so a client can offer the
    # alternatives without a second request.
    sibling_count: int = 0


class SearchResultItem(BaseModel):
    rank: int
    family_id: str
    title: str
    # Deliberately not "score" or "confidence": an uncalibrated cross-encoder logit.
    unnormalized_ranking_score: float | None = None
    selected_instance: SelectedInstance
    grounded_evidence: GroundedEvidence


class Clarification(BaseModel):
    question: str
    options: list[str] = []


class SearchResponse(BaseModel):
    request_id: str
    decision: Decision
    results: list[SearchResultItem] = []
    clarification: Clarification | None = None
    warnings: list[str] = []
    catalog_version: str
    model_bundle_version: str
    latency_ms: float
    # Disclosed on every response. A fallback the client cannot see is a fallback
    # that gets mistaken for a trained model.
    active_fallbacks: list[str] = []
    decision_calibrated: bool = False


class FeedbackBody(BaseModel):
    """One impression.

    This is behavioural data, **not** a relevance label. Clicks are confounded by
    position, presentation and the slate the ranker chose; recording the slate and
    the positions is what makes any later de-biasing possible at all.
    """

    request_id: str
    idempotency_key: str = Field(min_length=1, max_length=200)
    slate: list[str] = Field(default=[], description="family ids, in shown order")
    selected_family_id: str | None = None
    dismissed_family_ids: list[str] = []
    reformulated_to: str | None = Field(default=None, max_length=2000)
    explicit_judgement: Literal["relevant", "not_relevant"] | None = None
    exploration_metadata: dict[str, float] = {}


class FeedbackResponse(BaseModel):
    recorded: bool
    idempotency_key: str
    # False when this key was already recorded; the write is a no-op, not an error.
    duplicate: bool = False


class ComponentInfo(BaseModel):
    status: str
    requirement: str
    fallback: str | None = None
    model_id: str = ""
    model_revision: str = ""


class ModelInfoResponse(BaseModel):
    catalog_version: str
    model_bundle_version: str
    code_version: str
    retrieval_mode: str
    components: dict[str, ComponentInfo] = {}
    active_fallbacks: list[str] = []
    fusion_trained: bool = False
    decision_trained: bool = False
    decision_calibrated: bool = False
    authorization_resolver: str = ""
    authorization_is_development_default: bool = True
    # The bundle id covers corpus content and index config only, so it cannot
    # distinguish two runs that retrieve differently. These can.
    build_config_hash: str = ""
    runtime_config_hash: str = ""
    config_drift: bool = False


class HealthResponse(BaseModel):
    status: Literal["ok", "not_ready"]
    detail: str = ""
    blocking_components: list[str] = []
