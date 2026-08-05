"""The HTTP service.

    uv run uvicorn reportfinder.api.service:app --port 8080

Endpoints:

    POST /v1/search       run one authorized search
    POST /v1/feedback     record one impression (not a relevance label)
    GET  /v1/model-info   what is loaded, what is trained, what is falling back
    GET  /health/live     the process is up
    GET  /health/ready    every *required* bundle component is ready

The service is a thin transport over `SearchPipeline`. It adds exactly three
things: identity resolution from the transport, the wire contract, and the
impression log. It must not add ranking behaviour -- anything that changes results
belongs below this boundary where the tests and the bundle manifest can see it.
"""

from __future__ import annotations

import logging
from pathlib import Path

from fastapi import FastAPI, Request, Response, status
from fastapi.responses import JSONResponse

from ..auth import SearchRequest
from .deps import Unauthenticated, get_finder, principal_from_headers
from .feedback import FeedbackStore
from .schemas import Clarification as ClarificationModel
from .schemas import (
    ComponentInfo,
    FeedbackBody,
    FeedbackResponse,
    GroundedEvidence,
    HealthResponse,
    ModelInfoResponse,
    SearchRequestBody,
    SearchResponse,
    SearchResultItem,
    SelectedInstance,
)

log = logging.getLogger(__name__)

app = FastAPI(
    title="Report Finder",
    version="0.2.0",
    description="Find the Workday-style report that answers a plain-English request.",
)


_STORES: dict[str, FeedbackStore] = {}


def _store_for(cfg) -> FeedbackStore:
    path = Path(cfg.cache_dir) / "feedback.jsonl"
    if str(path) not in _STORES:
        _STORES[str(path)] = FeedbackStore(path)
    return _STORES[str(path)]


# -- search -------------------------------------------------------------------


_STOPWORDS = frozenset({
    "the", "and", "for", "with", "from", "that", "this", "are", "was", "our",
    "show", "give", "list", "find", "get", "all", "any", "how", "what", "which",
    "who", "why", "can", "you", "per", "into", "over", "than", "then", "them",
})


def _query_tokens(query: str) -> set[str]:
    out = set()
    for raw in query.replace("/", " ").replace("-", " ").split():
        token = raw.strip(",;:()?.\"'").casefold()
        if len(token) > 2 and token not in _STOPWORDS:
            out.add(token)
    return out


def _matched(values, tokens: set[str]) -> list[str]:
    """The values that share a token with the query.

    Returning *every* field under the name `matched_fields` would be a lie a user
    cannot detect: it looks like evidence and is actually the report's whole
    schema. Only genuine overlap is reported, and an empty list is an honest answer.
    """
    matched = []
    for value in values:
        words = {
            w.strip(",;:()").casefold()
            for w in str(value).replace("/", " ").replace("-", " ").split()
        }
        if words & tokens:
            matched.append(str(value))
    return matched


def _evidence(finder, family, query: str) -> GroundedEvidence:
    """Catalog-grounded reasons, never model reasoning.

    Everything here is a fact a user can check against the catalog. Nothing is a
    narrative about how the model decided, and no hidden chain of thought is
    exposed -- there is none to expose, and inventing one would be worse.
    """
    selected = family.selected
    instance = finder.pipeline.corpus.instance(selected.instance_id)
    record = selected.record
    tokens = _query_tokens(query)
    return GroundedEvidence(
        matched_title_concepts=sorted(tokens & _query_tokens(instance.title)),
        matched_aliases=_matched(family.aliases, tokens),
        matched_fields=_matched(instance.fields, tokens)[:20],
        matched_prompts=_matched(instance.prompts, tokens)[:10],
        category=instance.category,
        data_source=instance.data_source,
        report_type=instance.report_type,
        retriever_support={k: int(v) for k, v in record.ranks.items()},
        admitted_via=selected.admitted_via,
        via_family_expansion=bool(selected.expanded),
    )


def _result_item(finder, family, rank: int, query: str) -> SearchResultItem:
    selected = family.selected
    instance = finder.pipeline.corpus.instance(selected.instance_id)
    return SearchResultItem(
        rank=rank,
        family_id=family.family_id,
        title=family.canonical_title,
        # A raw cross-encoder logit or nothing. Never a probability: no calibration
        # artifact ships and the decision fallback is a deterministic policy.
        unnormalized_ranking_score=selected.cross_encoder_score,
        selected_instance=SelectedInstance(
            report_instance_id=instance.report_instance_id,
            title=instance.title,
            fields=list(instance.fields),
            prompts=list(instance.prompts),
            data_source=instance.data_source,
            report_type=instance.report_type,
            sibling_count=family.instance_count,
        ),
        grounded_evidence=_evidence(finder, family, query),
    )


@app.post("/v1/search", response_model=SearchResponse)
def search(body: SearchRequestBody, request: Request) -> SearchResponse:
    finder, cfg = get_finder()
    principal = principal_from_headers(request.headers)

    outcome = finder.search(SearchRequest(
        raw_query=body.query,
        principal=principal,
        top_k=body.top_k,
        clarification_context=(
            (body.clarification_context,) if body.clarification_context else ()
        ),
    ))

    clarification = None
    if outcome.clarification is not None:
        clarification = ClarificationModel(
            question=outcome.clarification.question,
            options=list(outcome.clarification.options),
        )

    return SearchResponse(
        request_id=outcome.request_id,
        decision=outcome.decision.value,
        # NO_CONFIDENT_MATCH carries no families, so this is empty by construction
        # rather than by a check here.
        results=[
            _result_item(finder, family, rank, body.query)
            for rank, family in enumerate(outcome.families, start=1)
        ],
        clarification=clarification,
        warnings=list(outcome.warnings),
        catalog_version=outcome.catalog_version,
        model_bundle_version=outcome.model_bundle_version,
        latency_ms=round(outcome.latency_ms, 2),
        active_fallbacks=sorted(outcome.active_fallbacks),
        decision_calibrated=False,
    )


# -- feedback -----------------------------------------------------------------


@app.post("/v1/feedback", response_model=FeedbackResponse)
def feedback(body: FeedbackBody, request: Request) -> FeedbackResponse:
    finder, cfg = get_finder()
    principal = principal_from_headers(request.headers)
    manifest = getattr(getattr(finder.pipeline, "bundle", None), "manifest", None)

    stored = _store_for(cfg).record({
        "request_id": body.request_id,
        "idempotency_key": body.idempotency_key,
        # The safe label, not the raw principal: this file outlives the request and
        # is read by more people than the request was.
        "principal": principal.safe_label,
        "slate": list(body.slate),
        "positions": {family: i for i, family in enumerate(body.slate, start=1)},
        "selected_family_id": body.selected_family_id,
        "dismissed_family_ids": list(body.dismissed_family_ids),
        "reformulated_to": body.reformulated_to,
        "explicit_judgement": body.explicit_judgement,
        "exploration_metadata": dict(body.exploration_metadata),
        "catalog_version": manifest.catalog_version if manifest else "",
        "model_bundle_version": manifest.bundle_version if manifest else "",
        "policy_version": cfg.retrieval_mode,
        # Both hashes, so a judgement recorded today can be tied to the exact
        # configuration that produced the slate. `policy_version` is only the
        # coarse retrieval mode, which two materially different runs share.
        "build_config_hash": manifest.config_hash if manifest else "",
        "runtime_config_hash": getattr(
            getattr(finder.pipeline, "bundle", None), "runtime_config_hash", ""
        ),
        # Stated in the record itself so no downstream consumer can mistake this
        # for adjudicated relevance.
        "label_basis": "implicit impression; not a relevance judgement",
    })
    return FeedbackResponse(
        recorded=True, idempotency_key=body.idempotency_key, duplicate=not stored,
    )


# -- model info ---------------------------------------------------------------


@app.get("/v1/model-info", response_model=ModelInfoResponse)
def model_info() -> ModelInfoResponse:
    finder, cfg = get_finder()
    bundle = getattr(finder.pipeline, "bundle", None)
    manifest = getattr(bundle, "manifest", None)

    components = {}
    if manifest is not None:
        for name, record in manifest.components.items():
            components[name] = ComponentInfo(
                status=record.status.value if hasattr(record.status, "value")
                else str(record.status),
                requirement=record.requirement.value
                if hasattr(record.requirement, "value") else str(record.requirement),
                fallback=record.fallback or None,
                model_id=getattr(record, "model_id", "") or "",
                model_revision=getattr(record, "model_revision", "") or "",
            )

    return ModelInfoResponse(
        catalog_version=manifest.catalog_version if manifest else "",
        model_bundle_version=manifest.bundle_version if manifest else "",
        code_version=manifest.code_version if manifest else "",
        retrieval_mode=cfg.retrieval_mode,
        components=components,
        active_fallbacks=list(bundle.active_fallbacks) if bundle else [],
        # Stated flatly. These are the three claims most easily misread as true.
        fusion_trained=cfg.fusion.artifact_path is not None,
        decision_trained=cfg.decision.artifact_path is not None,
        decision_calibrated=cfg.decision.calibration_path is not None,
        authorization_resolver=cfg.auth.resolver,
        authorization_is_development_default=cfg.auth.resolver == "allow_all",
        build_config_hash=manifest.config_hash if manifest else "",
        runtime_config_hash=getattr(bundle, "runtime_config_hash", "") if bundle else "",
        config_drift=bool(getattr(bundle, "config_drift", False)) if bundle else False,
    )


# -- health -------------------------------------------------------------------


@app.get("/health/live", response_model=HealthResponse)
def live() -> HealthResponse:
    """The process is up. Deliberately does not touch the bundle.

    Liveness that fails on a degraded dependency gets the container killed and
    restarted into the same degraded state, forever.
    """
    return HealthResponse(status="ok")


@app.get("/health/ready", response_model=HealthResponse)
def ready(response: Response) -> HealthResponse:
    """Every *required* component is ready.

    Optional components on a declared fallback do not block readiness -- that is
    what makes them optional -- but they are visible through /v1/model-info.
    """
    try:
        finder, cfg = get_finder()
    except Exception as error:  # noqa: BLE001 - readiness reports, never raises
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
        return HealthResponse(
            status="not_ready", detail=f"{type(error).__name__}: {error}"
        )

    bundle = getattr(finder.pipeline, "bundle", None)
    manifest = getattr(bundle, "manifest", None)
    blocking = manifest.blocking() if manifest is not None else []
    if blocking:
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
        return HealthResponse(
            status="not_ready",
            detail="required bundle components are missing or stale",
            blocking_components=blocking,
        )
    return HealthResponse(status="ok")


# -- errors -------------------------------------------------------------------


@app.exception_handler(Unauthenticated)
def _unauthenticated(request: Request, exc: Unauthenticated) -> JSONResponse:
    return JSONResponse(status_code=401, content={"detail": str(exc)})
