"""The HTTP contract.

Most of what matters here is negative: what the API must *not* do. It must not
accept a caller-supplied identity, must not serve a request with no identity at
all, must not leak owner/creator columns, must not present an uncalibrated logit as
a probability, and must not report ready when a required component is missing.

The pipeline is stubbed rather than built. These tests are about the transport and
the wire contract; retrieval behaviour is tested where retrieval lives.

Run: uv run pytest tests/test_api.py -v
"""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("fastapi", reason="the `api` extra is not installed")

from fastapi.testclient import TestClient  # noqa: E402

from reportfinder.auth.universe import AuthorizedUniverse  # noqa: E402
from reportfinder.config import from_mapping  # noqa: E402
from reportfinder.corpus import build_corpus_model  # noqa: E402
from reportfinder.pipeline import RrfFuser, SearchPipeline  # noqa: E402
from reportfinder.query.intent import IntentParser  # noqa: E402
from reportfinder.retrieval.field_index import FieldIndex  # noqa: E402

from .fakes import ScriptedGenerator, ScriptedPairScorer  # noqa: E402
from .vague_outcome_fixture import frame  # noqa: E402

# Columns that must never appear in a response, at any nesting depth. These are the
# sensitive operational fields the view contract already keeps out of embeddings;
# the wire contract has no field for them, and this asserts that stays true.
FORBIDDEN_KEYS = (
    "owner", "created_by", "created_date", "last_run_by", "last_run",
    "last_updated", "runs", "runs_total", "available_usage",
)

TITLES = {"R0105": "Payroll Earnings Detail", "R0107": "Learning Course Completion"}


@pytest.fixture(scope="module")
def fixture_frame():
    return frame()


@pytest.fixture(scope="module")
def corpus(fixture_frame):
    return build_corpus_model(
        fixture_frame, ingest_mode="legacy_single_file", source_file="fixture"
    )


@pytest.fixture(scope="module")
def cfg():
    return from_mapping({
        "ingest_mode": "legacy_single_file",
        "corpus_granularity": "report_row",
        "retrieval_mode": "generators",
        "top_k": 5,
    })


class _GrantingResolver:
    """Grants only the instances named for this principal id."""

    name = "test_grants"

    def __init__(self, grants: dict[str, list[str]]) -> None:
        self.grants = grants

    def resolve(self, principal, corpus):
        allowed = self.grants.get(principal.principal_id, [])
        mask = np.zeros(len(corpus), dtype=bool)
        for instance_id in allowed:
            mask[corpus.position_of(instance_id)] = True
        return AuthorizedUniverse(mask=mask, resolver=self.name, acl_source="test")


class _StubFinder:
    """A ReportFinder-shaped object wrapping a real, fully-faked pipeline."""

    def __init__(self, pipeline) -> None:
        self.pipeline = pipeline

    def search(self, request):
        return self.pipeline.run(request)


def _build(corpus, fixture_frame, cfg, resolver):
    generator = ScriptedGenerator("bm25f", {})

    class _Any(ScriptedGenerator):
        def generate(self, plan, universe, k):
            self.results = {
                plan.raw_query: [
                    (i, 1.0) for i in TITLES
                    if universe.allows_position(corpus.position_of(i))
                ]
            }
            return super().generate(plan, universe, k)

    generator = _Any("bm25f", {})
    pipeline = SearchPipeline(
        cfg=cfg,
        corpus=corpus,
        generators=[generator],
        resolver=resolver,
        intent_parser=IntentParser(fixture_frame, cfg),
        reranker=ScriptedPairScorer(
            {"R0105": 6.0, "R0107": 1.0}, default=0.0, corpus=corpus
        ),
        fuser=RrfFuser(constant=cfg.retrieval.rrf_constant),
        field_index=FieldIndex(fixture_frame),
    )
    return _StubFinder(pipeline)


@pytest.fixture
def client(monkeypatch, corpus, fixture_frame, cfg, tmp_path):
    from reportfinder.api import deps, service

    resolver = _GrantingResolver({
        "alice": list(TITLES),
        "bob": ["R0107"],
        "carol": [],
    })
    finder = _build(corpus, fixture_frame, cfg, resolver)
    served_cfg = cfg.with_overrides(cache_dir=tmp_path)

    monkeypatch.setattr(deps, "get_finder", lambda: (finder, served_cfg))
    monkeypatch.setattr(service, "get_finder", lambda: (finder, served_cfg))
    service._STORES.clear()
    return TestClient(service.app, raise_server_exceptions=False)


def _search(client, query="payroll earnings", principal="alice", **kwargs):
    return client.post(
        "/v1/search",
        json={"query": query, **kwargs},
        headers={"x-principal-id": principal} if principal else {},
    )


# --- shape -------------------------------------------------------------------


def test_search_returns_the_specified_envelope(client):
    body = _search(client).json()

    assert set(body) >= {
        "request_id", "decision", "results", "clarification", "warnings",
        "catalog_version", "model_bundle_version", "latency_ms",
    }
    assert body["decision"] in {
        "RETURN_RESULTS", "ASK_CLARIFICATION", "NO_CONFIDENT_MATCH"
    }


def test_each_result_carries_family_and_selected_instance(client):
    body = _search(client).json()
    # Asserted, not skipped. This used to skip when the pipeline returned nothing,
    # so a decision policy that regressed to always-abstain would have reported
    # "skipped" for the result-shape contract instead of failing. The stub scores
    # R0105 at 6.0 against a 0.0 default, which is a decisive margin by design.
    assert body["results"], (
        f"the scripted pipeline decided {body['decision']}, so the result contract "
        "is unreachable -- fix the fixture rather than skipping the assertion"
    )

    item = body["results"][0]
    assert set(item) >= {
        "rank", "family_id", "title", "unnormalized_ranking_score",
        "selected_instance", "grounded_evidence",
    }
    assert item["selected_instance"]["report_instance_id"].startswith("R")
    # Family and instance identity stay distinct -- collapsing them is what the
    # two-level model exists to prevent.
    assert item["family_id"] != item["selected_instance"]["report_instance_id"]


def test_no_score_is_presented_as_a_probability(client):
    body = _search(client).json()
    assert body["decision_calibrated"] is False
    for item in body["results"]:
        assert "probability" not in item
        assert "confidence" not in item
        # The name states what it is. A raw cross-encoder logit is not in [0, 1].
        assert "unnormalized_ranking_score" in item


def test_grounded_evidence_reports_overlap_not_the_whole_schema(client):
    """`matched_fields` must mean matched.

    Returning every field under that name is a lie a user cannot detect: it looks
    like evidence and is actually the report's entire schema.
    """
    body = _search(client, query="payroll earnings").json()
    item = next(
        r for r in body["results"]
        if r["selected_instance"]["report_instance_id"] == "R0105"
    )
    evidence = item["grounded_evidence"]
    all_fields = item["selected_instance"]["fields"]

    assert set(evidence["matched_fields"]) < set(all_fields), (
        "evidence must be a strict subset of the schema"
    )
    # "Net Pay" and "Pay Period" share the token 'pay' with the query; "Gross Pay"
    # does too. Nothing else should.
    assert all(
        "pay" in f.casefold() for f in evidence["matched_fields"]
    ), evidence["matched_fields"]
    assert "earnings" in evidence["matched_title_concepts"]


def test_evidence_is_empty_rather_than_padded_when_nothing_overlaps(client):
    body = _search(client, query="zzzqqq nonexistent vocabulary").json()
    for item in body["results"]:
        assert item["grounded_evidence"]["matched_fields"] == []
        assert item["grounded_evidence"]["matched_title_concepts"] == []


def test_active_fallbacks_are_disclosed_on_every_response(client):
    body = _search(client).json()
    assert "active_fallbacks" in body
    assert isinstance(body["active_fallbacks"], list)


# --- authorization -----------------------------------------------------------


def test_a_request_with_no_identity_is_refused(client):
    """Fail closed. A service that assumes an identity grants the whole estate to
    an unauthenticated caller."""
    assert _search(client, principal=None).status_code == 401


def test_the_body_cannot_name_its_own_principal(client):
    """Identity comes from the transport. A caller-supplied one would make every
    entitlement check advisory."""
    response = client.post(
        "/v1/search",
        json={"query": "payroll earnings", "principal_id": "alice",
              "user": "alice", "acl_grants": ["*"]},
        headers={"x-principal-id": "carol"},
    )
    # The extra fields are ignored, and carol's empty universe still applies.
    assert response.status_code == 200
    assert response.json()["results"] == []


def test_two_principals_get_different_results_for_one_query(client):
    alice = _search(client, principal="alice").json()
    bob = _search(client, principal="bob").json()

    alice_ids = {r["selected_instance"]["report_instance_id"] for r in alice["results"]}
    bob_ids = {r["selected_instance"]["report_instance_id"] for r in bob["results"]}

    assert "R0105" in alice_ids
    assert "R0105" not in bob_ids, "bob has no grant for R0105"


def test_an_unauthorized_report_appears_nowhere_in_the_response(client):
    body = _search(client, principal="bob")
    assert "R0105" not in body.text
    assert "Payroll Earnings Detail" not in body.text


def test_an_empty_universe_returns_no_confident_match_not_an_error(client):
    body = _search(client, principal="carol").json()
    assert body["decision"] == "NO_CONFIDENT_MATCH"
    assert body["results"] == []
    assert body["warnings"]


# --- sensitive fields --------------------------------------------------------


def _keys(payload) -> set[str]:
    found: set[str] = set()
    if isinstance(payload, dict):
        for key, value in payload.items():
            found.add(key)
            found |= _keys(value)
    elif isinstance(payload, list):
        for item in payload:
            found |= _keys(item)
    return found


def test_sensitive_operational_columns_are_absent_from_responses(client):
    """Absent by construction, not stripped by a filter.

    A response model that *could* carry these and relies on filtering is one
    refactor away from leaking; a model with no such field cannot.
    """
    keys = _keys(_search(client).json())
    assert not (keys & set(FORBIDDEN_KEYS)), sorted(keys & set(FORBIDDEN_KEYS))


def test_the_openapi_schema_declares_no_sensitive_field(client):
    """Not just this response -- no shape the API can ever emit has one."""
    keys = _keys(client.get("/openapi.json").json())
    leaked = {k for k in FORBIDDEN_KEYS if k in keys}
    assert not leaked, sorted(leaked)


# --- feedback ----------------------------------------------------------------


def _feedback(client, key="k1", principal="alice", **kwargs):
    return client.post(
        "/v1/feedback",
        json={"request_id": "r1", "idempotency_key": key,
              "slate": ["payroll earnings detail"], **kwargs},
        headers={"x-principal-id": principal},
    )


def test_feedback_is_recorded_with_the_full_slate(client, tmp_path):
    assert _feedback(client, selected_family_id="payroll earnings detail").json() == {
        "recorded": True, "idempotency_key": "k1", "duplicate": False,
    }
    lines = (tmp_path / "feedback.jsonl").read_text().strip().splitlines()
    record = __import__("json").loads(lines[0])

    assert record["slate"] == ["payroll earnings detail"]
    assert record["positions"] == {"payroll earnings detail": 1}
    # Recorded as what it is. Nothing downstream may mistake it for adjudicated
    # relevance.
    assert record["label_basis"] == "implicit impression; not a relevance judgement"
    # Both config hashes, so a judgement can be tied to the configuration that
    # produced the slate. `policy_version` alone is the coarse retrieval mode,
    # which two materially different runs share.
    assert "build_config_hash" in record
    assert "runtime_config_hash" in record


def test_replaying_a_feedback_key_is_a_no_op_not_an_error(client, tmp_path):
    _feedback(client, key="dup")
    second = _feedback(client, key="dup")

    assert second.status_code == 200
    assert second.json()["duplicate"] is True
    assert len((tmp_path / "feedback.jsonl").read_text().strip().splitlines()) == 1


def test_feedback_records_the_safe_principal_label_not_the_raw_identity(
    client, tmp_path
):
    """This file outlives the request and is read by more people than it was."""
    _feedback(client, key="safe", principal="alice")
    record = __import__("json").loads((tmp_path / "feedback.jsonl").read_text())
    assert record["principal"] == "-:alice"


def test_feedback_requires_an_identity(client):
    response = client.post(
        "/v1/feedback", json={"request_id": "r1", "idempotency_key": "x"},
    )
    assert response.status_code == 401


# --- model info and health ---------------------------------------------------


def test_model_info_states_what_is_not_trained(client):
    body = client.get("/v1/model-info").json()

    # The three claims most easily misread as true, stated flatly.
    assert body["fusion_trained"] is False
    assert body["decision_trained"] is False
    assert body["decision_calibrated"] is False
    assert body["retrieval_mode"] == "generators"


def test_model_info_reports_build_and_runtime_configuration(client):
    """The bundle id keys on corpus content plus index config only, so it cannot
    distinguish two runs that retrieve differently. These fields can."""
    body = client.get("/v1/model-info").json()
    assert set(body) >= {"build_config_hash", "runtime_config_hash", "config_drift"}
    assert isinstance(body["config_drift"], bool)


def test_model_info_discloses_the_development_authorization_default(client):
    body = client.get("/v1/model-info").json()
    assert "authorization_resolver" in body
    assert "authorization_is_development_default" in body


def test_liveness_does_not_depend_on_the_bundle(client):
    assert client.get("/health/live").json() == {
        "status": "ok", "detail": "", "blocking_components": [],
    }


def test_readiness_reports_not_ready_when_a_required_component_blocks(
    client, monkeypatch
):
    from reportfinder.api import service

    class _Manifest:
        def blocking(self):
            return ["splade"]

    class _Bundle:
        manifest = _Manifest()

    finder, cfg = service.get_finder()
    monkeypatch.setattr(finder.pipeline, "bundle", _Bundle(), raising=False)

    response = client.get("/health/ready")
    assert response.status_code == 503
    assert response.json()["status"] == "not_ready"
    assert response.json()["blocking_components"] == ["splade"]
