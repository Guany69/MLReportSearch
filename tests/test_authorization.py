"""Authorization: resolved before candidates exist, and failing closed.

The claims that matter here are the ones whose failure is invisible in normal use:
a permissive default that stops announcing itself, a resolver error that returns
results anyway, or an unauthorized row that can still win a top-k because it was
scored 0 rather than excluded.

Run: uv run pytest tests/test_authorization.py -v
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from reportfinder.auth import (
    AllowAllResolver,
    AuthorizedUniverse,
    ExplicitAclResolver,
    FailClosedResolver,
    Principal,
    SearchRequest,
    build_resolver,
)
from reportfinder.config import DEFAULT, from_mapping
from reportfinder.corpus import UNRESTRICTED_ACL_KEY, build_corpus_model


def _corpus(n: int = 5):
    frame = pd.DataFrame([
        {
            "report_key": f"R{4 + i:04d}", "title_key": f"report {i}", "source_row": 4 + i,
            "title": f"Report {i}", "description": "", "category": "Worker Data",
            "data_source": "All Workers", "report_type": "Advanced",
            "prompts": [], "fields": [f"Field {i}"], "tags": "",
            "area_where_used": "", "worklet": "", "chart_type": "",
            "landing_page": "", "worklet_landing_pages": "", "shared": "Yes",
        }
        for i in range(n)
    ])
    return build_corpus_model(frame, ingest_mode="legacy_single_file", source_file="f")


def _restrict_acl(corpus, mapping: dict[str, str]):
    """Rebuild the corpus with per-instance acl_keys, as a real source would."""
    import dataclasses

    instances = tuple(
        dataclasses.replace(i, acl_key=mapping.get(i.report_instance_id, i.acl_key))
        for i in corpus.instances
    )
    return dataclasses.replace(corpus, instances=instances)


# --- the permissive default must stay loud ----------------------------------


def test_allow_all_requires_explicit_opt_in():
    with pytest.raises(ValueError, match="explicit opt-in"):
        AllowAllResolver(allow_dev_default=False)


def test_allow_all_discloses_itself_as_a_development_default():
    """If this stops being disclosed, an open door becomes invisible."""
    corpus = _corpus()
    universe = AllowAllResolver(allow_dev_default=True).resolve(
        Principal(principal_id="u1"), corpus
    )
    assert len(universe) == len(corpus)
    assert universe.development_default is True
    assert universe.acl_source == "none"
    telemetry = universe.telemetry()
    assert telemetry["development_default"] is True
    assert telemetry["resolver"] == "allow_all"


# --- explicit ACLs ----------------------------------------------------------


def test_explicit_acl_grants_only_what_the_principal_holds():
    corpus = _restrict_acl(_corpus(), {"R0004": "payroll", "R0005": "payroll",
                                       "R0006": "hr"})
    universe = ExplicitAclResolver().resolve(
        Principal(principal_id="u1", acl_grants=frozenset({"payroll"})), corpus
    )
    allowed = {corpus.instances[p].report_instance_id for p in universe.positions}
    # The two payroll reports, plus the three that carry the unrestricted sentinel.
    assert allowed == {"R0004", "R0005", "R0007", "R0008"}
    assert "R0006" not in allowed
    assert universe.development_default is False
    assert universe.acl_source == "instance.acl_key"


def test_unrestricted_sentinel_is_visible_to_everyone():
    corpus = _corpus()
    assert all(i.acl_key == UNRESTRICTED_ACL_KEY for i in corpus.instances)
    universe = ExplicitAclResolver().resolve(Principal(principal_id="nobody"), corpus)
    assert len(universe) == len(corpus)


def test_a_principal_with_no_grants_sees_no_restricted_reports():
    corpus = _restrict_acl(_corpus(3), {f"R{4 + i:04d}": "secret" for i in range(3)})
    universe = ExplicitAclResolver().resolve(Principal(principal_id="u1"), corpus)
    assert universe.is_empty
    assert len(universe) == 0


# --- fail closed ------------------------------------------------------------


class _Exploding:
    name = "exploding"

    def resolve(self, principal, corpus):
        raise RuntimeError("directory unreachable")


class _WrongShape:
    name = "wrong_shape"

    def resolve(self, principal, corpus):
        return AuthorizedUniverse(
            mask=np.ones(len(corpus) + 3, dtype=bool),
            resolver=self.name, acl_source="broken",
        )


def test_resolver_failure_denies_rather_than_grants():
    corpus = _corpus()
    universe = FailClosedResolver(_Exploding()).resolve(Principal(principal_id="u1"), corpus)
    assert universe.is_empty
    assert universe.fail_closed is True
    assert universe.reason == "entitlements could not be resolved"


def test_malformed_mask_is_rejected_rather_than_misaligned():
    """A wrong-length mask would expose the wrong reports, not merely fail."""
    corpus = _corpus()
    universe = FailClosedResolver(_WrongShape()).resolve(Principal(principal_id="u1"), corpus)
    assert universe.is_empty
    assert "malformed" in universe.reason


def test_fail_open_must_be_opted_into_and_still_raises():
    corpus = _corpus()
    with pytest.raises(RuntimeError, match="directory unreachable"):
        FailClosedResolver(_Exploding(), fail_closed=False).resolve(
            Principal(principal_id="u1"), corpus
        )


def test_build_resolver_always_wraps_fail_closed():
    resolver = build_resolver(DEFAULT)
    assert isinstance(resolver, FailClosedResolver)
    assert resolver.fail_closed is True

    explicit = build_resolver(from_mapping({"auth": {"resolver": "explicit_acl"}}))
    assert isinstance(explicit.inner, ExplicitAclResolver)


# --- scores of unauthorized rows can never win ------------------------------


def test_restrict_makes_unauthorized_rows_unrankable():
    """-inf, not 0: a zero can still be returned when everything scores zero."""
    mask = np.array([True, False, True, False])
    universe = AuthorizedUniverse(mask=mask, resolver="t", acl_source="t")

    scores = np.array([0.1, 99.0, 0.2, 50.0])
    restricted = universe.restrict(scores)
    assert restricted[1] == -np.inf and restricted[3] == -np.inf
    assert int(np.argmax(restricted)) == 2

    # Even with every authorized score at zero, no unauthorized row surfaces.
    all_zero = universe.restrict(np.zeros(4))
    assert int(np.argmax(all_zero)) in {0, 2}


def test_restrict_does_not_mutate_the_caller_scores():
    universe = AuthorizedUniverse(
        mask=np.array([True, False]), resolver="t", acl_source="t"
    )
    scores = np.array([1.0, 2.0])
    universe.restrict(scores)
    assert scores.tolist() == [1.0, 2.0]


def test_universe_mask_is_immutable():
    universe = AuthorizedUniverse(
        mask=np.array([True, False]), resolver="t", acl_source="t"
    )
    with pytest.raises(ValueError):
        universe.mask[0] = False


# --- request contract -------------------------------------------------------


def test_search_request_requires_a_query():
    with pytest.raises(ValueError, match="Query is empty"):
        SearchRequest(raw_query="   ", principal=Principal(principal_id="u1"))


def test_principal_requires_an_identity():
    with pytest.raises(ValueError, match="principal_id must not be empty"):
        Principal(principal_id="")


def test_request_ids_are_unique_per_request():
    p = Principal(principal_id="u1")
    assert SearchRequest("a", p).request_id != SearchRequest("a", p).request_id


def test_clarification_preserves_the_raw_query_and_request_id():
    """The user's words are never rewritten; the answer is added as context."""
    request = SearchRequest("headcount", Principal(principal_id="u1"))
    followup = request.with_clarification("current, not historical")
    assert followup.raw_query == "headcount"
    assert followup.request_id == request.request_id
    assert followup.clarification_context == ("current, not historical",)


def test_safe_label_does_not_expose_attributes():
    principal = Principal(
        principal_id="u1", tenant_id="acme",
        attributes={"email": "someone@example.com"},
    )
    assert principal.safe_label == "acme:u1"
    assert "example.com" not in principal.safe_label
