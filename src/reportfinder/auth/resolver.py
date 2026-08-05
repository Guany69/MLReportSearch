"""Entitlement resolution.

This estate exports no ACL data, so the shipped resolver is permissive. That is a
real risk if it is ever inherited silently, so three things guard it:

1. `AllowAllResolver` refuses to construct unless `auth.allow_dev_default` is set.
2. It marks every universe it returns `development_default=True`, which reaches
   telemetry and the response as a disclosed fallback.
3. `FailClosedResolver` wraps whatever resolver is configured, so a resolver that
   raises yields an empty universe -- never a full-estate search.

`ExplicitAclResolver` is the shape a real deployment plugs in: it keys instances by
`acl_key` and grants by principal. It is exercised by tests today and is the reason
`ReportInstance.acl_key` exists.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Protocol, runtime_checkable

import numpy as np

from ..corpus import UNRESTRICTED_ACL_KEY
from .principal import Principal
from .universe import AuthorizedUniverse, empty_universe, full_universe

if TYPE_CHECKING:
    from ..config import Config
    from ..corpus import CorpusModel

log = logging.getLogger(__name__)


@runtime_checkable
class EntitlementResolver(Protocol):
    """Decides which instances a principal may see."""

    name: str

    def resolve(self, principal: Principal, corpus: CorpusModel) -> AuthorizedUniverse:
        ...


class AllowAllResolver:
    """Grants the whole estate. Development default for a catalog with no ACLs."""

    name = "allow_all"

    def __init__(self, *, allow_dev_default: bool) -> None:
        if not allow_dev_default:
            raise ValueError(
                "AllowAllResolver requires auth.allow_dev_default=true. "
                "Refusing to grant the whole estate without an explicit opt-in."
            )

    def resolve(self, principal: Principal, corpus: CorpusModel) -> AuthorizedUniverse:
        return full_universe(
            corpus,
            resolver=self.name,
            acl_source="none",
            development_default=True,
        )


class ExplicitAclResolver:
    """Grants instances whose `acl_key` the principal holds.

    Instances carrying the unrestricted sentinel are visible to everyone; that is
    what the sentinel means. Everything else requires a matching grant.
    """

    name = "explicit_acl"

    def resolve(self, principal: Principal, corpus: CorpusModel) -> AuthorizedUniverse:
        grants = principal.acl_grants
        mask = np.fromiter(
            (
                instance.acl_key == UNRESTRICTED_ACL_KEY or instance.acl_key in grants
                for instance in corpus.instances
            ),
            dtype=bool,
            count=len(corpus),
        )
        return AuthorizedUniverse(
            mask=mask,
            resolver=self.name,
            acl_source="instance.acl_key",
        )


class FailClosedResolver:
    """Wraps a resolver so failure denies rather than grants.

    An entitlement lookup that raises is the case where an open failure mode is
    most dangerous and least visible: the request still returns results, so nothing
    looks broken.
    """

    def __init__(self, inner: EntitlementResolver, *, fail_closed: bool = True) -> None:
        self.inner = inner
        self.fail_closed = fail_closed
        self.name = getattr(inner, "name", type(inner).__name__)

    def resolve(self, principal: Principal, corpus: CorpusModel) -> AuthorizedUniverse:
        try:
            universe = self.inner.resolve(principal, corpus)
        except Exception as error:  # noqa: BLE001 - any failure must deny
            if not self.fail_closed:
                raise
            log.warning(
                "entitlement_resolution_failed",
                extra={"resolver": self.name, "error": type(error).__name__},
            )
            return empty_universe(
                corpus,
                resolver=self.name,
                reason="entitlements could not be resolved",
            )

        if universe.mask.shape != (len(corpus),):
            # A malformed mask would misalign every position downstream, which
            # would expose the wrong reports rather than merely failing.
            if not self.fail_closed:
                raise ValueError("resolver returned a mask of the wrong length")
            log.warning("entitlement_mask_shape_invalid", extra={"resolver": self.name})
            return empty_universe(
                corpus,
                resolver=self.name,
                reason="entitlement resolver returned a malformed universe",
            )
        return universe


def build_resolver(cfg: Config) -> FailClosedResolver:
    """Construct the configured resolver, always fail-closed wrapped."""
    if cfg.auth.resolver == "allow_all":
        inner: EntitlementResolver = AllowAllResolver(
            allow_dev_default=cfg.auth.allow_dev_default
        )
    elif cfg.auth.resolver == "explicit_acl":
        inner = ExplicitAclResolver()
    else:  # pragma: no cover - Config validation rejects this first
        raise ValueError(f"unknown auth.resolver: {cfg.auth.resolver}")
    return FailClosedResolver(inner, fail_closed=cfg.auth.fail_closed_on_acl_error)
