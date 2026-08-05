"""Process-wide state and server-side identity resolution.

Two things happen here and nowhere else: the finder is built once per process, and
the principal is derived from the transport credential.

**Identity never comes from the request body.** `SearchRequestBody` has no user
field, and this module is the only place a `Principal` is constructed for an HTTP
request. If a caller could name its own principal, every entitlement check below
would be advisory.
"""

from __future__ import annotations

import functools
import logging
import os

from ..auth import Principal
from ..config import DEFAULT, Config, from_mapping
from ..model import ReportFinder
from ..represent import load_or_build

log = logging.getLogger(__name__)

# The header a fronting proxy or gateway is expected to set after it has
# authenticated the caller. It is trusted precisely because the service is not
# meant to be exposed directly -- which is why `REPORTFINDER_API_REQUIRE_AUTH`
# defaults to on and refuses requests without it.
PRINCIPAL_HEADER = "x-principal-id"
TENANT_HEADER = "x-tenant-id"
ROLES_HEADER = "x-principal-roles"
GRANTS_HEADER = "x-principal-grants"


class Unauthenticated(Exception):
    """No usable identity on the request."""


def _config() -> Config:
    path = os.environ.get("REPORTFINDER_CONFIG")
    if not path:
        return DEFAULT
    from pathlib import Path

    import yaml

    return from_mapping(yaml.safe_load(Path(path).read_text()) or {})


@functools.lru_cache(maxsize=1)
def get_finder() -> tuple[ReportFinder, Config]:
    """Build the finder once per process.

    Eager rather than lazy: a checkpoint that fails to load should fail startup and
    readiness, not the first user's query.
    """
    cfg = _config()
    finder = ReportFinder(load_or_build(cfg, rebuild=False, verbose=False), cfg)
    if cfg.retrieval_mode == "generators":
        _ = finder.pipeline  # constructs indexes and asserts bundle readiness
    return finder, cfg


def _split(value: str | None) -> frozenset[str]:
    return frozenset(part.strip() for part in (value or "").split(",") if part.strip())


def principal_from_headers(headers) -> Principal:
    """Resolve identity from the transport, never from the payload.

    Fails closed. An absent or blank principal header raises rather than falling
    back to a development identity: a service that silently substitutes one grants
    the whole estate to an unauthenticated caller.
    """
    principal_id = (headers.get(PRINCIPAL_HEADER) or "").strip()
    if not principal_id:
        if os.environ.get("REPORTFINDER_API_ALLOW_ANONYMOUS") == "1":
            # Explicitly opted into, for local development only, and it still
            # yields a *named* development principal rather than a wildcard.
            log.warning("api_anonymous_principal_enabled")
            from ..auth import DEVELOPMENT_PRINCIPAL

            return DEVELOPMENT_PRINCIPAL
        raise Unauthenticated(
            f"missing {PRINCIPAL_HEADER}; the service does not authenticate callers "
            "itself and will not assume an identity"
        )
    return Principal(
        principal_id=principal_id,
        tenant_id=(headers.get(TENANT_HEADER) or "").strip(),
        roles=_split(headers.get(ROLES_HEADER)),
        acl_grants=_split(headers.get(GRANTS_HEADER)),
    )
