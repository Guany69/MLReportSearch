"""Identity and entitlement, resolved before any candidate is generated."""

from __future__ import annotations

from .principal import DEVELOPMENT_PRINCIPAL, Principal, SearchRequest
from .resolver import (
    AllowAllResolver,
    EntitlementResolver,
    ExplicitAclResolver,
    FailClosedResolver,
    build_resolver,
)
from .universe import AuthorizedUniverse, empty_universe, full_universe

__all__ = [
    "DEVELOPMENT_PRINCIPAL",
    "AllowAllResolver",
    "AuthorizedUniverse",
    "EntitlementResolver",
    "ExplicitAclResolver",
    "FailClosedResolver",
    "Principal",
    "SearchRequest",
    "build_resolver",
    "empty_universe",
    "full_universe",
]
