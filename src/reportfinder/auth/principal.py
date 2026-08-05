"""Who is asking, and what they asked for.

Identity is resolved server-side and passed in. Nothing downstream may derive
entitlements from the query text, and no caller may hand the pipeline a list of
report ids it is "allowed" to see -- that decision belongs to the resolver.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field


@dataclass(frozen=True)
class Principal:
    """A server-resolved identity.

    `roles` and `acl_grants` are what an entitlement resolver keys on.
    `attributes` carries anything else a deployment's resolver needs; it is never
    interpreted here.
    """

    principal_id: str
    tenant_id: str = ""
    roles: frozenset[str] = frozenset()
    acl_grants: frozenset[str] = frozenset()
    attributes: dict[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.principal_id:
            raise ValueError("Principal.principal_id must not be empty")
        object.__setattr__(self, "roles", frozenset(self.roles))
        object.__setattr__(self, "acl_grants", frozenset(self.acl_grants))

    @property
    def safe_label(self) -> str:
        """A telemetry-safe identifier.

        Logs and traces reference this rather than the raw principal, so identity
        does not leak into artifacts that are shared more widely than the request.
        """
        return f"{self.tenant_id or '-'}:{self.principal_id}"


# The identity used by the CLI, the Streamlit app and tests. It is a *development*
# identity, not an anonymous or public one: it exists so that no code path can run
# without a principal at all, which is how authorization checks get skipped.
DEVELOPMENT_PRINCIPAL = Principal(
    principal_id="local-development",
    tenant_id="dev",
    roles=frozenset({"developer"}),
)


@dataclass(frozen=True)
class SearchRequest:
    """One authorized search."""

    raw_query: str
    principal: Principal
    top_k: int | None = None
    request_id: str = field(default_factory=lambda: uuid.uuid4().hex[:16])
    # Set when a clarification answer is folded back in, so the rerun records that
    # it is a continuation rather than a fresh request.
    clarification_context: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.raw_query or not self.raw_query.strip():
            raise ValueError("Query is empty.")
        object.__setattr__(self, "clarification_context", tuple(self.clarification_context))

    def with_clarification(self, answer: str) -> SearchRequest:
        """Fold a clarification answer into the request and rerun the full pipeline.

        The raw query is preserved; the answer is added as explicit context rather
        than being spliced into the user's words.
        """
        return SearchRequest(
            raw_query=self.raw_query,
            principal=self.principal,
            top_k=self.top_k,
            request_id=self.request_id,
            clarification_context=(*self.clarification_context, answer),
        )
