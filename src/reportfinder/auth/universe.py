"""The set of instances one principal may see.

Every generator takes an `AuthorizedUniverse` and searches only inside it. That is
stricter than filtering afterwards, and deliberately so: `top_k` over the whole
estate then filtered is not the same as `top_k` over the authorized estate. The
first silently returns fewer results the less access someone has, and leaks the
*existence* of reports through result counts and score distributions.

The universe also carries how it was decided. That provenance reaches telemetry and
the response, so a permissive development default can never look like a real
entitlement decision.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from ..corpus import CorpusModel


@dataclass(frozen=True)
class AuthorizedUniverse:
    """A boolean mask over corpus positions, plus who decided it."""

    mask: np.ndarray
    resolver: str
    acl_source: str
    development_default: bool = False
    fail_closed: bool = False
    reason: str = ""

    def __post_init__(self) -> None:
        mask = np.asarray(self.mask, dtype=bool)
        mask.setflags(write=False)
        object.__setattr__(self, "mask", mask)

    def __len__(self) -> int:
        return int(self.mask.sum())

    @property
    def is_empty(self) -> bool:
        return not self.mask.any()

    @property
    def positions(self) -> np.ndarray:
        """Corpus positions this principal may see, ascending."""
        return np.flatnonzero(self.mask)

    def allows_position(self, position: int) -> bool:
        return bool(self.mask[position])

    def restrict(self, scores: np.ndarray) -> np.ndarray:
        """Blank out unauthorized rows in a full-corpus score vector.

        `-inf` rather than 0: a zero is a legitimate score that could still be
        ranked and returned, whereas `-inf` can never win a top-k. Generators that
        rank by score therefore cannot surface an unauthorized row even by
        accident.
        """
        restricted = np.asarray(scores, dtype=np.float64).copy()
        restricted[~self.mask] = -np.inf
        return restricted

    def telemetry(self) -> dict[str, object]:
        return {
            "resolver": self.resolver,
            "acl_source": self.acl_source,
            "development_default": self.development_default,
            "fail_closed": self.fail_closed,
            "authorized_instance_count": len(self),
            "reason": self.reason,
        }


def full_universe(corpus: CorpusModel, *, resolver: str, acl_source: str,
                  development_default: bool = False) -> AuthorizedUniverse:
    return AuthorizedUniverse(
        mask=np.ones(len(corpus), dtype=bool),
        resolver=resolver,
        acl_source=acl_source,
        development_default=development_default,
    )


def empty_universe(corpus: CorpusModel, *, resolver: str, reason: str) -> AuthorizedUniverse:
    return AuthorizedUniverse(
        mask=np.zeros(len(corpus), dtype=bool),
        resolver=resolver,
        acl_source="none",
        fail_closed=True,
        reason=reason,
    )
