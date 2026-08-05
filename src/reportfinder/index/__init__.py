"""Index components and the bundle manifest that versions them together."""

from __future__ import annotations

from .bundle import (
    MANIFEST_VERSION,
    BundleManifest,
    BundleNotReady,
    ComponentRecord,
    ComponentStatus,
    Requirement,
    component_signature,
    config_hash,
    environment_record,
    index_config_hash,
)
from .dense_views import INDEX_SCHEMA_VERSION, BuildStats, DenseViewIndex
from .encoders import SentenceTransformerEncoder, TextEncoder, l2_normalize
from .late_interaction import (
    LateInteractionIndex,
    LateInteractionStatus,
    LateInteractionUnavailable,
    TokenEncoder,
    maxsim,
)
from .prototypes import (
    FamilyPrototypeIndex,
    PrototypeSource,
    QueryPrototype,
    ValidationStatus,
    seed_prototypes_from_catalog,
)
from .splade import SparseEncoder, SpladeIndex, SpladeTorchEncoder

__all__ = [
    "INDEX_SCHEMA_VERSION",
    "MANIFEST_VERSION",
    "BuildStats",
    "BundleManifest",
    "BundleNotReady",
    "ComponentRecord",
    "ComponentStatus",
    "DenseViewIndex",
    "FamilyPrototypeIndex",
    "LateInteractionIndex",
    "LateInteractionStatus",
    "LateInteractionUnavailable",
    "PrototypeSource",
    "QueryPrototype",
    "Requirement",
    "SentenceTransformerEncoder",
    "SparseEncoder",
    "SpladeIndex",
    "SpladeTorchEncoder",
    "TextEncoder",
    "TokenEncoder",
    "ValidationStatus",
    "component_signature",
    "config_hash",
    "environment_record",
    "index_config_hash",
    "l2_normalize",
    "maxsim",
    "seed_prototypes_from_catalog",
]
