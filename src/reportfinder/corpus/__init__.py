"""Two-level report model: families, instances, and their searchable views."""

from __future__ import annotations

from .frames import CorpusModel, build_corpus_model, catalog_version
from .identity import (
    UNRESTRICTED_ACL_KEY,
    LifecycleState,
    Provenance,
    ReportFamily,
    ReportInstance,
    build_families,
    build_instances,
    content_hash,
)
from .views import (
    ALL_VIEW_TYPES,
    FORBIDDEN_VIEW_SOURCES,
    VIEW_SCHEMA_VERSION,
    InstanceView,
    ViewType,
    authoritative_text,
    build_view,
    build_views,
    corpus_content_hash,
)

__all__ = [
    "ALL_VIEW_TYPES",
    "FORBIDDEN_VIEW_SOURCES",
    "UNRESTRICTED_ACL_KEY",
    "VIEW_SCHEMA_VERSION",
    "CorpusModel",
    "InstanceView",
    "LifecycleState",
    "Provenance",
    "ReportFamily",
    "ReportInstance",
    "ViewType",
    "authoritative_text",
    "build_corpus_model",
    "build_families",
    "build_instances",
    "build_view",
    "build_views",
    "catalog_version",
    "content_hash",
    "corpus_content_hash",
]
