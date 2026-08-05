"""The bundle manifest: what was built, from what, with which model.

A retrieval system is only reproducible if the corpus, the checkpoints, the
configuration and the indexes are versioned *together*. Any one of them changing
alone silently changes results. The manifest records all of it and is the single
place readiness is decided.

Two rules it exists to enforce:

* A **required** component that is missing or stale blocks startup. Serving a stale
  index is worse than failing, because the results look normal.
* An **optional** component that is missing must name an explicit fallback, and
  that fallback is disclosed in telemetry and in the response. There is no silent
  degradation.

The second rule is enforced structurally: `validate_structure` rejects any manifest
where a non-ready component has no fallback, so "optional missing -> explicit
fallback" cannot be forgotten when a new component is added.
"""

from __future__ import annotations

import hashlib
import json
import platform
import sys
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path

MANIFEST_VERSION = "1"


class ComponentStatus(str, Enum):
    READY = "ready"
    ABSENT = "absent"
    STALE = "stale"
    BUILD_FAILED = "build_failed"
    # Implemented and possibly built, deliberately not serving.
    BUILT_DISABLED = "built_disabled"
    # Built, but the source text carries too little variation to retrieve on.
    DEGENERATE_LOW_ENTROPY = "degenerate_low_entropy"
    INCOMPATIBLE_SCHEMA = "incompatible_schema"


class Requirement(str, Enum):
    REQUIRED = "required"
    OPTIONAL = "optional"


class BundleNotReady(RuntimeError):
    """A required component is missing, stale or incompatible."""

    def __init__(self, blocking: list[str], detail: str) -> None:
        super().__init__(detail)
        self.blocking = blocking


@dataclass
class ComponentRecord:
    """One index component's build result."""

    name: str
    requirement: Requirement
    status: ComponentStatus
    signature: str = ""
    model_id: str = ""
    model_revision: str = ""
    rows: int = 0
    fallback: str | None = None
    detail: dict[str, object] = field(default_factory=dict)

    @property
    def is_ready(self) -> bool:
        return self.status is ComponentStatus.READY

    def to_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["requirement"] = self.requirement.value
        payload["status"] = self.status.value
        return payload

    @classmethod
    def from_dict(cls, payload: dict) -> ComponentRecord:
        return cls(
            name=payload["name"],
            requirement=Requirement(payload["requirement"]),
            status=ComponentStatus(payload["status"]),
            signature=payload.get("signature", ""),
            model_id=payload.get("model_id", ""),
            model_revision=payload.get("model_revision", ""),
            rows=payload.get("rows", 0),
            fallback=payload.get("fallback"),
            detail=payload.get("detail", {}),
        )


def component_signature(
    *,
    schema_version: str,
    corpus_content_hash: str,
    model_id: str,
    model_revision: str,
    params: dict[str, object],
) -> str:
    """Per-component signature.

    Scoped to one component on purpose: folding everything into one digest would
    mean changing the dense checkpoint invalidates the SPLADE index, and every
    rebuild becomes all-or-nothing.
    """
    payload = json.dumps(
        {
            "schema_version": schema_version,
            "corpus_content_hash": corpus_content_hash,
            "model_id": model_id,
            "model_revision": model_revision,
            "params": params,
        },
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def environment_record() -> dict[str, str]:
    def version(package: str) -> str:
        try:
            from importlib.metadata import version as _version

            return _version(package)
        except Exception:  # noqa: BLE001 - a missing package is data, not an error
            return "missing"

    return {
        "python": platform.python_version(),
        "platform": f"{sys.platform}-{platform.release()}",
        "torch": version("torch"),
        "sentence_transformers": version("sentence-transformers"),
        "transformers": version("transformers"),
        "numpy": version("numpy"),
        "scipy": version("scipy"),
    }


@dataclass
class BundleManifest:
    """Everything needed to reproduce and validate a served bundle."""

    bundle_version: str
    catalog_version: str
    corpus_content_hash: str
    ingest_mode: str
    corpus_granularity: str
    source_files: list[dict[str, object]]
    instance_count: int
    family_count: int
    config_hash: str
    code_version: str
    created_at: str
    environment: dict[str, str] = field(default_factory=environment_record)
    components: dict[str, ComponentRecord] = field(default_factory=dict)
    manifest_version: str = MANIFEST_VERSION

    # -- readiness ---------------------------------------------------------

    def validate_structure(self) -> None:
        """Every non-ready component must name a fallback.

        Structural rather than per-component so a newly added component cannot be
        merged without deciding what happens when it is missing.
        """
        offenders = [
            name
            for name, record in self.components.items()
            if not record.is_ready and not record.fallback
        ]
        if offenders:
            raise ValueError(
                "components are not ready and declare no fallback: "
                f"{sorted(offenders)}. Every non-ready component must state what "
                "happens in its absence."
            )

    def blocking(self) -> list[str]:
        return sorted(
            name
            for name, record in self.components.items()
            if record.requirement is Requirement.REQUIRED and not record.is_ready
        )

    def degraded(self) -> list[str]:
        return sorted(
            name
            for name, record in self.components.items()
            if record.requirement is Requirement.OPTIONAL and not record.is_ready
        )

    def active_fallbacks(self) -> list[str]:
        return sorted(
            f"{name}:{record.fallback}"
            for name, record in self.components.items()
            if not record.is_ready and record.fallback
        )

    def assert_ready(self) -> None:
        self.validate_structure()
        blocking = self.blocking()
        if blocking:
            commands = "\n".join(
                f"  reportfinder-bundle build --component {name}" for name in blocking
            )
            raise BundleNotReady(
                blocking,
                "Bundle is not ready. Required components are missing or stale: "
                f"{blocking}.\nBuild them with:\n{commands}",
            )

    def mark_stale_components(self, corpus_content_hash: str) -> None:
        """Flag components built against a different corpus.

        A stale index still returns plausible results, which is exactly why it must
        be detected rather than trusted.
        """
        if corpus_content_hash == self.corpus_content_hash:
            return
        for record in self.components.values():
            if record.is_ready:
                record.status = ComponentStatus.STALE
                record.fallback = record.fallback or (
                    "component_disabled"
                    if record.requirement is Requirement.OPTIONAL
                    else "blocks_readiness"
                )

    def readiness(self) -> dict[str, object]:
        return {
            "ready": not self.blocking(),
            "blocking": self.blocking(),
            "degraded": self.degraded(),
            "active_fallbacks": self.active_fallbacks(),
        }

    # -- persistence -------------------------------------------------------

    def to_dict(self) -> dict[str, object]:
        return {
            "manifest_version": self.manifest_version,
            "bundle_version": self.bundle_version,
            "catalog_version": self.catalog_version,
            "corpus_content_hash": self.corpus_content_hash,
            "ingest_mode": self.ingest_mode,
            "corpus_granularity": self.corpus_granularity,
            "source_files": self.source_files,
            "instance_count": self.instance_count,
            "family_count": self.family_count,
            "config_hash": self.config_hash,
            "code_version": self.code_version,
            "created_at": self.created_at,
            "environment": self.environment,
            "components": {k: v.to_dict() for k, v in sorted(self.components.items())},
            "readiness": self.readiness(),
        }

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n")

    @classmethod
    def load(cls, path: Path) -> BundleManifest:
        payload = json.loads(Path(path).read_text())
        if payload.get("manifest_version") != MANIFEST_VERSION:
            raise ValueError(
                f"bundle manifest at {path} has version "
                f"{payload.get('manifest_version')!r}; expected {MANIFEST_VERSION!r}"
            )
        return cls(
            bundle_version=payload["bundle_version"],
            catalog_version=payload["catalog_version"],
            corpus_content_hash=payload["corpus_content_hash"],
            ingest_mode=payload["ingest_mode"],
            corpus_granularity=payload["corpus_granularity"],
            source_files=payload["source_files"],
            instance_count=payload["instance_count"],
            family_count=payload["family_count"],
            config_hash=payload["config_hash"],
            code_version=payload["code_version"],
            created_at=payload["created_at"],
            environment=payload.get("environment", {}),
            components={
                name: ComponentRecord.from_dict(record)
                for name, record in payload.get("components", {}).items()
            },
        )


def _encode_for_hash(value):
    from dataclasses import fields, is_dataclass

    if is_dataclass(value) and not isinstance(value, type):
        return {f.name: _encode_for_hash(getattr(value, f.name)) for f in fields(value)}
    if isinstance(value, (list, tuple)):
        return [_encode_for_hash(v) for v in value]
    if isinstance(value, dict):
        return {str(k): _encode_for_hash(v) for k, v in sorted(value.items())}
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Enum):
        return value.value
    return value


def index_config_hash(cfg) -> str:
    """Digest of the settings that change what is *stored* in an index.

    Deliberately narrower than `config_hash`. Only checkpoints, revisions and
    encoding parameters affect the vectors on disk; shortlist depths, risk
    thresholds and fusion weights change how a query is served, not what was
    encoded. Folding those into the bundle identity would force a full re-encode
    of 4000 documents every time a threshold was tuned.
    """
    relevant = {
        "dense": _encode_for_hash(cfg.retrieval.dense),
        "splade": _encode_for_hash(cfg.retrieval.splade),
        "prototypes": _encode_for_hash(cfg.retrieval.prototypes),
        "late_interaction": _encode_for_hash(cfg.retrieval.late_interaction),
    }
    return hashlib.sha256(
        json.dumps(relevant, sort_keys=True).encode("utf-8")
    ).hexdigest()[:16]


def config_hash(cfg) -> str:
    """Digest of every setting that can change retrieval results.

    Recorded in the manifest for reproducibility. Broader than
    `index_config_hash`: it covers serving behaviour as well as index content.
    """
    from dataclasses import fields, is_dataclass

    def encode(value):
        if is_dataclass(value) and not isinstance(value, type):
            return {f.name: encode(getattr(value, f.name)) for f in fields(value)}
        if isinstance(value, (list, tuple)):
            return [encode(v) for v in value]
        if isinstance(value, dict):
            return {str(k): encode(v) for k, v in sorted(value.items())}
        if isinstance(value, Path):
            return str(value)
        if isinstance(value, Enum):
            return value.value
        return value

    relevant = {
        name: encode(getattr(cfg, name))
        for name in (
            "ingest_mode", "corpus_granularity", "retrieval_mode",
            "retrieval", "shortlist", "rerank", "fusion", "decision", "risk",
            "bm25_title", "bm25_description", "bm25_fields", "bm25_prompts",
            "bm25_category", "bm25_tags", "bm25_data_source",
            "bm25_area_where_used", "bm25_field_descriptions",
            "bm25_business_objects",
        )
    }
    return hashlib.sha256(
        json.dumps(relevant, sort_keys=True).encode("utf-8")
    ).hexdigest()[:16]
