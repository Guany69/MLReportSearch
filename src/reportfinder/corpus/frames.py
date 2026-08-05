"""The boundary between the pandas frame and the typed corpus model.

Everything downstream of here works with `ReportInstance` / `ReportFamily` objects
and integer positions. This module is the only place that knows the frame's column
names, so a schema change has one place to land rather than forty `.iloc[i]["title"]`
call sites.

Positions matter: generators score frame rows and return integer positions, and the
union maps those back to instance ids. `CorpusModel.instances` is in frame order and
`position_of` is its inverse, so that mapping is stated once and asserted in tests.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .identity import ReportFamily, ReportInstance, build_families, build_instances
from .views import (
    InstanceView,
    ViewType,
    build_views,
    corpus_content_hash,
)


def catalog_version(path: Path | str) -> str:
    """Content version of the source workbook.

    A digest rather than a number: the estate has no version column, and file
    mtime is not stable across checkouts. Missing files yield an explicit marker
    instead of raising, so a bundle can record *why* it could not be built.
    """
    source = Path(path)
    if not source.exists():
        return "absent"
    digest = hashlib.sha256()
    with source.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()[:16]


@dataclass(frozen=True)
class CorpusModel:
    """The typed, two-level view of a built corpus."""

    instances: tuple[ReportInstance, ...]
    families: dict[str, ReportFamily]
    views: dict[str, dict[ViewType, InstanceView]]
    catalog_version: str
    content_hash: str
    ingest_mode: str
    source_file: str

    def __len__(self) -> int:
        return len(self.instances)

    @property
    def instance_ids(self) -> tuple[str, ...]:
        return tuple(i.report_instance_id for i in self.instances)

    def position_of(self, instance_id: str) -> int:
        return self._positions[instance_id]

    def instance(self, instance_id: str) -> ReportInstance:
        return self.instances[self._positions[instance_id]]

    def family_of(self, instance_id: str) -> ReportFamily:
        return self.families[self.instance(instance_id).family_id]

    def view_texts(self, view_type: ViewType) -> list[str]:
        """View text for every instance, in frame order."""
        return [self.views[i.report_instance_id][view_type].text for i in self.instances]

    def view_hashes(self, view_type: ViewType) -> list[str]:
        return [
            self.views[i.report_instance_id][view_type].content_hash
            for i in self.instances
        ]

    @property
    def _positions(self) -> dict[str, int]:
        # Computed once and cached on the frozen instance via object.__setattr__.
        cached = self.__dict__.get("_position_cache")
        if cached is None:
            cached = {
                instance.report_instance_id: position
                for position, instance in enumerate(self.instances)
            }
            object.__setattr__(self, "_position_cache", cached)
        return cached


def build_corpus_model(
    frame: Any,
    *,
    ingest_mode: str,
    source_file: str,
) -> CorpusModel:
    """Turn a row-level frame into the typed two-level model.

    Raises if handed a family-collapsed frame: that shape has no instance identity,
    and silently degrading would produce a corpus whose "instances" are already
    merged.
    """
    if "report_key" not in getattr(frame, "columns", ()):
        raise ValueError(
            "build_corpus_model requires a row-level corpus "
            "(corpus_granularity='report_row'); got a frame without 'report_key'."
        )

    version = catalog_version(source_file)
    instances = build_instances(
        frame,
        catalog_version=version,
        ingest_mode=ingest_mode,
        source_file=str(source_file),
    )
    families = build_families(instances)

    views = {
        instance.report_instance_id: build_views(
            instance, aliases=families[instance.family_id].aliases
        )
        for instance in instances
    }

    return CorpusModel(
        instances=instances,
        families=families,
        views=views,
        catalog_version=version,
        content_hash=corpus_content_hash(views),
        ingest_mode=ingest_mode,
        source_file=str(source_file),
    )
