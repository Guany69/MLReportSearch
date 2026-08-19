"""The two-level report model: families and the instances that realize them.

A **ReportFamily** is the conceptual capability a user is asking for ("headcount by
organization"). A **ReportInstance** is one concrete, runnable catalog row. The
distinction matters because the estate contains 533 titles that appear on more than
one row with different field sets, prompts or data sources. Returning "the family"
without saying *which* instance answers the request is only half an answer, and
collapsing the instances loses the half that tells the user what to run.

Family identity is the **normalized title**, not title-plus-field-set. The latter
distinguishes 4000 catalog rows into 3999 families, which makes family aggregation a
no-op; the former yields 3280 families with up to 7 instances each, which is the
structure users actually experience.

Two fields are honest placeholders on this catalog and are marked as such rather
than invented:

* `lifecycle_state` -- the workbook carries no lifecycle column, so it is UNKNOWN.
* `acl_key` -- no entitlement source exists, so every instance carries the
  unrestricted sentinel. `auth.resolver` is what decides visibility; this field is
  the hook a real ACL source would populate.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from ..ingest.normalize import normalize_match

# No entitlement source exists for this catalog. Instances carry this sentinel so
# an ACL-aware resolver has a field to key on the day one arrives, and so that
# "unrestricted" is an explicit, greppable state rather than an empty string that
# could equally mean "not populated".
UNRESTRICTED_ACL_KEY = "__unrestricted__"


class LifecycleState(str, Enum):
    """Whether a report is current. UNKNOWN is the honest default here."""

    UNKNOWN = "UNKNOWN"
    ACTIVE = "ACTIVE"
    DEPRECATED = "DEPRECATED"
    RETIRED = "RETIRED"


@dataclass(frozen=True)
class Provenance:
    """Where a record came from, and by what route."""

    source_file: str
    source_row: int
    ingest_mode: str
    loader: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "source_file": self.source_file,
            "source_row": self.source_row,
            "ingest_mode": self.ingest_mode,
            "loader": self.loader,
        }


@dataclass(frozen=True)
class ReportInstance:
    """One concrete, executable catalog row."""

    report_instance_id: str
    source_system_id: str
    family_id: str
    title: str
    description: str
    category: str
    data_source: str
    report_type: str
    prompts: tuple[str, ...]
    fields: tuple[str, ...]
    tags: str
    area_where_used: str
    interface_metadata: dict[str, str]
    lifecycle_state: LifecycleState
    acl_key: str
    catalog_version: str
    provenance: Provenance

    @property
    def instance_id(self) -> str:
        """Short spelling used throughout the pipeline."""
        return self.report_instance_id


@dataclass(frozen=True)
class ReportFamily:
    """The conceptual report capability one or more instances realize."""

    family_id: str
    canonical_title: str
    normalized_title: str
    aliases: tuple[str, ...]
    lifecycle_state: LifecycleState
    catalog_version: str
    provenance: Provenance
    instance_ids: tuple[str, ...] = field(default_factory=tuple)


def content_hash(*parts: str) -> str:
    """Stable 16-hex digest over ordered text parts.

    for old cache artifacts
    """
    digest = hashlib.sha256()
    for part in parts:
        digest.update(part.encode("utf-8"))
        digest.update(b"\x1f")  # unit separator: "a","b" must not collide with "ab"
    return digest.hexdigest()[:16]


def _text(value: Any) -> str:
    """Frame cells arrive as str, NaN, None or numpy scalars. Normalize to text."""
    if value is None:
        return ""
    try:
        if value != value:  # noqa: PLR0124 - NaN is the only value unequal to itself
            return ""
    except (TypeError, ValueError):
        pass
    text = str(value).strip()
    return "" if text.casefold() in {"nan", "nat", "none", "null"} else text


def _sequence(value: Any) -> tuple[str, ...]:
    """Frame list-columns hold lists, tuples or a delimited string."""
    if value is None:
        return ()
    if isinstance(value, (list, tuple)):
        return tuple(_text(v) for v in value if _text(v))
    text = _text(value)
    return tuple(part.strip() for part in text.split(";") if part.strip()) if text else ()


def instance_from_row(
    row: Any,
    *,
    catalog_version: str,
    ingest_mode: str,
    source_file: str,
    loader: str,
) -> ReportInstance:
    """Build an instance from one row-level frame row.

    Requires row-level granularity: `report_key` is the stable instance identity and
    a family-collapsed frame does not have one.
    """
    instance_id = _text(row.get("report_key"))
    if not instance_id:
        raise ValueError(
            "ReportInstance requires a row-level corpus (corpus_granularity="
            "'report_row'); this frame has no 'report_key' column."
        )
    title = _text(row.get("title"))
    return ReportInstance(
        report_instance_id=instance_id,
        # This estate exports no source-system GUID. The physical spreadsheet row
        # is the most stable identifier available and is recorded as such; a real
        # deployment should map this to the source system's own key.
        source_system_id=f"{source_file}#row{_text(row.get('source_row')) or '?'}",
        family_id=_text(row.get("title_key")) or normalize_match(title),
        title=title,
        description=_text(row.get("description")),
        category=_text(row.get("category")),
        data_source=_text(row.get("data_source")),
        report_type=_text(row.get("report_type")),
        prompts=_sequence(row.get("prompts")),
        fields=_sequence(row.get("fields")),
        tags=_text(row.get("tags")),
        area_where_used=_text(row.get("area_where_used")),
        interface_metadata={
            "report_type": _text(row.get("report_type")),
            "worklet": _text(row.get("worklet")),
            "chart_type": _text(row.get("chart_type")),
            "landing_page": _text(row.get("landing_page")),
            "worklet_landing_pages": _text(row.get("worklet_landing_pages")),
            "shared": _text(row.get("shared")),
        },
        # The catalog has no lifecycle column. Deriving one from run counts would
        # be inventing a fact the source never stated.
        lifecycle_state=LifecycleState.UNKNOWN,
        acl_key=UNRESTRICTED_ACL_KEY,
        catalog_version=catalog_version,
        provenance=Provenance(
            source_file=source_file,
            source_row=int(_text(row.get("source_row")) or 0),
            ingest_mode=ingest_mode,
            loader=loader,
        ),
    )


def build_instances(
    frame: Any,
    *,
    catalog_version: str,
    ingest_mode: str,
    source_file: str,
    loader: str = "build_row_level_frame",
) -> tuple[ReportInstance, ...]:
    """Build every instance, in frame order.

    Frame order is the pipeline's positional index: generators return row positions
    and the union maps them back through this tuple.
    """
    return tuple(
        instance_from_row(
            frame.iloc[position],
            catalog_version=catalog_version,
            ingest_mode=ingest_mode,
            source_file=source_file,
            loader=loader,
        )
        for position in range(len(frame))
    )


def build_families(instances: tuple[ReportInstance, ...]) -> dict[str, ReportFamily]:
    """Group instances into families by normalized title.

    The canonical title is the first instance's raw title in frame order, which is
    deterministic. Aliases are the *other* distinct raw spellings in the family --
    real cosmetic variants (case, dashes, unicode form) rather than invented
    synonyms.
    """
    grouped: dict[str, list[ReportInstance]] = {}
    for instance in instances:
        grouped.setdefault(instance.family_id, []).append(instance)

    families: dict[str, ReportFamily] = {}
    for family_id, members in grouped.items():
        canonical = members[0].title
        aliases = tuple(
            dict.fromkeys(m.title for m in members if m.title and m.title != canonical)
        )
        families[family_id] = ReportFamily(
            family_id=family_id,
            canonical_title=canonical,
            normalized_title=family_id,
            aliases=aliases,
            lifecycle_state=LifecycleState.UNKNOWN,
            catalog_version=members[0].catalog_version,
            provenance=members[0].provenance,
            instance_ids=tuple(m.report_instance_id for m in members),
        )
    return families
