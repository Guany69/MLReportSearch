"""Four independently searchable views of a report instance.

A single blended document per report forces one embedding to represent everything
the report *is*, everything it is *for*, everything it *contains* and how it is
*presented*. Queries do not work that way: "why are we losing people" is a purpose
question, "termination reason and time to fill" is a schema question, and "worklet
on the People landing page" is an interface question. Splitting the text lets each
be retrieved on its own terms, and lets a report that only matches on schema still
reach the ranker.

The split also makes re-embedding cheap: a title correction changes only the
identity view's content hash, so the other three keep their vectors.

**What must never enter a view.** Owner, creator, last-run user, run counts, run
dates and available-usage are operational metadata, not report meaning. Embedding
them teaches the retriever that popular reports are semantically closer to every
query. They remain available as ranking features, lifecycle policy and bounded
tie-breakers -- but never as text. `FORBIDDEN_VIEW_SOURCES` names them and
`test_views.py` asserts none of their values appear in any view.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from enum import Enum

from .identity import ReportInstance, content_hash

# Instance attributes and frame columns that carry operational meaning rather than
# report meaning. None of these may be rendered into view text.
FORBIDDEN_VIEW_SOURCES = (
    "owner",
    "created_by",
    "created_date",
    "last_run_by",
    "last_run",
    "last_updated",
    "runs",
    "runs_total",
    "available_usage",
)

# Views are versioned: changing a template changes every content hash, which must
# invalidate every dense index built from the old text.
VIEW_SCHEMA_VERSION = "1"


class ViewType(str, Enum):
    IDENTITY = "identity"
    PURPOSE = "purpose"
    SCHEMA = "schema"
    INTERFACE = "interface"


ALL_VIEW_TYPES = tuple(ViewType)


@dataclass(frozen=True)
class InstanceView:
    """One view's text plus the metadata needed to index and re-index it."""

    instance_id: str
    view_type: ViewType
    text: str
    token_count: int
    content_hash: str
    catalog_version: str
    provenance: dict[str, object]
    # Set by the index when the view is embedded; empty until then. Keeping it on
    # the view means a stored vector can always name the checkpoint that produced it.
    embedding_revision: str = ""

    def with_embedding_revision(self, revision: str) -> InstanceView:
        return replace(self, embedding_revision=revision)

    @property
    def is_empty(self) -> bool:
        """A view with no text carries no signal and must not be embedded."""
        return not self.text.strip()


def _join(parts: list[str]) -> str:
    """Join non-empty labelled parts deterministically."""
    return " | ".join(part for part in parts if part and part.strip())


def _labelled(label: str, value: str) -> str:
    return f"{label}: {value}" if value and value.strip() else ""


def _identity_text(instance: ReportInstance, aliases: tuple[str, ...]) -> str:
    return _join([
        _labelled("Title", instance.title),
        _labelled("Also known as", "; ".join(aliases)),
        _labelled("Tags", instance.tags),
        _labelled("Category", instance.category),
    ])


def _purpose_text(instance: ReportInstance) -> str:
    """Why the report exists and where it is used.

    This estate's `Description` column holds only 8 distinct values across 4000
    rows, so on this catalog the purpose view is thin and often empty. That is a
    property of the data, not of the design -- the index build measures it and
    records a degenerate status rather than pretending the view is informative.
    """
    return _join([
        _labelled("Purpose", instance.description),
        _labelled("Used in", instance.area_where_used),
    ])


def _schema_text(instance: ReportInstance) -> str:
    return _join([
        _labelled("Fields", "; ".join(instance.fields)),
        _labelled("Prompts", "; ".join(instance.prompts)),
        _labelled("Data source", instance.data_source),
    ])


def _interface_text(instance: ReportInstance) -> str:
    meta = instance.interface_metadata
    return _join([
        _labelled("Report type", meta.get("report_type", "")),
        _labelled("Worklet", meta.get("worklet", "")),
        _labelled("Chart type", meta.get("chart_type", "")),
        _labelled("Landing page", meta.get("landing_page", "")),
        _labelled("Worklet landing pages", meta.get("worklet_landing_pages", "")),
    ])


_BUILDERS = {
    ViewType.PURPOSE: _purpose_text,
    ViewType.SCHEMA: _schema_text,
    ViewType.INTERFACE: _interface_text,
}


def build_view(
    instance: ReportInstance,
    view_type: ViewType,
    *,
    aliases: tuple[str, ...] = (),
) -> InstanceView:
    """Render one view. Deterministic: same instance in, same hash out."""
    if view_type is ViewType.IDENTITY:
        text = _identity_text(instance, aliases)
    else:
        text = _BUILDERS[view_type](instance)

    return InstanceView(
        instance_id=instance.report_instance_id,
        view_type=view_type,
        text=text,
        # A whitespace count, not a model token count: this is for budgeting and
        # diagnostics, and must not be read as a tokenizer result.
        token_count=len(text.split()),
        content_hash=content_hash(VIEW_SCHEMA_VERSION, view_type.value, text),
        catalog_version=instance.catalog_version,
        provenance=instance.provenance.as_dict(),
    )


def build_views(
    instance: ReportInstance,
    *,
    aliases: tuple[str, ...] = (),
) -> dict[ViewType, InstanceView]:
    return {
        view_type: build_view(instance, view_type, aliases=aliases)
        for view_type in ALL_VIEW_TYPES
    }


def corpus_content_hash(views_by_instance: dict[str, dict[ViewType, InstanceView]]) -> str:
    """One digest over every instance's view hashes.

    This is what index components key on: if it is unchanged, every stored vector
    is still valid. Sorted by instance id so frame order cannot change the result.
    """
    parts: list[str] = []
    for instance_id in sorted(views_by_instance):
        views = views_by_instance[instance_id]
        parts.append(instance_id)
        for view_type in ALL_VIEW_TYPES:
            view = views.get(view_type)
            parts.append(view.content_hash if view is not None else "")
    return content_hash(*parts)


def authoritative_text(instance: ReportInstance, *, max_chars: int = 1200) -> str:
    """The bounded report representation the cross-encoder reads.

    Deliberately assembled from catalog fields only. Generated query prototypes are
    never included: they are retrieval aids, not evidence about what the report is,
    and letting them in would let generated language justify its own match.
    """
    text = _join([
        _labelled("Title", instance.title),
        _labelled("Category", instance.category),
        _labelled("Purpose", instance.description),
        _labelled("Used in", instance.area_where_used),
        _labelled("Data source", instance.data_source),
        _labelled("Report type", instance.report_type),
        _labelled("Fields", "; ".join(instance.fields)),
        _labelled("Prompts", "; ".join(instance.prompts)),
        _labelled("Tags", instance.tags),
    ])
    if len(text) <= max_chars:
        return text
    # Truncate on a separator so the model never sees half a field name.
    cut = text.rfind(" | ", 0, max_chars)
    return text[: cut if cut > 0 else max_chars]
