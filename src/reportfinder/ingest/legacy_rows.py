"""Row-level ingestion for the legacy single-file workbook.

The Phase 2 path reconstructs a report's fields by inverting the field dictionary's
`Where_Used` column, which is why it needs a linker. The legacy workbook needs no
such reconstruction: it carries a `Fields` column that already states, per row,
exactly which fields that report uses. This module turns that column into the same
`EnrichedField` records the linker produces, so `build_row_level_frame` can serve
both modes without knowing which workbook it was handed.

Why this exists at all: `data.load_corpus` collapses the legacy workbook to one row
per *family*, which discards the per-row identity the two-level Report Family /
Report Instance model is built on. Legacy mode needs an instance-level corpus for
the same reason Phase 2 does.

Field provenance in this mode is deliberately thin. The workbook gives a field's
*name* and nothing else, so business object, domain, categories and field-level
descriptions are empty. `MatchMethod.LEGACY_FIELDS_COLUMN` records that, so a
consumer can tell "this report genuinely has no business object" apart from "the
linker failed to find one".
"""

from __future__ import annotations

from .models import (
    AmbiguityStatus,
    EnrichedField,
    MatchMethod,
    ReportCatalogRecord,
)
from .normalize import normalize_match, split_multi

# The legacy Fields column is the report's own definition of what it selects, so
# the association is stated by the source rather than inferred. It is not a
# linker confidence estimate and must not be read as one.
LEGACY_LINK_CONFIDENCE = 1.0


def _field_key(field_name: str) -> str:
    """Identity for a legacy field.

    Phase 2 keys a field by ``business_object|field_name`` because the same field
    name means different things on different objects. The legacy workbook has no
    business object, so the object half is empty. The separator is kept so the two
    modes produce keys of the same shape.
    """
    return f"|{normalize_match(field_name)}"


def fields_from_column(
    records: list[ReportCatalogRecord],
    *,
    source_file: str = "",
) -> dict[int, list[EnrichedField]]:
    """Turn each record's `Fields` cell into the linker's output shape.

    Returns the same ``{row_index: [EnrichedField, ...]}`` mapping
    `ReportFieldLinker.link` produces, keyed by `ReportCatalogRecord.row_index` so
    it lines up with `build_row_level_frame`'s positional expectations.

    Field names are de-duplicated within a row by normalized identity: a report
    that lists the same field twice uses it once, and the repetition carries no
    retrieval signal.
    """
    linked: dict[int, list[EnrichedField]] = {}
    for record in records:
        linked[record.row_index] = [
            EnrichedField(
                business_object="",
                field_name=name,
                description="",
                report_field_type="",
                related_business_object_name="",
                built_in_prompts="",
                domain="",
                categories="",
                authorized_usage="",
                field_key=_field_key(name),
                match_method=MatchMethod.LEGACY_FIELDS_COLUMN,
                match_confidence=LEGACY_LINK_CONFIDENCE,
                ambiguity_status=AmbiguityStatus.RESOLVED,
                source_file=source_file or record.source.source_file,
                source_row=record.source.source_row,
            )
            for name in split_multi(record.fields_raw or "")
        ]
    return linked
