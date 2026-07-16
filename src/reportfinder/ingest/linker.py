"""Report-field linker.

Reconstructs the report->fields relationship that Phase 1 read straight from the
`Fields` column, by inverting each field's `Where_Used` list.

The hard problem is identity, not parsing. Report titles are not unique in this
estate (533 titles occur on more than one catalog row), so ~30% of Where_Used
entries name a report that could be several different rows. Matching therefore runs
as an ordered precedence, and every link records how it was made:

    1. exact_name                 unique exact title match
    2. normalized_name            unique match after cosmetic normalization
    3. composite_business_object  name + Business Object == Data Source narrows to 1
    4. ambiguous_multi            still >1 candidate; see AmbiguityPolicy
    5. fuzzy                      opt-in only, blocked candidates, never Cartesian

Nothing here fabricates certainty: a link that could not be uniquely determined is
flagged AMBIGUOUS and keeps the full candidate list, regardless of whether the
policy chose to attach it.
"""

from __future__ import annotations

import collections
import difflib

from .models import (
    AmbiguityPolicy,
    AmbiguityStatus,
    EnrichedField,
    FieldDictionaryRecord,
    ImportSummary,
    LinkResult,
    MatchMethod,
    ReportCatalogRecord,
    ReportFieldLink,
    UnmatchedWhereUsed,
)
from .normalize import normalize_match

# Confidence is a coarse, ordinal signal about *how* the link was made -- it is not
# a probability and must never be presented to a user as certainty.
_CONFIDENCE = {
    MatchMethod.EXACT_NAME: 1.0,
    MatchMethod.NORMALIZED_NAME: 0.98,
    MatchMethod.COMPOSITE_BUSINESS_OBJECT: 0.90,
    MatchMethod.AMBIGUOUS_MULTI: 0.50,
    MatchMethod.FUZZY: 0.60,
}


class ReportFieldLinker:
    """Links field records to catalog rows via Where_Used."""

    def __init__(
        self,
        policy: AmbiguityPolicy = AmbiguityPolicy.PERMISSIVE,
        enable_composite: bool = True,
        enable_fuzzy: bool = False,
        fuzzy_threshold: float = 0.93,
    ):
        self.policy = policy
        self.enable_composite = enable_composite
        self.enable_fuzzy = enable_fuzzy
        self.fuzzy_threshold = fuzzy_threshold

    # -- indexes ----------------------------------------------------------

    @staticmethod
    def _build_indexes(reports: list[ReportCatalogRecord]):
        """Indexed lookups so linking is linear in entries, not reports x fields."""
        by_exact: dict[str, list[int]] = collections.defaultdict(list)
        by_normalized: dict[str, list[int]] = collections.defaultdict(list)
        for record in reports:
            by_exact[record.report_name].append(record.row_index)
            by_normalized[record.report_key].append(record.row_index)
        return by_exact, by_normalized

    # -- matching ---------------------------------------------------------

    def _candidates(self, name: str, by_exact, by_normalized) -> tuple[list[int], MatchMethod]:
        """Resolve a Where_Used name to candidate rows, cheapest method first."""
        exact = by_exact.get(name)
        if exact:
            return exact, MatchMethod.EXACT_NAME
        normalized = by_normalized.get(normalize_match(name))
        if normalized:
            return normalized, MatchMethod.NORMALIZED_NAME
        return [], MatchMethod.EXACT_NAME

    def _fuzzy_candidates(self, name: str, by_normalized) -> list[int]:
        """Opt-in fuzzy fallback with candidate blocking.

        Blocking key is the first token of the normalized name, which keeps this
        away from an unrestricted Cartesian comparison. Off by default.
        """
        key = normalize_match(name)
        if not key:
            return []
        block = key.split(" ", 1)[0]
        pool = [k for k in by_normalized if k.startswith(block)]
        if not pool:
            return []
        best = difflib.get_close_matches(key, pool, n=1, cutoff=self.fuzzy_threshold)
        return by_normalized[best[0]] if best else []

    @staticmethod
    def _narrow_by_business_object(
        candidates: list[int],
        reports: list[ReportCatalogRecord],
        business_object: str,
    ) -> list[int]:
        """Corroborate a name match with the field's business object.

        `Business Object` (field dictionary) and `Data Source` (catalog) share an
        identical 68-value vocabulary in this dataset, and narrowing on it never
        contradicted a name match across 12,629 resolutions -- strong evidence they
        denote the same thing. It remains an inference about the data rather than a
        documented guarantee, so it is confined to this one method and is
        switchable via `enable_composite`.
        """
        key = normalize_match(business_object)
        if not key:
            return candidates
        return [i for i in candidates if reports[i].data_source_key == key]

    # -- main -------------------------------------------------------------

    def link(
        self,
        reports: list[ReportCatalogRecord],
        field_records: list[FieldDictionaryRecord],
    ) -> LinkResult:
        by_exact, by_normalized = self._build_indexes(reports)

        links: list[ReportFieldLink] = []
        ambiguous: list[ReportFieldLink] = []
        unmatched: list[UnmatchedWhereUsed] = []
        linked: dict[int, list[EnrichedField]] = collections.defaultdict(list)

        summary = ImportSummary()
        seen_edges: set[tuple[int, str]] = set()
        fields_with_no_matches = 0
        fields_no_where_used = 0

        for field_record in field_records:
            if not field_record.where_used_report_names:
                fields_no_where_used += 1
                continue

            matched_any = False
            source_row = field_record.sources[0].source_row if field_record.sources else 0

            for name in field_record.where_used_report_names:
                summary.where_used_entries += 1
                candidates, method = self._candidates(name, by_exact, by_normalized)

                if not candidates and self.enable_fuzzy:
                    candidates = self._fuzzy_candidates(name, by_normalized)
                    if candidates:
                        method = MatchMethod.FUZZY

                if not candidates:
                    summary.unmatched_where_used += 1
                    unmatched.append(
                        UnmatchedWhereUsed(
                            report_name=name,
                            field_key=field_record.field_key,
                            source=field_record.sources[0],
                        )
                    )
                    continue

                all_candidates = tuple(candidates)

                # Ambiguous by name: try to corroborate with the business object.
                if len(candidates) > 1 and self.enable_composite:
                    narrowed = self._narrow_by_business_object(
                        candidates, reports, field_record.business_object
                    )
                    if len(narrowed) == 1:
                        candidates = narrowed
                        method = MatchMethod.COMPOSITE_BUSINESS_OBJECT
                    elif narrowed:
                        # Narrowing helped but did not decide it.
                        candidates = narrowed

                if len(candidates) == 1:
                    status = AmbiguityStatus.RESOLVED
                    targets = candidates
                else:
                    status = AmbiguityStatus.AMBIGUOUS
                    method = MatchMethod.AMBIGUOUS_MULTI
                    summary.ambiguous_links += 1
                    # STRICT withholds the attachment; the ambiguity is still
                    # recorded below either way.
                    targets = [] if self.policy is AmbiguityPolicy.STRICT else candidates

                link = ReportFieldLink(
                    report_key=reports[all_candidates[0]].report_key,
                    field_key=field_record.field_key,
                    match_method=method,
                    match_confidence=_CONFIDENCE[method],
                    ambiguity_status=status,
                    source_field_row=source_row,
                    report_row_index=targets[0] if targets else -1,
                    candidate_row_indices=all_candidates,
                )
                if status is AmbiguityStatus.AMBIGUOUS:
                    ambiguous.append(link)
                links.append(link)

                if method is MatchMethod.EXACT_NAME:
                    summary.exact_name_links += 1
                elif method is MatchMethod.NORMALIZED_NAME:
                    summary.normalized_name_links += 1
                elif method is MatchMethod.COMPOSITE_BUSINESS_OBJECT:
                    summary.composite_links += 1
                elif method is MatchMethod.FUZZY:
                    summary.fuzzy_links += 1

                for row_index in targets:
                    edge = (row_index, field_record.field_key)
                    if edge in seen_edges:
                        summary.duplicate_links_removed += 1
                        continue
                    seen_edges.add(edge)
                    matched_any = True
                    linked[row_index].append(
                        _to_enriched(field_record, method, status, source_row)
                    )

            if not matched_any:
                fields_with_no_matches += 1

        # Deterministic ordering: field lists must not depend on dict iteration or
        # row order, or the same input could produce different documents.
        for row_index in linked:
            linked[row_index].sort(key=lambda f: (f.business_object, f.field_name))

        summary.report_field_links_created = len(seen_edges)
        summary.fields_with_no_matched_reports = fields_with_no_matches
        summary.fields_with_no_where_used = fields_no_where_used
        summary.reports_with_zero_fields = sum(
            1 for r in reports if not linked.get(r.row_index)
        )

        return LinkResult(
            linked_reports=dict(linked),
            report_field_links=links,
            unmatched_where_used=unmatched,
            ambiguous_matches=ambiguous,
            validation_metrics=summary,
        )


def _to_enriched(
    record: FieldDictionaryRecord,
    method: MatchMethod,
    status: AmbiguityStatus,
    source_row: int,
) -> EnrichedField:
    source = record.sources[0]
    return EnrichedField(
        business_object=record.business_object,
        field_name=record.field_name,
        description=record.description,
        report_field_type=record.report_field_type,
        related_business_object_name=record.related_business_object_name,
        built_in_prompts=record.built_in_prompts,
        domain=record.domain,
        categories=record.categories,
        authorized_usage=record.authorized_usage,
        field_key=record.field_key,
        match_method=method.value,
        match_confidence=_CONFIDENCE[method],
        ambiguity_status=status.value,
        source_file=source.source_file,
        source_row=source_row,
    )
