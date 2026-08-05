"""Corpus-derived canonical vocabulary built once per parser."""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass


def normalize_phrase(value: str) -> str:
    return " ".join(re.findall(r"[a-z0-9]+", str(value).casefold()))


@dataclass(frozen=True)
class CorpusVocabulary:
    normalized_field_to_canonical: dict[str, str]
    normalized_title_to_canonical: dict[str, str]
    normalized_category_to_canonical: dict[str, str]
    normalized_source_to_canonical: dict[str, str]
    normalized_object_to_canonical: dict[str, str]
    canonical_tokens: frozenset[str]
    canonical_phrases: frozenset[str]
    acronym_to_canonical_values: dict[str, tuple[str, ...]]
    token_df: dict[str, int]

    @classmethod
    def from_frame(cls, frame) -> CorpusVocabulary:
        def mapping(values) -> dict[str, str]:
            result: dict[str, str] = {}
            for value in values:
                surface = str(value).strip()
                normalized = normalize_phrase(surface)
                if normalized:
                    result.setdefault(normalized, surface)
            return result

        fields = mapping(field for values in frame["fields"] for field in values)
        titles = mapping(frame.get("title", ()))
        categories = mapping(frame.get("category", ()))
        sources = mapping(frame.get("data_source", ()))
        objects_set: set[str] = set()
        for metas in frame.get("field_meta", ()):
            for meta in metas if isinstance(metas, (list, tuple)) else ():
                value = getattr(meta, "business_object", "")
                if value:
                    objects_set.add(str(value))
        objects = mapping(objects_set)
        all_maps = (fields, titles, categories, sources, objects)
        phrases = frozenset(key for values in all_maps for key in values)
        tokens = frozenset(token for phrase in phrases for token in phrase.split())

        token_docs: Counter[str] = Counter()
        for _, row in frame.iterrows():
            row_tokens = set()
            for value in (
                str(row.get("title", "")),
                *(str(v) for v in row.get("fields", ())),
                str(row.get("category", "")),
                str(row.get("data_source", "")),
            ):
                row_tokens.update(normalize_phrase(value).split())
            token_docs.update(row_tokens)

        acronyms: dict[str, set[str]] = {}
        for values in all_maps:
            for phrase, canonical in values.items():
                words = phrase.split()
                if len(words) >= 2:
                    acronym = "".join(word[0] for word in words).upper()
                    if 2 <= len(acronym) <= 8:
                        acronyms.setdefault(acronym, set()).add(canonical)
        return cls(
            fields, titles, categories, sources, objects, tokens, phrases,
            {key: tuple(sorted(values)) for key, values in acronyms.items()},
            dict(token_docs),
        )

    def canonical(self, kind: str, value: str) -> str | None:
        normalized = normalize_phrase(value)
        lookup = {
            "field": self.normalized_field_to_canonical,
            "category": self.normalized_category_to_canonical,
            "data_source": self.normalized_source_to_canonical,
            "business_object": self.normalized_object_to_canonical,
        }.get(kind)
        if kind == "topic":
            return value
        return lookup.get(normalized) if lookup is not None else None
