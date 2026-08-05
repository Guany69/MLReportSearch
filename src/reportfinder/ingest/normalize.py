"""Shared normalization primitives for ingestion.

Two rules govern everything here:

1. **Display values are never mutated.** Normalization produces a *separate* matching
   value. What the user sees is always the original spreadsheet text.
2. **Normalization never removes meaning.** Versions, suffixes and prefixes are
   load-bearing in this estate -- "Report", "Report v2", "Report - Copy",
   "Report - Summary" are four different reports. We only fold differences that are
   genuinely cosmetic (case, unicode form, whitespace runs, dash glyphs).
"""

from __future__ import annotations

import re
import unicodedata

# Dash glyphs that are cosmetically interchangeable in report titles.
# en-dash, em-dash, non-breaking hyphen, minus sign -> plain hyphen.
_DASHES = {
    "–": "-",
    "—": "-",
    "‑": "-",
    "−": "-",
}
_DASH_RE = re.compile("|".join(map(re.escape, _DASHES)))

_WHITESPACE_RE = re.compile(r"\s+")

# Where_Used may be separated by semicolons or line breaks (CRLF or LF).
# Commas are deliberately NOT a delimiter: report names legitimately contain them
# (e.g. "Headcount by Region, Country and Company").
_MULTI_SPLIT_RE = re.compile(r"[;\r\n]+")

_MISSING = {"", "nan", "none", "nat", "null", "n/a"}


def is_blank(value: object) -> bool:
    """True for blank cells, including pandas' stringified NaN/NaT sentinels."""
    if value is None:
        return True
    try:
        # Catches float('nan') without importing pandas here.
        if value != value:  # noqa: PLR0124 - NaN is the only value unequal to itself
            return True
    except (TypeError, ValueError):
        pass
    return str(value).strip().casefold() in _MISSING


def normalize_display(value: object) -> str:
    """The value as shown to a human: original text, only stripped of edge whitespace.

    Blank-ish cells collapse to "" so downstream code never has to special-case
    the string "nan".
    """
    if is_blank(value):
        return ""
    return str(value).strip()


def normalize_match(value: object) -> str:
    """The value used for matching only. Never displayed, never stored as truth.

    NFKC -> dash standardization -> whitespace collapse -> casefold.
    Meaningful tokens (numbers, versions, suffixes) are preserved verbatim.
    """
    if is_blank(value):
        return ""
    text = unicodedata.normalize("NFKC", str(value))
    text = _DASH_RE.sub(lambda m: _DASHES[m.group()], text)
    text = _WHITESPACE_RE.sub(" ", text).strip()
    return text.casefold()


def normalize_header(value: object) -> str:
    """Normalize a column header for alias resolution.

    Headers vary in case, spacing and separator style across exports, so
    "Report Tag(s)", "report_tags" and "REPORT TAGS" all reduce to "report tag s"
    -style keys. Resolution is by normalized name, never by column position.
    """
    if is_blank(value):
        return ""
    text = unicodedata.normalize("NFKC", str(value))
    text = _DASH_RE.sub(lambda m: _DASHES[m.group()], text)
    text = text.replace("_", " ")
    text = re.sub(r"[^0-9a-zA-Z]+", " ", text)
    return _WHITESPACE_RE.sub(" ", text).strip().casefold()


def split_multi(value: object) -> list[str]:
    """Split a multi-valued cell on ';' or line breaks, preserving order.

    Duplicates are removed but first-appearance order is kept, so provenance and
    display order stay faithful to the source cell. Commas are never split on.
    """
    if is_blank(value):
        return []
    parts = _MULTI_SPLIT_RE.split(str(value))
    out: list[str] = []
    seen: set[str] = set()
    for part in parts:
        cleaned = part.strip()
        if not cleaned:
            continue
        key = normalize_match(cleaned)
        if key in seen:
            continue
        seen.add(key)
        out.append(cleaned)
    return out


def resolve_columns(
    columns: list[str], aliases: dict[str, tuple[str, ...]]
) -> dict[str, str]:
    """Map logical field names to actual column names via normalized headers.

    `aliases` maps a logical name to the header spellings it may appear under.
    Returns only the logical names that were found, so callers can distinguish
    "missing required column" from "missing optional column".
    """
    by_normalized: dict[str, str] = {}
    for col in columns:
        key = normalize_header(col)
        # First spelling wins: duplicate headers in a sheet are a data problem,
        # and silently preferring the later one would hide it.
        by_normalized.setdefault(key, col)

    resolved: dict[str, str] = {}
    for logical, spellings in aliases.items():
        for spelling in spellings:
            actual = by_normalized.get(normalize_header(spelling))
            if actual is not None:
                resolved[logical] = actual
                break
    return resolved
