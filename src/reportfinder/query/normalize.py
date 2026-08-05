"""Query tokenization that preserves offsets into the user's original text."""

from __future__ import annotations

import re
from dataclasses import dataclass

_TOKEN_RE = re.compile(r"[^\W_]+", re.UNICODE)


@dataclass(frozen=True)
class Token:
    text: str
    surface: str
    char_start: int
    char_end: int


def tokenize(query: str) -> tuple[Token, ...]:
    return tuple(
        Token(m.group(0).casefold(), m.group(0), m.start(), m.end())
        for m in _TOKEN_RE.finditer(query)
    )


def depluralize(token: str) -> str:
    value = token.casefold()
    if len(value) <= 3:
        return value
    if value.endswith("ies") and len(value) > 4:
        return value[:-3] + "y"
    for suffix in ("ches", "shes", "ses", "xes", "zes"):
        if value.endswith(suffix) and len(value) > len(suffix) + 2:
            return value[: -len(suffix)]
    if value.endswith("s") and not value.endswith(("ss", "us", "is")):
        return value[:-1]
    return value
