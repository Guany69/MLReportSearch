"""Structural concept specificity inferred entirely from corpus names."""

from __future__ import annotations


class ConceptHierarchy:
    def __init__(self, canonicals) -> None:
        tokenized = {str(value): tuple(str(value).casefold().split()) for value in canonicals}
        normalized_to_canonical = {" ".join(tokens): canonical for canonical, tokens in tokenized.items()}
        self._specializes: set[tuple[str, str]] = set()
        for specific, tokens in tokenized.items():
            for start in range(len(tokens)):
                for end in range(start + 1, len(tokens) + 1):
                    if end - start == len(tokens):
                        continue
                    generic = normalized_to_canonical.get(" ".join(tokens[start:end]))
                    if generic:
                        self._specializes.add((specific.casefold(), generic.casefold()))

    def specializes(self, a: str, b: str) -> bool:
        return (a.casefold(), b.casefold()) in self._specializes
