"""Pure-Python token phrase trie returning every terminal match."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from .normalize import Token, depluralize

# token -> child node, plus the _END sentinel -> list of payloads.
TrieNode = dict[object, object]


@dataclass(frozen=True)
class Match:
    tok_start: int
    tok_end: int
    payload: object
    morphological: bool = False


class PhraseTrie:
    _END = object()

    def __init__(self) -> None:
        # A node maps a token to a child node, and the _END sentinel to the list
        # of payloads terminating there. Two value shapes in one dict, so the
        # annotation is deliberately loose and narrowed at each use.
        self.root: TrieNode = {}

    def add(self, tokens: tuple[str, ...], payload: object) -> None:
        if not tokens:
            return
        node = self.root
        for token in tokens:
            child = node.setdefault(token.casefold(), {})
            assert isinstance(child, dict)
            node = child
        terminals = node.setdefault(self._END, [])
        assert isinstance(terminals, list)
        terminals.append(payload)

    @staticmethod
    def _variants(token: str) -> tuple[tuple[str, bool], ...]:
        candidates = [(token, False)]
        singular = depluralize(token)
        if singular != token:
            candidates.append((singular, True))
        if len(token) > 3:
            candidates.append((token + "s", True))
        return tuple(dict.fromkeys(candidates))

    def find_all(self, tokens: Sequence[Token]) -> list[Match]:
        found: list[Match] = []
        for start in range(len(tokens)):
            states: list[tuple[dict, bool]] = [(self.root, False)]
            for end in range(start, len(tokens)):
                next_states: list[tuple[dict, bool]] = []
                for node, changed in states:
                    for variant, morph in self._variants(tokens[end].text):
                        child = node.get(variant)
                        if isinstance(child, dict):
                            next_states.append((child, changed or morph))
                if not next_states:
                    break
                dedup: dict[int, tuple[dict, bool]] = {}
                for node, changed in next_states:
                    old = dedup.get(id(node))
                    dedup[id(node)] = (node, changed if old is None else old[1] and changed)
                states = list(dedup.values())
                for node, changed in states:
                    for payload in node.get(self._END, ()):
                        found.append(Match(start, end + 1, payload, changed))
        return found
