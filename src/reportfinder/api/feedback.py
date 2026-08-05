"""Append-only impression store.

What is recorded is an **impression**, not a relevance label. A click is confounded
by position, by presentation, and by the slate the ranker chose to show -- treating
one as a judgement bakes the current ranker's bias into its own successor. Recording
the whole slate, the positions and the policy versions is what makes any later
de-biasing possible at all; storing only the click makes it impossible forever.

JSONL because it is append-only, survives a crash mid-write with at most one bad
line, and needs no schema migration when a field is added.
"""

from __future__ import annotations

import json
import threading
from dataclasses import dataclass
from pathlib import Path


@dataclass
class FeedbackStore:
    """Append-only JSONL with idempotent writes."""

    path: Path
    _lock: threading.Lock = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        self.path = Path(self.path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        object.__setattr__(self, "_lock", threading.Lock())
        self._seen = self._load_keys()

    def _load_keys(self) -> set[str]:
        """Idempotency keys already on disk.

        Read at startup rather than kept only in memory: a retry that arrives after
        a restart is exactly the case idempotency exists for.
        """
        if not self.path.exists():
            return set()
        keys: set[str] = set()
        with self.path.open() as stream:
            for line in stream:
                line = line.strip()
                if not line:
                    continue
                try:
                    keys.add(json.loads(line)["idempotency_key"])
                except (json.JSONDecodeError, KeyError):
                    # A truncated final line from an interrupted write. Skipped
                    # rather than fatal -- one lost impression must not stop the
                    # service from accepting new ones.
                    continue
        return keys

    def record(self, payload: dict) -> bool:
        """Append one impression. Returns False if this key was already stored."""
        key = payload["idempotency_key"]
        with self._lock:
            if key in self._seen:
                return False
            with self.path.open("a") as stream:
                stream.write(json.dumps(payload, sort_keys=True) + "\n")
            self._seen.add(key)
            return True

    def __len__(self) -> int:
        return len(self._seen)
