"""Generate the v2 synthetic label set.

    uv run python scripts/generate_labels_v2.py --out data/relevance_v2 -v

Thin wrapper; everything lives in `reportfinder.relevance.synthesize` so the
generator is importable and testable rather than script-only.
"""

from __future__ import annotations

from reportfinder.relevance.synthesize import main

if __name__ == "__main__":
    raise SystemExit(main())
