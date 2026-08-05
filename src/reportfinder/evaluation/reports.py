from __future__ import annotations

import json
from pathlib import Path


def write_summary(summary, path):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
