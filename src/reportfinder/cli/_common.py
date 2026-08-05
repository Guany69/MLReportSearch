from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

import yaml


def load_yaml(path): return yaml.safe_load(Path(path).read_text()) or {}
def sha256(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def metadata(command, config, **extra):
    try: commit = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True).stdout.strip()
    except Exception: commit = None
    return {"started_at": datetime.now(UTC).isoformat(), "command": command,
            "config": config, "python": sys.version, "git_commit": commit, **extra}
def new_directory(path):
    path = Path(path)
    if path.exists(): raise FileExistsError(f"refusing to overwrite existing artifact directory: {path}")
    path.mkdir(parents=True); return path
def write_json(path, value): Path(path).write_text(json.dumps(value, indent=2, sort_keys=True, default=str)+"\n")
