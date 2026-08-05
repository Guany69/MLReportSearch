"""HTTP service over the search pipeline.

Importing this package requires the `api` extra (`uv sync --extra api`). It is
optional so the CLI, the Streamlit app and the test suite do not depend on a web
framework they never use.

    uv run uvicorn reportfinder.api.service:app --port 8080
"""

from __future__ import annotations

__all__ = ["schemas", "service"]
