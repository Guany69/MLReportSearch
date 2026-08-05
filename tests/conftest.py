"""Shared guards and markers.

Two things live here:

1. `requires_real_estate` -- `data/Reports.xlsx` is the estate everything runs
   against, and it is gitignored, so it is present locally and absent in CI.
   Tests needing it skip *there* rather than failing.

   This replaces the old `requires_phase2_workbooks`. Those workbooks are not in
   this tree and 30 tests were skipping for them while asserting claims that hold
   perfectly well against `Reports.xlsx` -- the guard was standing in for an
   ingest-mode string in a fixture, not for anything the assertions needed.

2. The `integration` marker for tests that load real model checkpoints. These are
   deselected by default via `addopts` in `pyproject.toml`, so a plain `pytest`
   run stays fast and offline; run them with `uv run pytest -m integration`.

   That deselection is new. The claim was here before it was true: nothing
   implemented it, so the two ONNX integration tests ran on every plain `pytest`
   invocation whenever the exported graph happened to exist -- loading a real
   cross-encoder inside a run documented as offline. A second, unused
   `RF_INTEGRATION` env guard existed alongside the marker and has been removed
   rather than left as a decoy.
"""

from __future__ import annotations

import pytest

from reportfinder.config import DEFAULT

requires_real_estate = pytest.mark.skipif(
    not DEFAULT.data_path.exists(),
    reason=(
        f"{DEFAULT.data_path.name} absent from data/ -- blocked, not regressed. "
        "The workbook is gitignored; place it in data/ to run these."
    ),
)


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line(
        "markers",
        "integration: loads real model checkpoints; run with `pytest -m integration`",
    )
