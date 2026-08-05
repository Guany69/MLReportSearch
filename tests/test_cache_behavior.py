"""The cache signature must track what was *built*, not what was asked.

Hermetic on purpose: the signature hashes the source file's bytes, so these used
to require `data/Reports.xlsx` and failed outright wherever it is absent (CI).
The claim under test is about which config fields enter the key, which any file
can demonstrate -- so they use a tmp stand-in and now run everywhere.
"""

from __future__ import annotations

import pandas as pd
import pytest

from reportfinder.config import DEFAULT
from reportfinder.represent import _cache_signature, _dependency_version


@pytest.fixture
def source(tmp_path):
    """Any file with stable bytes; only its digest reaches the signature."""
    path = tmp_path / "corpus.xlsx"
    path.write_bytes(b"deterministic bytes for hashing")
    return path


def test_missing_dependency_version_is_safe():
    assert _dependency_version("definitely-not-an-installed-package") == "missing"


def test_signature_tracks_representation_config_not_query_thresholds(source):
    """A knob that changes doc(r) must invalidate the cache; a query knob must not.

    (This was parametrized over both ingest modes to assert mode-parity. The
    dual-file arm was permanently skipped, so the parametrize proved parity with
    nothing; the claim is about the signature, not about which workbook fed it.)
    """
    cfg = DEFAULT.with_overrides(ingest_mode="legacy_single_file", dense_mode="off")
    baseline = _cache_signature(cfg, source)

    assert _cache_signature(cfg.with_overrides(w_title=cfg.w_title + 1), source) != baseline
    assert _cache_signature(cfg.with_overrides(tau=cfg.tau / 2), source) == baseline


def test_signature_tracks_corpus_granularity(source):
    """Family and row-level corpora are different corpora and must not share a cache."""
    cfg = DEFAULT.with_overrides(
        ingest_mode="legacy_single_file", dense_mode="off",
        # Family granularity is incompatible with the `generators` default; this
        # test is about the cache key, so the mode is pinned rather than exercised.
        retrieval_mode="hybrid",
    )
    family = _cache_signature(cfg.with_overrides(corpus_granularity="family"), source)
    rows = _cache_signature(cfg.with_overrides(corpus_granularity="report_row"), source)
    assert family != rows


def test_signature_tracks_the_source_bytes(tmp_path):
    """Two different corpora must not collide, however identical the config."""
    cfg = DEFAULT.with_overrides(ingest_mode="legacy_single_file", dense_mode="off")
    first, second = tmp_path / "a.xlsx", tmp_path / "b.xlsx"
    first.write_bytes(b"corpus one")
    second.write_bytes(b"corpus two")

    assert _cache_signature(cfg, first) != _cache_signature(cfg, second)


def test_frame_pickle_round_trip_without_arrow(tmp_path):
    frame = pd.DataFrame({"fields": [("Worker", "Manager")], "title": ["Roster"]})
    path = tmp_path / "frame.pkl"
    frame.to_pickle(path)
    pd.testing.assert_frame_equal(frame, pd.read_pickle(path))
