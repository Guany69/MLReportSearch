from __future__ import annotations

import argparse

from ..config import ROOT
from ..ranking.features import FEATURE_HASH, FEATURE_NAMES, FEATURE_SCHEMA_VERSION
from ..relevance.loaders import RelevanceDataLoader
from ..relevance.splits import SplitGuard
from ._common import new_directory, write_json


def main(argv=None):
    p=argparse.ArgumentParser(prog="reportfinder-features"); sub=p.add_subparsers(dest="command", required=True)
    build=sub.add_parser("build"); build.add_argument("--split", choices=["train","validation","test"], required=True); build.add_argument("--out"); build.add_argument("--i-am-releasing", action="store_true")
    a=p.parse_args(argv); data=RelevanceDataLoader(ROOT/"data/relevance").load(validate_processed=False)
    if a.split=="test" and not a.i_am_releasing: raise SystemExit("sealed test feature access requires --i-am-releasing")
    SplitGuard(data.splits).assert_allowed(data.splits[a.split], "feature_fitting")
    out=new_directory(a.out or ROOT/f"artifacts/features/schema-{FEATURE_SCHEMA_VERSION}-{a.split}")
    write_json(out/"manifest.json", {"split": a.split, "scenario_count": len(data.splits[a.split]), "feature_schema_version": FEATURE_SCHEMA_VERSION, "feature_hash": FEATURE_HASH, "feature_names": FEATURE_NAMES,
        "status": "manifest-only; runtime-consistent pair generation is performed by the training pipeline"})
    print(out); return 0
