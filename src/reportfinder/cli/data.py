from __future__ import annotations

import argparse

from ..config import ROOT
from ..ingest.catalog import ReportCatalogLoader
from ..relevance.loaders import RelevanceDataLoader
from ._common import load_yaml


def main(argv=None):
    parser=argparse.ArgumentParser(prog="reportfinder-data"); sub=parser.add_subparsers(dest="command", required=True)
    validate=sub.add_parser("validate"); validate.add_argument("--config", required=True)
    args=parser.parse_args(argv); raw=load_yaml(args.config); root=ROOT/str(raw.get("relevance_root", "data/relevance"))
    catalog=ReportCatalogLoader().load(ROOT/str(raw.get("catalog_path", "data/Reports.xlsx")))
    data=RelevanceDataLoader(root).load(catalog=catalog)
    print(f"PASS scenarios={len(data.scenarios)} judgments={len(data.judgments)} catalog_rows={len(catalog)}")
    return 0
