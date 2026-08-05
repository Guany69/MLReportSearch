from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path

from ..config import ROOT, from_mapping
from ..ingest.catalog import ReportCatalogLoader
from ..relevance.loaders import RelevanceDataLoader
from ._common import load_yaml

# What the ingest validation artifact is, stated inside the artifact itself. These
# are counts of what was read and linked. Nothing here measures whether a linked
# field is the *right* field, and a reader who mistakes an import count for a
# quality metric will read 75,000 links as 75,000 correct answers.
INGEST_BASIS = (
    "import counts from the supplied workbooks; not relevance or quality metrics"
)


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()[:16]


def _validate(args) -> int:
    raw = load_yaml(args.config)
    root = ROOT / str(raw.get("relevance_root", "data/relevance"))
    catalog = ReportCatalogLoader().load(
        ROOT / str(raw.get("catalog_path", "data/Reports.xlsx"))
    )
    data = RelevanceDataLoader(root).load(catalog=catalog)
    print(
        f"PASS scenarios={len(data.scenarios)} judgments={len(data.judgments)} "
        f"catalog_rows={len(catalog)}"
    )
    return 0


def _phase2_ingest(args) -> int:
    """Run dual-file ingestion and record what it read, machine-readably.

    The human-readable `ImportSummary.render()` is printed for the operator; the
    JSON exists so the counts can be diffed across runs and referenced from docs
    without being retyped -- retyped numbers are how a count becomes a claim.
    """
    from ..ingest import build_corpus

    cfg = from_mapping(load_yaml(args.config))
    if cfg.ingest_mode != "phase2_dual_file":
        print(
            f"error: {args.config} selects ingest_mode={cfg.ingest_mode!r}; this "
            "subcommand reports on dual-file ingestion only"
        )
        return 2

    _, summary = build_corpus(cfg, verbose=False)
    print(summary.render())

    sources = [cfg.catalog_path, cfg.field_dictionary_path]
    payload = {
        "basis": INGEST_BASIS,
        "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "config": str(args.config),
        "sources": [
            {"path": str(p), "sha256": _digest(Path(p))}
            for p in sources if Path(p).exists()
        ],
        "counts": dataclasses.asdict(summary),
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(f"\nwrote {out}")
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(prog="reportfinder-data")
    sub = parser.add_subparsers(dest="command", required=True)

    validate = sub.add_parser("validate")
    validate.add_argument("--config", required=True)

    phase2 = sub.add_parser(
        "phase2-ingest", help="Dual-file ingest counts, human- and machine-readable."
    )
    phase2.add_argument("--config", default="configs/phase2_generators.yaml")
    phase2.add_argument(
        "--out", type=Path, default=Path("artifacts/phase2_ingest_validation.json")
    )

    args = parser.parse_args(argv)
    if args.command == "phase2-ingest":
        return _phase2_ingest(args)
    return _validate(args)
