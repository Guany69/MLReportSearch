"""CLI entry point.

    python -m reportfinder "show me terminated workers by supervisory organization"
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

from .config import DEFAULT
from .ingest.models import AmbiguityPolicy, IngestMode
from .model import ReportFinder
from .render import format_result
from .represent import load_or_build


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m reportfinder",
        description="Find the Workday-style custom report that answers a plain-English request.",
    )
    parser.add_argument("query", nargs="*", help="Plain-English request.")
    parser.add_argument(
        "--rebuild", action="store_true", help="Rebuild representations, ignoring cache."
    )
    parser.add_argument("--top-k", type=int, default=None, help="Candidates when ambiguous.")

    ingest = parser.add_argument_group("ingestion")
    ingest.add_argument(
        "--mode",
        choices=[m.value for m in IngestMode],
        default=None,
        help="legacy_single_file reads the Phase 1 Fields column; phase2_dual_file "
        "reconstructs fields from the catalog + field dictionary. "
        "Default: legacy_single_file.",
    )
    ingest.add_argument("--data", type=Path, default=None, help="Phase 1 workbook (legacy mode).")
    ingest.add_argument("--catalog", type=Path, default=None, help="Phase 2 report catalog.")
    ingest.add_argument(
        "--field-dictionary", type=Path, default=None, help="Phase 2 field dictionary."
    )
    ingest.add_argument(
        "--ambiguity-policy",
        choices=[p.value for p in AmbiguityPolicy],
        default=None,
        help="How to handle Where_Used names matching several catalog rows. "
        "permissive attaches to all candidates; strict withholds. Both flag the "
        "ambiguity in diagnostics.",
    )
    ingest.add_argument(
        "--fuzzy",
        action="store_true",
        default=None,
        help="Enable constrained fuzzy report-name matching (off by default).",
    )
    ingest.add_argument(
        "--no-composite",
        action="store_true",
        default=None,
        help="Disable business-object corroboration for ambiguous names.",
    )

    knobs = parser.add_argument_group("model knobs")
    knobs.add_argument("--t-dense", type=float, default=None, help="Dense softmax temperature T_d.")
    knobs.add_argument("--t-lsa", type=float, default=None, help="LSA softmax temperature T_l.")
    knobs.add_argument("--alpha", type=float, default=None, help="PoE mixture weight (1=dense only).")
    knobs.add_argument("--tau", type=float, default=None, help="Min top-1 probability for a confident answer.")
    knobs.add_argument("--delta", type=float, default=None, help="Min margin p1-p2 for a confident answer.")
    knobs.add_argument(
        "--field-expert",
        action="store_true",
        default=None,
        help="Enable the optional third (field-term) expert.",
    )

    parser.add_argument(
        "-v", "--verbose", action="store_true", help="Show posterior diagnostics."
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    cfg = DEFAULT.with_overrides(
        t_dense=args.t_dense,
        t_lsa=args.t_lsa,
        alpha=args.alpha,
        tau=args.tau,
        delta=args.delta,
        top_k=args.top_k,
        use_field_expert=args.field_expert,
        data_path=args.data,
        ingest_mode=args.mode,
        catalog_path=args.catalog,
        field_dictionary_path=args.field_dictionary,
        ambiguity_policy=args.ambiguity_policy,
        enable_fuzzy_match=args.fuzzy,
        enable_composite_match=(False if args.no_composite else None),
    )

    if not 0.0 <= cfg.alpha <= 1.0:
        print(f"error: --alpha must be in [0, 1], got {cfg.alpha}", file=sys.stderr)
        return 2

    query = " ".join(args.query).strip()
    if not query and not args.rebuild:
        print('error: provide a query, e.g. python -m reportfinder "terminated workers by supervisory org"',
              file=sys.stderr)
        return 2

    try:
        rep = load_or_build(cfg, rebuild=args.rebuild, verbose=True)
    except (FileNotFoundError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    if args.verbose and rep.import_summary is not None:
        print()
        print(rep.import_summary.render())

    if not query:
        print(f"Index ready: {len(rep)} report families.")
        return 0

    finder = ReportFinder(rep, cfg)
    t0 = time.perf_counter()
    result = finder.query(query)
    elapsed = time.perf_counter() - t0

    print()
    print(format_result(result, show_diagnostics=args.verbose))
    if args.verbose:
        print(f"   query answered in {elapsed * 1000:.0f} ms")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
