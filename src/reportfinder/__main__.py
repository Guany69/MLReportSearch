"""CLI entry point.

    python -m reportfinder "show me terminated workers by supervisory organization"
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from .config import DEFAULT
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
        "--config", type=Path, default=None,
        help=(
            "YAML config used as the base; individual flags still override it. "
            "Loaded through the same mapping logic as `reportfinder-bundle`, so a "
            "search runs the configuration its bundle was built with. The API "
            "server reads REPORTFINDER_CONFIG instead."
        ),
    )
    parser.add_argument(
        "--rebuild", action="store_true", help="Rebuild representations, ignoring cache."
    )
    parser.add_argument("--top-k", type=int, default=None, help="Candidates when ambiguous.")

    ingest = parser.add_argument_group("ingestion")
    ingest.add_argument(
        "--data", type=Path, default=None,
        help="Report workbook. Default: data/Reports.xlsx.",
    )

    knobs = parser.add_argument_group("model knobs")
    knobs.add_argument("--retrieval-mode",
                       choices=["generators", "hybrid", "legacy_weighted_logit"], default=None,
                       help="Search architecture. 'generators' is the source-preserving "
                            "pipeline and needs a built bundle (reportfinder-bundle build); "
                            "'hybrid' is the previous path, retained for comparison.")
    knobs.add_argument("--dense-mode", choices=["auto", "local", "off"], default=None,
                       help="Dense model policy. 'off' leaves BM25F/LSA/char retrieval active.")
    knobs.add_argument("--no-query-expansion", action="store_true", default=None,
                       help="Use literal corpus vocabulary only (diagnostic A/B mode).")
    knobs.add_argument("--t-dense", type=float, default=None, help="Dense softmax temperature T_d.")
    knobs.add_argument("--t-lsa", type=float, default=None, help="LSA softmax temperature T_l.")
    knobs.add_argument("--alpha", type=float, default=None, help="Legacy weighted-logit mixture weight.")
    knobs.add_argument("--tau", type=float, default=None, help="Legacy top retrieval-share threshold.")
    knobs.add_argument("--delta", type=float, default=None, help="Legacy retrieval-share margin threshold.")
    knobs.add_argument(
        "--field-expert",
        action="store_true",
        default=None,
        help="Enable the optional third (field-term) expert.",
    )

    parser.add_argument(
        "-v", "--verbose", action="store_true", help="Show ranking diagnostics."
    )
    parser.add_argument(
        "--explain-features", action="store_true",
        help="Print every ranking feature for the returned candidates.",
    )
    return parser


class ConfigFileError(ValueError):
    """The --config file could not be read as a configuration."""


def base_config(path: Path | None):
    """The config a run starts from, before any flag overrides.

    Separate from `main` so the failure modes are unit-testable without loading a
    corpus: every one of them must be reported before a single model is touched.
    """
    if path is None:
        return DEFAULT

    import yaml

    from .config import from_mapping

    try:
        text = Path(path).read_text()
    except OSError as error:
        raise ConfigFileError(f"cannot read --config {path}: {error}") from error
    try:
        raw = yaml.safe_load(text)
    except yaml.YAMLError as error:
        raise ConfigFileError(f"--config {path} is not valid YAML: {error}") from error
    if raw is not None and not isinstance(raw, dict):
        raise ConfigFileError(
            f"--config {path} must be a YAML mapping of settings, got "
            f"{type(raw).__name__}"
        )
    try:
        return from_mapping(raw or {})
    except ValueError as error:
        raise ConfigFileError(f"--config {path}: {error}") from error


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    try:
        base = base_config(args.config)
    except ConfigFileError as error:
        print(f"error: {error}", file=sys.stderr)
        return 2

    # Flags override the file, and only where actually supplied: every optional
    # flag defaults to None and `with_overrides` drops Nones, so an unpassed flag
    # cannot silently reset a value the config file set.
    cfg = base.with_overrides(
        t_dense=args.t_dense,
        t_lsa=args.t_lsa,
        alpha=args.alpha,
        tau=args.tau,
        delta=args.delta,
        top_k=args.top_k,
        use_field_expert=args.field_expert,
        # `generators` is the default; the flag exists to select the deprecated
        # hybrid/legacy runtimes for ablation.
        retrieval_mode=args.retrieval_mode,
        dense_mode=args.dense_mode,
        use_query_expansion=(False if args.no_query_expansion else None),
        data_path=args.data,
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
    if args.explain_features:
        for rank, candidate in enumerate(result.candidates, 1):
            print(f"\nFeatures for candidate {rank} ({candidate.row['title']}):")
            print(json.dumps(candidate.features.values if candidate.features else {}, indent=2, sort_keys=True))
    if args.verbose:
        print(f"   query answered in {elapsed * 1000:.0f} ms")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
