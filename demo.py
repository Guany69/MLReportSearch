"""Acceptance demo: run example queries and print ranking + answerability.

    uv run python demo.py
    uv run python demo.py --data tests/fixtures/fixture_reports.xlsx   # without the real workbook

This is synthetic data, so exact hits depend on what the corpus happens to contain.
The answerability fallback is explicitly uncalibrated; this is not an accuracy test.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

from reportfinder.config import DEFAULT
from reportfinder.ingest.models import IngestMode
from reportfinder.model import ReportFinder, explain_fields, why_matched
from reportfinder.represent import load_or_build

QUERIES = [
    "employees who transferred between departments",
    "monthly turnover by organization",
    "current active headcount by company",
    "new hires this quarter with start dates",
    "show me terminated workers by supervisory organization",
    "gross to net payroll results by pay group",
]


def _demo_generators(finder, cfg) -> int:
    """The demo on the structured contract.

    Prints the three-way decision, the selected instance within each family, and
    the active fallbacks -- the things `query()` cannot express.
    """
    from reportfinder.auth import DEVELOPMENT_PRINCIPAL, SearchRequest

    for query in QUERIES:
        t0 = time.perf_counter()
        outcome = finder.search(SearchRequest(query, DEVELOPMENT_PRINCIPAL, top_k=5))
        elapsed = (time.perf_counter() - t0) * 1000

        print(f'\nQ: "{query}"')
        print(f"   {outcome.decision.value}")
        if outcome.clarification is not None:
            print(f"   ? {outcome.clarification.question}")
        for family in outcome.families[:3]:
            selected = family.selected
            score = (
                f"{selected.cross_encoder_score:+.2f}"
                if selected.cross_encoder_score is not None else "n/a"
            )
            print(f"   -> {family.canonical_title}")
            print(f"      instance {selected.instance_id} | reranker {score} "
                  f"| admitted via {selected.admitted_via}"
                  + (" | via family expansion" if selected.expanded else ""))
        if not outcome.families:
            print("   (no reports returned)")
        print(f"      [{elapsed:.0f} ms]")

    print("\n" + "=" * 78)
    fallbacks = sorted(set(outcome.active_fallbacks))
    if fallbacks:
        print("Active fallbacks (nothing here is trained or calibrated):")
        for name in fallbacks:
            print(f"  - {name}")
    print("\nNote: these are report DEFINITIONS, not report result-sets. The system\n"
          "identifies which report to run; it never fabricates report data rows.")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the acceptance demo queries.")
    parser.add_argument("--data", type=Path, default=None, help="Workbook to use.")
    parser.add_argument(
        "--mode",
        choices=[m.value for m in IngestMode],
        default=None,
        help="Ingestion mode (default: legacy_single_file).",
    )
    parser.add_argument("--catalog", type=Path, default=None)
    parser.add_argument("--field-dictionary", type=Path, default=None)
    parser.add_argument("--rebuild", action="store_true")
    args = parser.parse_args(argv)

    cfg = DEFAULT.with_overrides(
        data_path=args.data,
        ingest_mode=args.mode,
        catalog_path=args.catalog,
        field_dictionary_path=args.field_dictionary,
    )

    try:
        rep = load_or_build(cfg, rebuild=args.rebuild, verbose=True)
    except FileNotFoundError as exc:
        print(f"error: {exc}", file=sys.stderr)
        print(
            "\nTo run the demo without the real workbook, generate the test fixture:\n"
            "  uv run python tests/make_fixture.py\n"
            "  uv run python demo.py --data tests/fixtures/fixture_reports.xlsx",
            file=sys.stderr,
        )
        return 1

    finder = ReportFinder(rep, cfg)
    print(f"\nMode: {cfg.ingest_mode} | retrieval: {cfg.retrieval_mode}")
    print(f"Corpus: {len(rep)} report families (from {rep.raw_row_count} rows)")
    if cfg.retrieval_mode == "generators":
        print(f"Model: cross-encoder={cfg.rerank.checkpoint} | "
              f"shortlist={cfg.shortlist.standard_rerank_depth}/"
              f"{cfg.shortlist.high_risk_rerank_depth}")
    else:
        print(f"Model: dense={rep.dense_model_name} | alpha={cfg.alpha} "
              f"T_d={cfg.t_dense} T_l={cfg.t_lsa} | tau={cfg.tau} delta={cfg.delta}")
    print("=" * 78)

    if cfg.retrieval_mode == "generators":
        return _demo_generators(finder, cfg)

    for query in QUERIES:
        t0 = time.perf_counter()
        result = finder.query(query)
        elapsed = (time.perf_counter() - t0) * 1000
        top = result.candidates[0]
        row = top.row

        print(f'\nQ: "{query}"')
        print(f"   -> {row['title']}")
        print(f"      {top.confidence_pct:.1f}% retrieval share | {result.status.upper()} "
              f"| margin {result.margin:.1%} | H(P)={result.entropy_bits:.2f} bits "
              f"(norm {result.normalized_entropy:.2f})")
        print(f"      {row['category']} | {row['data_source']} | {row['report_type']}")
        print(f"      why: {why_matched(top)}")
        for line in explain_fields(top):
            print(f"           {line}")
        if not result.confident:
            others = [c.row["title"] for c in result.candidates[1:]]
            print(f"      ambiguous - also considered: {'; '.join(others[:4])}")
        print(f"      [{elapsed:.0f} ms]")

    print("\n" + "=" * 78)
    print("Note: these are report DEFINITIONS, not report result-sets. The system\n"
          "identifies which report to run; it never fabricates report data rows.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
