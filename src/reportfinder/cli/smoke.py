"""`reportfinder-smoke` -- prove the serving path answers a query end to end.

This once hard-coded an ingest mode whose workbooks were absent, so it could not
run at all. It now takes a config so it smoke-tests whatever is actually
configured, and it verifies the generator architecture rather than only the
legacy one.

It fails on a silent degradation as well as on an error: a run that returns
candidates while a *required* component is missing is not a passing smoke test.
"""

from __future__ import annotations

import argparse

from ..auth import DEVELOPMENT_PRINCIPAL, SearchRequest
from ..config import DEFAULT, from_mapping
from ..ingest import build_corpus
from ..model import ReportFinder
from ..represent import load_or_build
from ._common import load_yaml

DEFAULT_QUERY = "worker status report"

# The fallbacks this build is known to ship with, each documented in
# docs/known_limitations.md. Listed by name rather than waved through by prefix so
# that a fallback nobody expected -- a required index gone stale, a generator that
# stopped constructing -- still fails the smoke test.
KNOWN_SHIPPED_FALLBACKS = frozenset({
    "fusion_model:rrf",
    "decision_model:deterministic_three_way_policy",
    "late_interaction:generator_not_constructed",
    "views.interface:generator_disabled",
    "authorization:allow_all_dev_default",
})


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="reportfinder-smoke")
    parser.add_argument("--config", default=None,
                        help="Config file; defaults to the built-in defaults.")
    parser.add_argument("--query", default=DEFAULT_QUERY)
    parser.add_argument("--allow-fallback", action="store_true",
                        help="Tolerate optional components running on fallbacks.")
    args = parser.parse_args(argv)

    cfg = from_mapping(load_yaml(args.config)) if args.config else DEFAULT
    cfg = cfg.with_overrides(corpus_granularity="report_row")

    if cfg.retrieval_mode == "generators":
        corpus, _ = build_corpus(cfg, verbose=False)
        finder = ReportFinder(load_or_build(cfg, verbose=False), cfg)
        outcome = finder.search(
            SearchRequest(args.query, DEVELOPMENT_PRINCIPAL, top_k=3)
        )
        if not outcome.families:
            raise SystemExit(
                f"smoke failure: no families returned "
                f"(decision={outcome.decision.value})"
            )
        unexpected = [
            f for f in outcome.active_fallbacks
            # `fusion:` / `decision:` are the per-request restatements of the
            # fusion_model / decision_model component fallbacks above.
            if f not in KNOWN_SHIPPED_FALLBACKS
            and not f.startswith(("fusion:", "decision:"))
        ]
        if unexpected and not args.allow_fallback:
            raise SystemExit(
                "smoke failure: unexpected fallbacks active: " + ", ".join(unexpected)
            )
        top = outcome.families[0]
        print(
            f"PASS mode={cfg.ingest_mode} decision={outcome.decision.value} "
            f"families={len(outcome.families)} "
            f"top_family={top.family_id!r} top_instance={top.selected.instance_id} "
            f"bundle={outcome.model_bundle_version}"
        )
        return 0

    finder = ReportFinder(load_or_build(cfg, verbose=False), cfg)
    result = finder.query(args.query, top_k=3)
    if not result.candidates:
        raise SystemExit("smoke failure: runtime returned no candidates")
    if (
        cfg.use_ltr and finder._ranker and finder._ranker.fallback_reason
        and not args.allow_fallback
    ):
        raise SystemExit(f"smoke failure: LTR fallback: {finder._ranker.fallback_reason}")
    print(
        f"PASS mode={cfg.ingest_mode} candidates={len(result.candidates)} "
        f"top={result.candidates[0].row.get('report_key')}"
    )
    return 0
