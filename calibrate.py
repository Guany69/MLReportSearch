"""Legacy title-leak smoke test. This does not calibrate answerability.

The decision rule (`p1 >= τ and margin >= δ`) is only as good as the thresholds,
and the right thresholds depend on the shape of the corpus -- specifically on how
many semantically near-identical reports it contains. A corpus of 4000 genuinely
distinct reports concentrates posterior mass; one with deep clusters of clones
spreads it. So τ cannot be chosen in the abstract.

The smoke test uses each report's own title as a query. The answer is assumed by
construction (that report), so we get an accuracy signal with no hand labels --
the supervision comes from the corpus's own structure.

    hit@1 -- how often the title's own report wins
    p1    -- the confidence distribution when it does
    τ sweep -- precision/coverage trade-off at each threshold

CAVEAT: titles are part of doc(r) (weighted x3), so this is severely leaky and
optimistic. It is not calibration, production accuracy, or an answerability
estimate. Use ``python -m reportfinder.evaluation.benchmark`` for the labeled
weak/synthetic harness and collect human judgments before fitting a calibrator.

    uv run python calibrate.py --data tests/fixtures/fixture_reports.xlsx --n 300
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

from reportfinder.config import DEFAULT
from reportfinder.model import ReportFinder
from reportfinder.represent import load_or_build


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--data", type=Path, default=None)
    parser.add_argument("--n", type=int, default=300, help="How many probe queries to sample.")
    parser.add_argument("--alpha", type=float, default=None)
    parser.add_argument("--t-dense", type=float, default=None)
    parser.add_argument("--t-lsa", type=float, default=None)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)

    cfg = DEFAULT.with_overrides(
        retrieval_mode="legacy_weighted_logit",
        data_path=args.data,
        alpha=args.alpha,
        t_dense=args.t_dense,
        t_lsa=args.t_lsa,
    )
    try:
        rep = load_or_build(cfg, verbose=True)
    except FileNotFoundError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    finder = ReportFinder(rep, cfg)
    rng = np.random.default_rng(args.seed)
    n = min(args.n, len(rep))
    probe_idx = rng.choice(len(rep), size=n, replace=False)

    print(f"\nProbing {n} of {len(rep)} report families "
          f"(alpha={cfg.alpha} T_d={cfg.t_dense} T_l={cfg.t_lsa})")
    print("Probe is LEAKY (title is in doc(r) x3) -> these are UPPER bounds.\n")

    p1s, hits, margins, entropies = [], [], [], []
    for i in probe_idx:
        title = rep.frame["title"].iloc[int(i)]
        # top_k=1 is enough: we only need the argmax and the posterior shape.
        result = finder.query(title, top_k=1)
        p1s.append(result.p1)
        margins.append(result.margin)
        entropies.append(result.normalized_entropy)
        hits.append(result.candidates[0].index == int(i))

    p1s = np.array(p1s)
    hits = np.array(hits)
    margins = np.array(margins)

    print(f"hit@1 (own title retrieves own report): {hits.mean():.1%}")
    print(f"p1     mean {p1s.mean():.3f} | median {np.median(p1s):.3f} "
          f"| p10 {np.quantile(p1s, 0.1):.3f} | p90 {np.quantile(p1s, 0.9):.3f}")
    print(f"margin mean {margins.mean():.3f} | median {np.median(margins):.3f}")
    print(f"normalized entropy mean {np.mean(entropies):.3f}")

    print(f"\ntau sweep (delta fixed at {cfg.delta:.2f}):")
    print(f"{'tau':>6} {'coverage':>10} {'precision':>11} {'note':>26}")
    print("-" * 56)
    for tau in [0.05, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50]:
        fires = (p1s >= tau) & (margins >= cfg.delta)
        coverage = fires.mean()
        precision = hits[fires].mean() if fires.any() else float("nan")
        note = ""
        if tau == cfg.tau:
            note = "<- current default"
        if coverage == 0:
            note = "never fires"
        print(f"{tau:>6.2f} {coverage:>9.1%} {precision:>10.1%} {note:>26}")

    print(
        "\nRead it like this: coverage = share of queries answered with ONE confident\n"
        "report; precision = of those, how often it was right. Pick the largest tau\n"
        "whose coverage you can live with. Since the probe is optimistic, a real-world\n"
        "tau should sit below the value that looks ideal here."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
