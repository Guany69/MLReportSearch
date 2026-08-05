"""`reportfinder-train` -- fit, evaluate, and approve the learned components.

Three subcommands, in the order they must be run:

    reportfinder-train fusion    # fit + evaluate against the RRF fallback
    reportfinder-train decision  # fit + calibrate + evaluate against the policy
    reportfinder-train approve   # stamp an artifact servable, if it earned it

Training never produces a servable artifact by itself. `approve` is separate and
checks three things: that the evaluation report describes *this* model, that it
was earned on a split the model was not fitted on, and that the model actually
beat the fallback it would replace. A model that ties its fallback is refused --
the fallback is simpler, already serving, and has no artifact to keep in sync.

Approval is stamped `synthetic_v2_evaluation` and leaves `human_validated: false`.
It licenses serving; it does not license a claim about production accuracy.
"""

from __future__ import annotations

import argparse


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="reportfinder-train")
    sub = parser.add_subparsers(dest="kind", required=True)
    sub.add_parser("fusion", add_help=False,
                   help="Listwise fusion MLP over serving-time features.")
    sub.add_parser("decision", add_help=False,
                   help="Three-class decision head with temperature scaling.")
    sub.add_parser("approve", add_help=False,
                   help="Mark an artifact servable, if it beat its fallback.")
    known, rest = parser.parse_known_args(argv)

    if known.kind == "fusion":
        from ..training.train_fusion import main as run
    elif known.kind == "decision":
        from ..training.train_decision import main as run
    else:
        from ..training.approve import main as run
    return run(rest)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
