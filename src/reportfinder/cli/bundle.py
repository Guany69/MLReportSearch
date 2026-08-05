"""`reportfinder-bundle` -- build, inspect and verify retrieval indexes.

Index construction is its own command rather than something that happens on first
query. Encoding 4000 documents with SPLADE takes roughly 100 seconds; making a
Streamlit start or a pytest run pay that implicitly would be a bad trade, and it
would hide when models are actually being downloaded.
"""

from __future__ import annotations

import argparse
import json
import sys

from ..config import from_mapping
from ..corpus import build_corpus_model
from ..index.build import ALL_COMPONENTS, build_bundle, bundle_id, load_bundle
from ..ingest import build_corpus
from ._common import load_yaml


def _corpus_model(cfg):
    if cfg.corpus_granularity != "report_row":
        raise SystemExit(
            "bundle build requires corpus_granularity='report_row'; "
            f"config says {cfg.corpus_granularity!r}"
        )
    corpus, _ = build_corpus(cfg, verbose=False)
    source = (
        cfg.data_path if cfg.ingest_mode == "legacy_single_file" else cfg.catalog_path
    )
    return build_corpus_model(
        corpus.frame, ingest_mode=cfg.ingest_mode, source_file=str(source)
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="reportfinder-bundle")
    sub = parser.add_subparsers(dest="command", required=True)

    build = sub.add_parser("build", help="Build or refresh index components.")
    build.add_argument("--config", required=True)
    build.add_argument(
        "--component", action="append", choices=list(ALL_COMPONENTS), default=None,
        help="Build only this component; repeat for several. Default: all.",
    )
    build.add_argument("-v", "--verbose", action="store_true")

    inspect = sub.add_parser("inspect", help="Print the bundle manifest.")
    inspect.add_argument("--config", required=True)

    verify = sub.add_parser("verify", help="Fail unless the bundle is ready to serve.")
    verify.add_argument("--config", required=True)

    args = parser.parse_args(argv)
    cfg = from_mapping(load_yaml(args.config))
    corpus = _corpus_model(cfg)

    if args.command == "build":
        if args.verbose:
            print(
                f"corpus: {len(corpus)} instances, {len(corpus.families)} families "
                f"(catalog {corpus.catalog_version}, content {corpus.content_hash})"
            )
            print(f"bundle: {bundle_id(corpus, cfg)}")
        manifest = build_bundle(cfg, corpus, only=args.component, verbose=args.verbose)
        readiness = manifest.readiness()
        print(json.dumps(readiness, indent=2))
        if readiness["degraded"]:
            # Degraded is a legitimate serving state, but never a silent one.
            print(
                "\nDegraded components are using declared fallbacks:\n  "
                + "\n  ".join(manifest.active_fallbacks()),
                file=sys.stderr,
            )
        return 0 if readiness["ready"] else 1

    bundle = load_bundle(cfg, corpus)
    if args.command == "inspect":
        print(json.dumps(bundle.manifest.to_dict(), indent=2, sort_keys=True))
        return 0

    readiness = bundle.manifest.readiness()
    if not readiness["ready"]:
        print(json.dumps(readiness, indent=2), file=sys.stderr)
        print(
            "\nFAIL: required components are missing or stale.", file=sys.stderr
        )
        return 1
    print(
        f"PASS bundle={bundle.manifest.bundle_version} "
        f"instances={bundle.manifest.instance_count} "
        f"families={bundle.manifest.family_count} "
        f"degraded={readiness['degraded'] or 'none'}"
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
