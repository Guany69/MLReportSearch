"""Build the whole v2 label set, deterministically.

One RNG, seeded once and threaded through every archetype, so two runs with the
same seed produce byte-identical files. That is asserted, not assumed: the
manifest records a digest of each output and a test regenerates and compares.
Without it, "regenerate the labels" silently becomes "get different labels", and
every metric computed before and after becomes incomparable for a reason nobody
can see.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from ...config import DEFAULT, from_mapping
from ...corpus import build_corpus_model
from ...ingest import build_corpus
from ...query.expansion.rules import load_lexicon
from ...query.expansion.vocabulary import CorpusVocabulary
from . import archetypes, emit, surface, validate

GENERATOR_VERSION = "2.0.0"


def build_scenarios(corpus, lexicon, *, seed: int, budget: archetypes.Budget):
    """Every archetype, in a fixed order, from one seeded generator."""
    rng = np.random.default_rng(seed)
    ids = archetypes._Ids()

    phrasings = surface.lexicon_phrasings(lexicon)
    emissions = surface.rule_emissions(lexicon)
    vocabulary = surface.corpus_vocabulary_tokens(corpus)

    out = []
    out += archetypes.sibling_discriminating(
        corpus, rng, ids, budget.sibling_discriminating, phrasings)
    out += archetypes.vague_outcome(
        corpus, rng, ids, budget.vague_outcome, phrasings, emissions)
    out += archetypes.acronym(corpus, rng, ids, budget.acronym, phrasings)
    out += archetypes.misspelling(
        corpus, rng, ids, budget.misspelling, phrasings, vocabulary=vocabulary)
    out += archetypes.negation(corpus, rng, ids, budget.negation, phrasings)
    out += archetypes.short_query(corpus, rng, ids, budget.short_query)

    single, split = archetypes.multi_intent(
        corpus, rng, ids, budget.multi_intent_single, budget.multi_intent_split,
        phrasings,
    )
    out += single + split

    out += archetypes.no_answer_reserved(corpus, rng, ids, budget.no_answer_reserved)
    out += archetypes.no_answer_impossible_combo(
        corpus, rng, ids, budget.no_answer_impossible_combo, phrasings)
    out += archetypes.ambiguous_clarification(
        corpus, rng, ids, budget.ambiguous_clarification)
    return out


def generate(
    out_root: Path,
    *,
    config_path: Path | None = None,
    seed: int = 20260804,
    budget: archetypes.Budget | None = None,
    verbose: bool = False,
) -> dict:
    """Generate, validate, and write the v2 root. Returns its manifest."""
    import yaml

    cfg = DEFAULT
    if config_path:
        cfg = from_mapping(yaml.safe_load(Path(config_path).read_text()) or {})
    cfg = cfg.with_overrides(corpus_granularity="report_row")

    built, _ = build_corpus(cfg, verbose=False)
    corpus = build_corpus_model(
        built.frame, ingest_mode=cfg.ingest_mode, source_file=str(cfg.data_path),
    )
    lexicon = load_lexicon(CorpusVocabulary.from_frame(built.frame))
    budget = budget or archetypes.Budget()

    scenarios = build_scenarios(corpus, lexicon, seed=seed, budget=budget)
    if verbose:
        print(f"generated {len(scenarios)} scenarios")

    # Splitting uses its own generator so adding an archetype cannot silently
    # reshuffle the splits of every archetype before it.
    splits = emit.assign_splits(scenarios, np.random.default_rng(seed + 1))
    report = validate.validate(scenarios, corpus, lexicon, splits=splits)
    if verbose:
        for name, detail in sorted(report.checks.items()):
            print(f"  [{name}] {detail}")

    manifest = emit.write_root(
        out_root, scenarios, splits, seed=seed,
        generator_version=GENERATOR_VERSION, corpus=corpus,
        validation_report=report, budget=budget.as_dict(),
    )
    if verbose:
        print(f"-> {out_root}")
    return manifest


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="generate_labels_v2")
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--out", type=Path, default=Path("data/relevance_v2"))
    parser.add_argument("--seed", type=int, default=20260804)
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args(argv)

    manifest = generate(
        args.out, config_path=args.config, seed=args.seed, verbose=args.verbose,
    )
    print(
        f"PASS scenarios={manifest['scenario_count']} "
        f"judgements={manifest['judgement_count']} "
        f"splits={manifest['split_sizes']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
