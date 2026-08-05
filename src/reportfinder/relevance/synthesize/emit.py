"""Splitting, and writing the v2 root.

Two decisions worth stating:

**Grouping before splitting.** The split key is the target family, unioned with
near-duplicate query clusters. A family's scenarios must not straddle a boundary
-- if the same report is the answer in train and in validation, validation is
measuring memorisation rather than generalisation. v1's splits were pre-baked and
this was never verified.

**The v1 filenames are kept.** `load_labelled_queries` resolves the judgement file
by name, so emitting `axis_report_search_seed_judgments.jsonl` lets every existing
consumer read v2 through a one-line change rather than a rewrite.
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

# Fractions chosen so calibration and validation each hold enough
# sibling-discriminating scenarios to measure instance selection, while train
# stays small enough that one feature-generation pass is affordable at the
# measured per-query pipeline cost.
DEFAULT_FRACTIONS = {
    "train": 0.55, "calibration": 0.15, "validation": 0.15, "test": 0.15,
}

SCENARIO_COLUMNS = [
    "Scenario ID", "Employee Search Query", "Scenario Family", "Edge Case Tags",
    "Difficulty", "Answerability", "Intent Count", "Archetype",
    "Target Family ID", "Target Instance ID", "Construction",
    "Generator Version", "Generator Seed",
]

ANSWERABILITY_TEXT = {2: "Answerable", 1: "Partial / needs clarification", 0: "No answer"}


def assign_splits(scenarios, rng, *, fractions=None) -> dict[str, list[str]]:
    """Partition by group, stratified by archetype.

    Groups are assigned whole. Stratifying within archetype keeps every split
    representative: a validation set that happened to contain no no-answer cases
    would report perfect abstention behaviour while measuring none of it.
    """
    fractions = fractions or DEFAULT_FRACTIONS
    names = list(fractions)

    groups: dict[str, list] = defaultdict(list)
    for scenario in scenarios:
        groups[scenario.group_key].append(scenario)

    # A group's archetype is its most common one, so stratification is over a
    # single label per indivisible unit.
    by_archetype: dict[str, list[str]] = defaultdict(list)
    for key, members in sorted(groups.items()):
        dominant = Counter(m.archetype for m in members).most_common(1)[0][0]
        by_archetype[dominant].append(key)

    assignment: dict[str, list[str]] = {name: [] for name in names}
    for _, keys in sorted(by_archetype.items()):
        order = list(keys)
        rng.shuffle(order)
        # Deal round-robin against the target proportions rather than slicing:
        # slicing a short archetype list can starve a split entirely.
        quota = {name: 0.0 for name in names}
        for key in order:
            target = min(names, key=lambda n: quota[n] / fractions[n])
            quota[target] += 1.0
            assignment[target].extend(s.scenario_id for s in groups[key])

    return {name: sorted(ids) for name, ids in assignment.items()}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    digest.update(path.read_bytes())
    return digest.hexdigest()


def write_root(
    root: Path,
    scenarios,
    splits: dict[str, list[str]],
    *,
    seed: int,
    generator_version: str,
    corpus,
    validation_report,
    budget: dict[str, int],
) -> dict[str, object]:
    """Write the whole v2 dataset and return its manifest."""
    import pandas as pd

    root = Path(root)
    (root / "raw").mkdir(parents=True, exist_ok=True)
    (root / "processed").mkdir(parents=True, exist_ok=True)
    (root / "splits").mkdir(parents=True, exist_ok=True)

    # -- scenarios --------------------------------------------------------
    rows = []
    for scenario in scenarios:
        rows.append({
            "Scenario ID": scenario.scenario_id,
            "Employee Search Query": scenario.query,
            "Scenario Family": scenario.archetype,
            "Edge Case Tags": "; ".join(scenario.edge_case_tags),
            "Difficulty": _difficulty(scenario.archetype),
            "Answerability": ANSWERABILITY_TEXT[scenario.answerability],
            "Intent Count": 2 if "multi_intent" in scenario.archetype else 1,
            "Archetype": scenario.archetype,
            "Target Family ID": scenario.target_family_id or "",
            "Target Instance ID": scenario.target_instance_id or "",
            "Construction": json.dumps(scenario.provenance, sort_keys=True),
            "Generator Version": generator_version,
            "Generator Seed": seed,
        })
    scenarios_path = root / "raw" / "scenarios_v2.csv"
    pd.DataFrame(rows, columns=SCENARIO_COLUMNS).to_csv(scenarios_path, index=False)

    # -- judgements -------------------------------------------------------
    judgements_path = root / "raw" / "axis_report_search_seed_judgments.jsonl"
    with judgements_path.open("w") as stream:
        for scenario in scenarios:
            for report_key in sorted(scenario.grades):
                stream.write(json.dumps({
                    "Scenario ID": scenario.scenario_id,
                    "Report Key": report_key,
                    "Report Title": corpus.instance(report_key).title,
                    "Synthetic Relevance Grade": scenario.grades[report_key],
                    "Synthetic Judgment Role": scenario.judgment_roles[report_key],
                    "Review Status": "Unreviewed synthetic v2",
                    "Human Relevance Grade": "",
                    "Archetype": scenario.archetype,
                }, sort_keys=True) + "\n")

    # -- answerability labels --------------------------------------------
    # Labels only, deliberately not features: v1 shipped precomputed features in a
    # different feature space from serving, and the decision head was trained on
    # them while stamped with the serving hash. Features now come from a real
    # pipeline pass.
    labels_path = root / "processed" / "answerability_labels.parquet"
    pd.DataFrame([
        {"scenario_id": s.scenario_id, "answerability_label": s.answerability,
         "archetype": s.archetype}
        for s in scenarios
    ]).to_parquet(labels_path, index=False)

    # -- splits -----------------------------------------------------------
    for name, ids in sorted(splits.items()):
        (root / "splits" / f"{name}_queries.txt").write_text("\n".join(ids) + "\n")

    # -- reports and manifest --------------------------------------------
    (root / "validation_report.json").write_text(
        json.dumps(validation_report.as_dict(), indent=2, sort_keys=True) + "\n"
    )

    manifest = {
        "version": generator_version,
        "generator_seed": seed,
        "label_source": "synthetic_v2_generative",
        "human_validated": False,
        "human_judgement_count": 0,
        "scenario_count": len(scenarios),
        "judgement_count": sum(len(s.grades) for s in scenarios),
        "archetype_counts": dict(sorted(Counter(s.archetype for s in scenarios).items())),
        "archetype_budget": dict(sorted(budget.items())),
        "answerability_counts": dict(sorted(
            Counter(s.answerability for s in scenarios).items()
        )),
        "split_sizes": {name: len(ids) for name, ids in sorted(splits.items())},
        "corpus": {
            "source_file": corpus.source_file,
            "catalog_version": corpus.catalog_version,
            "content_hash": corpus.content_hash,
            "instance_count": len(corpus),
            "family_count": len(corpus.families),
        },
        "files": {
            "raw/scenarios_v2.csv": _sha256(scenarios_path),
            "raw/axis_report_search_seed_judgments.jsonl": _sha256(judgements_path),
            "processed/answerability_labels.parquet": _sha256(labels_path),
        },
        "label_policy": (
            "Queries are generated FROM catalog metadata, so each grade is true by "
            "construction rather than by matching the query against report text."
        ),
        "important_limitations": [
            "Synthetic. No human judged any of these. Metrics computed against them "
            "measure agreement with this generator, not production relevance.",
            "Grades hold only under the generator's assumptions: that a normalized "
            "title identifies a report family, and that carrying a field means being "
            "able to answer about it.",
            "v2 removes v1's lexical-matching circularity -- v1 built labels by "
            "matching titles and fields, which is the operation BM25F performs, so it "
            "favoured lexical retrieval by construction. v2 still shares corpus "
            "vocabulary by necessity.",
            "Not comparable across label versions. A v2 number is not a better or "
            "worse v1 number; it is a different measurement.",
            "Any model approved against this set carries approval_basis "
            "'synthetic_v2_evaluation' and must be re-earned against human "
            "judgements before any production-accuracy claim.",
        ],
    }
    (root / "dataset_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    (root / "README.md").write_text(_readme(manifest))
    return manifest


def _difficulty(archetype: str) -> str:
    if archetype in {"vague_outcome", "no_answer_impossible_combo", "multi_intent_split"}:
        return "Hard"
    if archetype in {"sibling_discriminating", "ambiguous_clarification", "negation"}:
        return "Medium-Hard"
    return "Medium"


def _readme(manifest: dict) -> str:
    limitations = "\n".join(f"- {line}" for line in manifest["important_limitations"])
    counts = "\n".join(
        f"| {name} | {count} |"
        for name, count in manifest["archetype_counts"].items()
    )
    splits = "\n".join(
        f"| {name} | {count} |" for name, count in manifest["split_sizes"].items()
    )
    return f"""# Relevance labels v2 (synthetic, generative)

{manifest['scenario_count']} scenarios / {manifest['judgement_count']} judgements,
generated from `{manifest['corpus']['source_file']}`
({manifest['corpus']['instance_count']} instances,
{manifest['corpus']['family_count']} families) with seed
{manifest['generator_seed']}.

## Why this exists

v1's labels were produced by matching generated queries back against report text.
That is the same operation BM25F performs, so measuring lexical retrieval against
them was circular -- and it showed: BM25F alone reached 0.864 union recall against
the full union's 0.872.

Worse, **zero** v1 queries graded two instances of the same family. Family
expansion and best-instance selection therefore could not be measured at all.

v2 builds each query *from* a chosen target's metadata, so the grade is true by
construction. 450 scenarios exist specifically to distinguish siblings.

## Archetypes

| archetype | count |
|---|---|
{counts}

## Splits

| split | scenarios |
|---|---|
{splits}

Groups (target family, plus near-duplicate query clusters) are assigned whole, so
no family's scenarios straddle a split boundary.

## Limitations

{limitations}
"""
