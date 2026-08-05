"""Training data assembly, and the leakage guards around it.

Every label available in this repository is synthetic. `data/relevance/` holds
10,874 seed judgements produced by a generator, and the two human-judgement files
are blank templates -- 0 of 290,876 cells filled. So anything trained here can
demonstrate that the pipeline runs end to end; it cannot demonstrate that the
pipeline is good. `label_provenance()` carries that fact into every artifact.

Three leakage guards, because each failure is invisible in the metrics:

* the sealed test split is refused for any development operation, via the existing
  `SplitGuard`;
* near-duplicate queries are clustered before splitting, so a paraphrase of a
  training query cannot appear in validation;
* fusion features are generated out of fold, so the model never sees features
  produced by a pipeline that had already been fitted on the same query.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ..relevance.splits import SplitGuard

# The relevance bundle's own vocabulary, kept verbatim so the join is auditable.
SCENARIO_FIELD = "Scenario ID"
REPORT_FIELD = "Report Key"
GRADE_FIELD = "Synthetic Relevance Grade"
TITLE_FIELD = "Report Title"


@dataclass(frozen=True)
class LabelledQuery:
    """One scenario with its graded reports."""

    scenario_id: str
    query: str
    grades: dict[str, int]

    @property
    def positives(self) -> set[str]:
        return {report for report, grade in self.grades.items() if grade > 0}


def label_provenance(root: Path) -> dict[str, object]:
    """What the labels actually are. Stamped into every trained artifact."""
    manifest = Path(root) / "dataset_manifest.json"
    payload = json.loads(manifest.read_text()) if manifest.exists() else {}
    return {
        # v2 roots declare their own provenance; v1 predates the field.
        "training_label_source": str(payload.get("label_source", "synthetic_seed")),
        "human_validated": bool(payload.get("human_validated", False)),
        "human_judgement_count": 0,
        # Set to True only after evaluation against real human judgements.
        "approved": False,
        "caveat": (
            "Trained on synthetic weak supervision. These labels were produced by a "
            "generator, not by Axis employees, so metrics computed against them "
            "measure agreement with the generator, not production accuracy."
        ),
    }


def _scenario_csv(root: Path) -> Path:
    """The scenarios file, v2 preferred.

    Two filenames rather than one so a v2 root and the original bundle can both
    be read by the same loaders -- which is what makes switching label sets a
    `--relevance-root` flag instead of a migration.
    """
    v2 = Path(root) / "raw" / "scenarios_v2.csv"
    if v2.exists():
        return v2
    return Path(root) / "raw" / "axis_report_search_scenarios_10000.csv"


def load_scenarios(root: Path) -> dict[str, str]:
    """Scenario id -> query text."""
    import pandas as pd

    frame = pd.read_csv(_scenario_csv(root))
    column = next(
        (c for c in frame.columns if c.strip().casefold() in
         {"query", "scenario", "request", "search text", "user request"}),
        frame.columns[1],
    )
    return {
        str(row[SCENARIO_FIELD]): str(row[column])
        for _, row in frame.iterrows()
        if str(row.get(SCENARIO_FIELD, "")).strip()
    }


def _load_splits(root: Path) -> dict[str, set[str]]:
    """Every split's query ids, for the leakage guard.

    Missing files yield an empty set rather than raising: an absent validation
    split must not stop the *test* split from being guarded.
    """
    splits: dict[str, set[str]] = {}
    for name in ("train", "validation", "calibration", "test"):
        path = root / "splits" / f"{name}_queries.txt"
        splits[name] = (
            {line.strip() for line in path.read_text().splitlines() if line.strip()}
            if path.exists() else set()
        )
    return splits


def load_labelled_queries(root: Path, split: str | None = None) -> list[LabelledQuery]:
    """Join scenarios to their graded judgements, optionally within one split."""
    root = Path(root)
    scenarios = load_scenarios(root)

    wanted: set[str] | None = None
    if split is not None:
        split_file = root / "splits" / f"{split}_queries.txt"
        wanted = {
            line.strip() for line in split_file.read_text().splitlines() if line.strip()
        }
        if split == "test":
            # Loading the sealed split runs the overlap check, and only that.
            #
            # `assert_allowed` blocks the operations in `DEVELOPMENT_OPERATIONS`;
            # reading the test split for a final evaluation is not one of them and
            # must not be blocked, so calling it here with any operation name would
            # be theatre. (It used to be called with `"final_evaluation"`, which is
            # not a known operation and so could never fire -- a guard in
            # appearance only.) What is worth checking at load time is that the
            # splits are actually disjoint, which the constructor does.
            #
            # Development operations are guarded where they happen: the trainers
            # call `assert_allowed` with their own operation names.
            SplitGuard(_load_splits(root))

    grades: dict[str, dict[str, int]] = {}
    judgements = root / "raw" / "axis_report_search_seed_judgements.jsonl"
    if not judgements.exists():
        judgements = root / "raw" / "axis_report_search_seed_judgments.jsonl"
    for line in judgements.read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        scenario = str(row[SCENARIO_FIELD])
        if wanted is not None and scenario not in wanted:
            continue
        grade = row.get(GRADE_FIELD)
        if grade in (None, ""):
            continue
        grades.setdefault(scenario, {})[str(row[REPORT_FIELD])] = int(grade)

    return [
        LabelledQuery(scenario_id=scenario, query=scenarios.get(scenario, ""), grades=g)
        for scenario, g in sorted(grades.items())
    ]


def cluster_near_duplicates(queries: list[str], threshold: float = 0.95) -> list[int]:
    """Group near-identical query forms so they cannot straddle a split.

    Character-trigram cosine, matching `SplitGuard.assert_no_near_duplicates`. A
    template and its rewrite landing on opposite sides of a split inflates every
    metric while looking like clean generalisation.
    """
    from sklearn.feature_extraction.text import TfidfVectorizer

    if not queries:
        return []
    vectorizer = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 3))
    matrix = vectorizer.fit_transform(queries)
    similarity = (matrix @ matrix.T).toarray()

    cluster = [-1] * len(queries)
    next_id = 0
    for i in range(len(queries)):
        if cluster[i] != -1:
            continue
        cluster[i] = next_id
        for j in range(i + 1, len(queries)):
            if cluster[j] == -1 and similarity[i, j] >= threshold:
                cluster[j] = next_id
        next_id += 1
    return cluster


@dataclass
class ListwiseBatch:
    """One query's shortlist: features, graded targets, and a padding mask."""

    features: np.ndarray
    grades: np.ndarray
    mask: np.ndarray
    scenario_id: str
    instance_ids: list[str]


def build_listwise_batches(
    examples: list[tuple[str, list[str], np.ndarray, dict[str, int]]],
    *,
    max_candidates: int = 200,
) -> list[ListwiseBatch]:
    """Pad each query's candidates into a fixed-width listwise batch."""
    batches: list[ListwiseBatch] = []
    for scenario_id, instance_ids, features, grades in examples:
        count = min(len(instance_ids), max_candidates)
        width = features.shape[1] if features.size else 0
        padded = np.zeros((max_candidates, width), dtype=np.float32)
        target = np.zeros(max_candidates, dtype=np.float32)
        mask = np.zeros(max_candidates, dtype=bool)

        padded[:count] = features[:count]
        for position, instance_id in enumerate(instance_ids[:count]):
            # Graded, so several correct answers can share the target mass.
            target[position] = float(grades.get(instance_id, 0))
        mask[:count] = True

        batches.append(ListwiseBatch(
            features=padded, grades=target, mask=mask,
            scenario_id=scenario_id, instance_ids=list(instance_ids[:count]),
        ))
    return batches


def co_positive_mask(grades: dict[str, int], instance_ids: list[str],
                     family_of) -> np.ndarray:
    """Pairs that must not be used as negatives.

    Three kinds: other graded positives for the same query, other instances of a
    positive's family (the same report under a different row), and anything with no
    grade at all -- unjudged is not the same as irrelevant, and treating it as a
    negative teaches the model that correct-but-unlabelled answers are wrong.
    """
    positives = {i for i, g in grades.items() if g > 0}
    positive_families = {family_of(i) for i in positives}
    return np.array([
        (instance_id in positives)
        or (family_of(instance_id) in positive_families)
        or (instance_id not in grades)
        for instance_id in instance_ids
    ], dtype=bool)
