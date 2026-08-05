# Evaluation methodology

Evaluation sources are never combined:

- `dev_queries.jsonl`: author-constructed synthetic development cases;
- `heldout_queries.jsonl`: frozen author-constructed synthetic wording;
- `transfer_queries.jsonl`: a further author-constructed transfer slice;
- the 10,000-scenario bundle: weak generated labels;
- human-reviewed: empty;
- production: empty.

Qrels can contain explicit report judgments or runtime predicates (`required_fields`, `title_terms_any`, `title_terms_all`, `categories`, `excluded_fields`). Explicit judgments override predicate grades. This avoids brittle single-report answers when many current catalog rows satisfy a request.

The benchmark reports Hit@1/5/10, MRR@10, nDCG, no-answer precision/recall, a status confusion matrix, status/ambiguity accuracy, linearly interpolated latency p95, slice metrics, cumulative dataset-version metrics, and complete per-query diagnostics. Answerability calibration-style statistics are retained only as engineering diagnostics; the runtime score is explicitly uncalibrated.

Failures are classified as expansion miss/drift, typo miss/drift, acronym collision, intent miss, incorrect mandatory field, retrieval miss, fusion failure, ranking inversion, false/failed abstention, ambiguity failure, corpus gap, defective label, or environment failure. Held-out phrases are not copied into rules after inspection; a general defect receives a general fix and a separate regression test.

```bash
uv run python -m reportfinder.evaluation.benchmark --config configs/legacy_generators.yaml \
  --qrels evaluation/dev_queries.jsonl --output artifacts/development_results.json
uv run python -m reportfinder.evaluation.benchmark --config configs/legacy_generators.yaml \
  --qrels evaluation/heldout_queries.jsonl --output artifacts/heldout_results.json
uv run python -m reportfinder.evaluation.benchmark --config configs/legacy_generators.yaml \
  --qrels evaluation/transfer_queries.jsonl --output artifacts/transfer_results.json
# Reproducible ablations; each run gets its own artifact.
uv run python -m reportfinder.evaluation.benchmark --config configs/legacy_generators.yaml \
  --qrels evaluation/dev_queries.jsonl --set use_query_expansion=false \
  --output artifacts/ablation_expansion_off.json
uv run python -m reportfinder.evaluation.benchmark --config configs/legacy_generators.yaml \
  --qrels evaluation/dev_queries.jsonl --set ranking_preset=legacy \
  --output artifacts/ablation_legacy_ranking.json
```

The debug loop stops when observed failures are corpus gaps, label defects, or accepted limitations rather than clear implementation defects. No human judgments currently exist, so no supervised ranker is trained and no production accuracy is claimed.

Held-out v2 appends salary-range, requisition, new-hire, and in-domain-unsatisfiable coverage. The original 15 v1 lines remain byte-identical; every appended line carries `version: 2` and a coverage-gap justification. Reports include v1-only and cumulative v2 metrics.
