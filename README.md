# reportfinder

Local search over Workday-style report definitions, built so that a report the
user needs cannot be discarded before anything has read their request properly.

Two architectures are present:

- **`generators`** (current) — several independent candidate generators (BM25F,
  SPLADE, multi-view dense, family query-prototypes), a source-preserving union
  that records *which* generator found what, a quota-based shortlist that protects
  candidates only one source found, and a PyTorch cross-encoder that scores the
  whole shortlist against the user's exact words. See
  [docs/generator_architecture.md](docs/generator_architecture.md).
- **`hybrid`** (previous) — **deprecated**, retained only for comparison and
  ablation. Select it explicitly with `--retrieval-mode hybrid`; nothing chooses
  it by default any more.

The included estate and relevance labels are synthetic/weak. Benchmark numbers are
development signals, not production accuracy. **No trained fusion or decision model
ships**: serving uses an explicit RRF fallback and a deterministic three-way policy,
and every active fallback is reported in the response. Nothing here is calibrated,
and no score is presented as a probability. Read
[docs/known_limitations.md](docs/known_limitations.md) before quoting any number.

## Setup and search

> **`data/Reports.xlsx` is the estate.** The Phase 2 catalog + field-dictionary
> workbooks are not in this tree, so the dual-file *ingest path* — which is real
> code and still tested against synthetic fixtures — has no real data to read.
> Every command, config and test here runs against `Reports.xlsx`. The workbook
> itself is gitignored, so a fresh clone skips the tests that need it.

```bash
uv sync --dev

# Build the retrieval indexes once (~4.5 min: SPLADE encoding dominates).
uv run reportfinder-bundle build --config configs/legacy_generators.yaml -v
uv run reportfinder-bundle verify --config configs/legacy_generators.yaml

# Search. `generators` is the default; no flag needed.
uv run python -m reportfinder "why are we losing people faster than we can backfill"

# The HTTP service (needs the api extra).
uv sync --extra api
uv run uvicorn reportfinder.api.service:app --port 8080
curl -XPOST localhost:8080/v1/search -H 'content-type: application/json' \
     -H 'x-principal-id: alice' -d '{"query":"headcount by organization"}'

# Lint, types, tests -- what CI runs.
uv run ruff check src tests scripts app.py demo.py calibrate.py
uv run mypy
uv run pytest tests/ -q -ra

# Inspect every ranking feature, interpretation, and correction
uv run python -m reportfinder --explain-features \
  "take-home pay by paycheck period"

# Lexical/LSA fallback remains fully operational without a dense model
uv run python -m reportfinder --dense-mode off "voluntary turnover by boss"

# UI and demo
uv run streamlit run app.py
uv run python demo.py
```

The first run downloads BGE, builds dense/LSA representations, and writes `.cache/`. Later runs load locally. `--rebuild` invalidates explicitly; input hashes, schema versions, model/dependency versions, and representation settings invalidate automatically.

The legacy single-file interface remains supported:

```bash
uv run python -m reportfinder --mode legacy_single_file --data data/Reports.xlsx "..."
```

`Config.retrieval_mode="legacy_weighted_logit"` reproduces the historical fusion. Its former “product of experts” description was incorrect: after renormalization it is exactly a softmax over a weighted sum of temperature-scaled logits. See [the audit](docs/current_model_audit.md).

## Architecture

The live path is documented in
[docs/generator_architecture.md](docs/generator_architecture.md), including measured
candidate survival at every pruning boundary. In summary:

```
authorize → prepare (Q0 raw … Q5) → independent generators → source-preserving union
→ recall risk → source-preserving shortlist → cross-encoder over the full shortlist
→ fusion (RRF fallback) → family aggregation → three-way decision
```

No single retriever, score or rewrite decides what reaches the cross-encoder. A
report nominated by any generator enters the union and is eligible for the
shortlist; a generator that missed a candidate is recorded as a **mask**, never as
a zero score.

### The previous `hybrid` path

1. A token trie finds literal corpus vocabulary and declarative plain-language rules. Context, specificity suppression, conservative typo correction, acronym resolution, and span-local requirement tiering produce an auditable expansion result while preserving the original query.
2. Independent retrievers score weighted original, expanded, and subquery forms on a shared scale. Exact-field retrieval consumes structured concepts and returns bounded requested-field coverage.
3. Reciprocal-rank fusion handles incompatible score scales: `RRF(r)=Σ_j 1/(k+rank_j(r))`.
4. Structured ranking combines bounded `[0,1]` evidence features. Required-field satisfaction uses a smooth multiplicative gate; operational priors cannot override clear semantic relevance.
5. Answerability is separate from ranking: `ANSWERABLE`, `NEEDS_CLARIFICATION`, or `NO_SATISFACTORY_REPORT`. Expansion-derived grounding counts as support, partial required-field coverage asks for clarification, and out-of-domain language abstains. The answerability score is uncalibrated and never presented as a probability.

The `find()`/CLI/Streamlit interfaces remain compatible. `Candidate.probability` is retained for older callers; use `retrieval_share` in new code. Neither is confidence.

## Evaluation

```bash
uv run python -m reportfinder.evaluation.benchmark \
  --config configs/legacy_generators.yaml --qrels evaluation/qrels.jsonl \
  --output artifacts/generators_summary.json

# The pre-migration path on the same estate, for an honest A/B.
uv run python -m reportfinder.evaluation.benchmark \
  --config configs/legacy_hybrid_baseline.yaml --qrels evaluation/dev_queries.jsonl \
  --output artifacts/development_results.json

uv run python -m reportfinder.evaluation.benchmark \
  --config configs/legacy_generators.yaml --qrels evaluation/heldout_queries.jsonl \
  --output artifacts/heldout_results.json

uv run python scripts/audit_labels.py
uv run python scripts/capture_baseline.py

uv run pytest tests/ -q
```

The harness reports ranking, status confusion, ambiguity/no-answer decisions, slice/version metrics, latency, and per-query interpretations and reasons. Repeatable `--set key=value` overrides support expansion, typo, dense, cross-encoder, and legacy-ranking ablations. Development, frozen held-out v1/v2, transfer, and weak-label sources remain separate. Do not interpret author-synthetic or weak labels as user relevance judgments.

## Data and ingestion

Both ingestion modes yield the same family-level corpus. Phase 2 reconstructs report→field links from dictionary `Where_Used` values. Report titles are not unique; `report_id` is the catalog row index. Each link retains method, confidence, ambiguity, and provenance.

The default permissive ambiguity policy attaches undecidable fields to all title candidates for recall while flagging every link. `strict` withholds them. In the supplied synthetic estate: 4,000 catalog rows collapse to 3,882 families; 50,464 links include 1,936 ambiguous links and 533 duplicate title identities.

Catalog description, area-where-used, landing/worklet/chart, ownership, creator/date, usage, and last-run metadata are preserved. Authorized usage is retained but not scored because it has no discriminating variation in the supplied dictionary.

## Important limits

- No human relevance or answerability labels exist. LTR and calibration cannot honestly be called production-trained.
- 6.72% of supplied BGE documents exceed 512 tokens (p95 525, max 895). BM25F and field indexes preserve tail zones; separate dense zone vectors remain a benchmark decision.
- Cross-encoder support is shipped disabled. Enable only after measuring a real gain and acceptable latency.
- Synthetic-data success does not establish production quality, fairness, or operational safety.

See [architecture decision](docs/architecture_decision.md), [research decisions](docs/model_improvement_research.md), and [migration notes](docs/migration.md).
The implementation details are documented in [query understanding](docs/query_understanding.md), [architecture](docs/architecture.md), [evaluation](docs/evaluation.md), and [known limitations](docs/known_limitations.md).
The concrete human-labeling, shadow-deployment, and acceptance program is in [production accuracy](docs/production_accuracy.md).
Runtime retrieval now addresses individual catalog rows (`R####`); near-identical families are grouped only when rendered. Training, sealed-split policy, artifact promotion, smoke testing, and rollback are documented in [docs/training.md](docs/training.md); data contracts and label precedence are in [docs/relevance_dataset.md](docs/relevance_dataset.md).
