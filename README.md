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

> **Two estates, and they are not comparable.** `data/Reports.xlsx` (4,000 rows →
> 3,280 families) is the legacy single-file estate: every measured number in
> `docs/` — recall, survival, the threshold sweep, both learned artifacts — was
> produced on it, with `configs/legacy_generators.yaml`. It is gitignored, so a
> fresh clone skips the tests that need it.
>
> The Phase 2 dual-file estate — `data/Phase2_Report_Catalog_No_Fields.xlsx` +
> `data/Phase2_Field_Dictionary.xlsx`, 4,368 rows → 4,299 families — **is** in
> this tree and is now runnable via `configs/phase2_generators.yaml`. It has no
> relevance labels and no evaluation of its own, so nothing about its quality is
> claimed here and its ingest counts must not be read as accuracy. See
> [Phase 2](#phase-2-dual-file-estate).

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

# Lint, types, tests -- what CI runs. Integration tests (real checkpoints) are
# deselected by default; run them deliberately with `-m integration`.
uv run ruff check src tests scripts app.py demo.py calibrate.py
uv run mypy
uv run pytest tests/ -q -ra
uv run pytest -m integration            # opt-in; loads real model checkpoints

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

The default permissive ambiguity policy attaches undecidable fields to all title candidates for recall while flagging every link. `strict` withholds them.

Catalog description, area-where-used, landing/worklet/chart, ownership, creator/date, usage, and last-run metadata are preserved. Authorized usage is retained but not scored because it has no discriminating variation in the supplied dictionary.

### Phase 2 dual-file estate

`Where_Used` is **not** a list of report titles. It is a list of typed Workday
references, and only one of the 60 observed types names a row in a report catalog:

```
Custom Report - Headcount by Supervisory Organization    ← a report
Calculated Field - CF_TF_CSO                             ← not a report
Condition Rule - Location is US Region 4                 ← not a report
```

Comparing those raw against catalog titles matched **0 of 20,951** distinct
entries, because the catalog stores `Headcount by Supervisory Organization` and
the dictionary stores `Custom Report - Headcount by Supervisory Organization`.
Resolution order:

1. **the raw value first**, exact then normalized — a report may legitimately be
   titled `Position Management Report - Calculations`, and stripping before
   matching would destroy such a title;
2. on a miss, split at the first `Type - ` whose left side is one of the
   enumerated types;
3. `Custom Report` → strip and re-resolve through the same exact / normalized /
   composite / ambiguity / opt-in-fuzzy path;
4. any other recognized type → **skipped**, counted in
   `non_report_where_used_skipped`, *not* recorded as unmatched;
5. an unrecognized prefix → treated as untyped, and an untyped miss stays in
   `unmatched_where_used`, which is the actionable channel. A new Workday type
   surfaces there for a human to add rather than silently deleting links.

Import counts on the supplied workbooks, reproduced by
`uv run reportfinder-data phase2-ingest` into
`artifacts/phase2_ingest_validation.json`:

| | |
|---|---|
| valid reports | 4,368 |
| families | 4,299 |
| report→field links | 75,689 |
| ambiguous links | 1,023 |
| typed non-report references skipped | 37,272 |
| unmatched report references | 274 |
| reports with zero linked fields | 96 |

**These are import counts, not quality metrics.** They say what was read and
linked; nothing here measures whether a linked field is the *right* field. No
relevance labels exist for this estate, so none of its numbers may be compared
with the legacy estate's — different corpus, different field reconstruction.

Commands:

```bash
uv run reportfinder-bundle build  --config configs/phase2_generators.yaml -v
uv run reportfinder-bundle verify --config configs/phase2_generators.yaml
uv run python -m reportfinder --config configs/phase2_generators.yaml "<query>"
uv run reportfinder-data phase2-ingest --config configs/phase2_generators.yaml
```

`--config` is how the CLI runs the configuration its bundle was built with; flags
still override individual values. The API server reads `REPORTFINDER_CONFIG`
instead:

```bash
REPORTFINDER_CONFIG=configs/phase2_generators.yaml \
  uv run uvicorn reportfinder.api.service:app --port 8080
```

### Build-time versus runtime configuration

A bundle's id keys on corpus content plus *index* configuration only, so tuning a
shortlist depth or a risk threshold does not invalidate 4,000 encoded documents.
The cost is that one bundle id can be served under materially different retrieval
behaviour. Both hashes are therefore carried and compared:
`build_config_hash` (recorded in the manifest at build) against
`runtime_config_hash` (computed from the config actually serving). A mismatch
raises a **warning, not an error** — the stored vectors stay valid, but
reproducing a recorded result needs the build-time configuration, not just the
bundle id. Disclosed in search warnings, search telemetry, `/v1/model-info`, and
every feedback record. Realign with `reportfinder-bundle build`, which rewrites
the manifest and reuses unchanged components.

### Which tests need which optional artifacts

Everything else runs offline with no downloads.

| Needs | Guard | Without it |
|---|---|---|
| `data/Reports.xlsx` | `requires_real_estate` | those tests skip; gitignored, so absent in CI |
| `artifacts/onnx/cross_encoder.onnx` | `requires_export` | ONNX parity tests skip, naming the export command |
| real model checkpoints | `-m integration` | deselected by default via `addopts` |
| `artifacts/feature_cache/*-validation-*` | discovered at import | decision-replay tests skip, distinguishing "absent" from "stale key" |
| `data/relevance*/splits/` | file check | split-guard tests skip |

## Important limits

- No human relevance or answerability labels exist. LTR and calibration cannot honestly be called production-trained.
- 6.72% of supplied BGE documents exceed 512 tokens (p95 525, max 895). BM25F and field indexes preserve tail zones; separate dense zone vectors remain a benchmark decision.
- Cross-encoder support is shipped disabled. Enable only after measuring a real gain and acceptable latency.
- Synthetic-data success does not establish production quality, fairness, or operational safety.

See [architecture decision](docs/architecture_decision.md), [research decisions](docs/model_improvement_research.md), and [migration notes](docs/migration.md).
The implementation details are documented in [query understanding](docs/query_understanding.md), [architecture](docs/architecture.md), [evaluation](docs/evaluation.md), and [known limitations](docs/known_limitations.md).
The concrete human-labeling, shadow-deployment, and acceptance program is in [production accuracy](docs/production_accuracy.md).
Runtime retrieval now addresses individual catalog rows (`R####`); near-identical families are grouped only when rendered. Training, sealed-split policy, artifact promotion, smoke testing, and rollback are documented in [docs/training.md](docs/training.md); data contracts and label precedence are in [docs/relevance_dataset.md](docs/relevance_dataset.md).
