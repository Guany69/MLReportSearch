# Training, evaluation, and approval

Three commands, in the order they must run. Training never produces a servable
artifact by itself; approval is a separate act with its own checks.

```bash
# 0. Labels. Generated from the catalog, deterministic, self-validating.
uv run python scripts/generate_labels_v2.py --out data/relevance_v2 --seed 20260804 -v

# 1. Fit + evaluate against the fallback each would replace.
uv run reportfinder-train fusion   --relevance-root data/relevance_v2 -v
uv run reportfinder-train decision --relevance-root data/relevance_v2 -v

# 2. Tune the two deterministic thresholds offline, on the calibration split.
uv run python scripts/sweep_decision_thresholds.py --relevance-root data/relevance_v2

# 3. Approve — only if the model beat its fallback on validation.
uv run reportfinder-train approve artifacts/models/fusion.pt \
    --evaluation artifacts/evals/fusion_eval.json
uv run reportfinder-train approve artifacts/models/decision.pt \
    --evaluation artifacts/evals/decision_eval.json

# 4. Point the config at the approved artifacts, then confirm.
uv run reportfinder-bundle verify --config configs/legacy_generators.yaml
uv run reportfinder-smoke --config configs/legacy_generators.yaml
```

## The shared feature pass

Both trainers consume the same thing: one real pipeline run per labelled query.
Fusion needs per-candidate features, the decision head needs per-query features,
and evaluating either against its fallback needs the fallback's actual output —
all three come out of that one search.

Passes are cached under `artifacts/feature_cache/<key>/`, keyed on the bundle
version, the dataset version, and **both** feature-schema hashes. A feature
*rename* leaves the array width unchanged, so only the hash catches it; without
that key a stale cache would be the right shape and the wrong numbers.

The pass refuses to run on a pipeline that already has a trained fuser or decision
head. Features describing a trained fuser would teach the next fuser to agree with
the last one — a feedback loop, not learning.

## What was wrong before

Four defects, all of which produced plausible-looking numbers:

| Defect | Consequence |
|---|---|
| Feature generation iterated `outcome.families`, which is empty on `NO_CONFIDENT_MATCH` | ~40% of queries contributed nothing, and the ones that vanished were the hard ones |
| The decision head read 14 columns from a precomputed parquet while stamping the 20-name *serving* feature hash | The loader's hash check passed and serving would have raised a shape error on the first request |
| No `SplitGuard` call anywhere in decision training; the calibration slice was a random cut of *all* scenarios | Sealed test data reached both fitting and calibration |
| Temperature was fitted and ECE measured on the same slice | The reported calibration described the fit, not the calibration |
| A `_PlanView` shim returned `{}` for stated facets | `query_specificity` was always 0 in training and real at serving — train/serve skew invisible in every metric |

Fixed: features come from `run_traced`, which keeps the pre-truncation family list;
the guard is called with operation names it actually knows; temperature is fitted
on `calibration` and every reported metric computed on `validation`.

## Splits

`train` fits, `calibration` sets the temperature, `validation` decides approval,
`test` is sealed. v2 emits all four; the v1 bundle had no calibration split, which
is why the head used to carve a random one out of everything.

`SplitGuard` raises on any overlap at construction, and `assert_allowed` blocks the
sealed split for the operations named in `DEVELOPMENT_OPERATIONS`. Note that a call
with an *unknown* operation name can never fire — the previous load-time call used
`"final_evaluation"`, which is not in that set, and was a guard in appearance only.

## Approval

`reportfinder-train approve` refuses unless all three hold:

1. **The report describes this model.** Matched on a digest of the weights, not the
   file, because approval rewrites metadata and a file digest would change with it.
2. **It was earned on `validation`.** Never the split the model was fitted on;
   never the sealed test split.
3. **The model beat the fallback it would replace.** A tie is refused: the fallback
   is simpler, already serving, and has no artifact to keep in sync.

On success it stamps `approved: true`, `approval.basis: "synthetic_v2_evaluation"`,
and the evaluation's digest. The loader refuses `approved: true` with no basis, so
hand-editing the flag no longer works.

**`human_validated` stays `false`.** Approval says a model beat its fallback on
synthetic labels. It licenses serving; it does not license a claim about production
accuracy, and it must be re-earned against human judgements before one is made.

## Rollback

Clear `fusion.artifact_path` / `decision.artifact_path` in the config. The
fallbacks — RRF and the deterministic three-way policy — are always present, are
what the models were measured against, and are disclosed in every response either
way. There is no state to unwind.
