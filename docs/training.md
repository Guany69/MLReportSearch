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
version, the dataset version, **both** feature-schema hashes, and the
**serving-config hash**. A feature *rename* leaves the array width unchanged, so
only the hash catches it; without that key a stale cache would be the right shape
and the wrong numbers.

The serving-config component is there because the bundle version deliberately is
not enough. It keys on corpus content plus *index* configuration, so a shortlist
change does not re-encode 4,000 documents — but shortlist policy, rerank, fusion,
decision and risk settings all decide which candidates a pass records and in what
order. Without it, a pass generated under one shortlist policy is
indistinguishable from one generated under another, and a model would be trained
on features its own pipeline no longer produces.

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

`SplitGuard` raises on any overlap at construction — across all four splits,
pairwise. `calibration` used to be excluded from that check, which is the one
place it matters most: a temperature fitted on rows the model trained on is not a
calibration.

`assert_allowed` blocks the sealed split for the operations named in
`DEVELOPMENT_OPERATIONS`, and now **refuses an operation name it does not
recognise**. A misspelled name previously disabled the guard silently, which is
the same class of defect as the load-time call that passed `"final_evaluation"` —
a name outside the development set, and so a guard in appearance only.
`"final_evaluation"` is now declared in `NON_DEVELOPMENT_OPERATIONS` and still
passes, deliberately: the sealed split exists for it.

## Approval

`reportfinder-train approve` refuses unless all of these hold:

1. **The report describes this model.** Matched on a digest of the weights, not the
   file, because approval rewrites metadata and a file digest would change with it.
   The artifact's kind is resolved from the architecture constants exactly —
   `pytorch_mlp_64_32_1` and `pytorch_mlp_32_16_3` contain neither the word
   "fusion" nor "decision", so a substring test classified every artifact as
   fusion and refused every real decision approval.
2. **It was earned on `validation`.** Never the split the model was fitted on;
   never the sealed test split.
3. **The model beat the fallback it would replace.** A tie is refused: the fallback
   is simpler, already serving, and has no artifact to keep in sync.
4. **Nothing measured went backwards** — aggregate *or* per-slice. An aggregate
   averages a slice regression away, which is not hypothetical: the shipped fusion
   model gains +0.0031 nDCG overall while losing on `sibling_discriminating`, the
   one archetype the v2 labels were built to measure. `--allow-regressions` permits
   the trade explicitly and records that it was permitted.

### Metric directions

A regression check needs to know which way is better, and every metric used to be
compared as if larger were. That silently inverts the judgement on every error,
loss and latency measure — a model with twice the calibration error would have
read as an improvement.

Direction is resolved per metric: the evaluation report's own `metric_directions`
block first (values must be exactly `higher` or `lower` after normalisation), then
name patterns (`*_ms`, `*_loss`, `*_error` and the tokens `latency`, `ece`, `nll`,
`brier`, `loss`, `risk`, `error` are lower-is-better; `recall`, `precision`, `f1`,
`accuracy`, `auc`, `mrr`, `map`, `ndcg`, `hit`, `coverage`, `specificity`,
`agreement`, `throughput` are higher). A comparable numeric metric whose direction
cannot be resolved **refuses the approval** — including under
`--allow-regressions`, because that flag permits a known trade-off and a metric
nobody can state the sign of is not a known anything.

On success it stamps `approved: true`, `approval.basis: "synthetic_v2_evaluation"`,
the evaluation's digest, and what the gate actually judged: resolved directions,
the metrics compared, the slices compared, the kind, architecture, split and query
count. The loader refuses `approved: true` with no basis, so hand-editing the flag
no longer works. A refused approval leaves the artifact byte-identical.

**`human_validated` stays `false`.** Approval says a model beat its fallback on
synthetic labels. It licenses serving; it does not license a claim about production
accuracy, and it must be re-earned against human judgements before one is made.

## Rollback

Clear `fusion.artifact_path` / `decision.artifact_path` in the config. The
fallbacks — RRF and the deterministic three-way policy — are always present, are
what the models were measured against, and are disclosed in every response either
way. There is no state to unwind.
