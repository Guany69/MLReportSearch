# The source-preserving generator architecture

This describes the retrieval path that is live under `retrieval_mode: generators`,
why the previous one lost candidates, and what is and is not proven about the
replacement.

## The failure this replaced

The previous `hybrid` path was **not** lexically gated — that is worth stating
plainly, because it is the natural assumption. Dense retrieval already ran
independently over the whole corpus (`model.py:214`), and fusion was rank-only RRF,
so BM25 scores and cosines were never compared on a common scale.

Candidates were lost at five other places:

| Boundary | Code | What was lost |
|---|---|---|
| Shared post-fusion cut | `model.py:480` — `sorted(fused, ...)[:100]` | The hard recall ceiling. A report found by one retriever ranks low in a fused ordering *because* only one retriever voted for it, so the candidates most in need of protection were dropped first. |
| Per-channel truncation | `model.py:423,431` | The same `candidate_k` truncated every channel *before* fusion, so it could not be ablated independently. |
| Negative-score zeroing | `forms.py:50` | Dense cosine can be legitimately negative and still be the best available evidence. |
| Top-2 form aggregation | `forms.py:62` | Only the best two query forms survived. |
| Single-result collapse | `model.py:619` — `n_show = 1 if confident` | An `ANSWERABLE` verdict returned exactly one result; ranks 2..k were computed and discarded. |

The cross-encoder also never ran (`use_cross_encoder=False`), and reranked only 20
candidates when enabled.

## The live runtime

This is the **default** (`retrieval_mode: generators`). Every entrypoint — CLI,
Streamlit, demo, benchmark, HTTP service — runs it.

```
SearchRequest(principal, raw_query)
  → EntitlementResolver → AuthorizedUniverse            (fail-closed)
  → QueryPlan: Q0 raw (immutable) … Q5 bounded alternate
  → independent generators, run concurrently, each over the authorized
    universe only, results collected in submission order:
        BM25F k=100 · SPLADE k=100 · dense{identity,purpose,schema} k=50 each
        · family prototypes k=50 · [late-interaction — built, disabled]
  → source-preserving union   (masks, not zeros; source-exclusive flags)
  → recall-risk policy → LOW/MEDIUM/HIGH
  → source-preserving shortlist  (quota floors + fused remainder; 120 / 200)
  → cross-encoder over the FULL shortlist, always fed the raw Q0
  → fusion: RRF fallback (no trained artifact ships)
  → instance scoring → max-oriented family aggregation
  → FAMILY-FIRST EXPANSION: every authorized instance of each leading family,
    whether or not any retriever found it, cross-encoded on the same raw Q0
  → best authorized instance per family
  → three-way decision → grounded response + telemetry
```

### Family-first expansion

Retrieval nominates *instances*, but 533 titles on this estate exist on more than
one row with different field sets, prompts and data sources. Aggregating only
shortlist survivors answers the wrong question — it reports the best *retrieved*
copy, not the best copy.

Three properties, all asserted in `tests/test_family_expansion.py`:

* **Expansion decides which copy, never which family.** An expanded instance
  carries a cross-encoder logit and no fused score; folding that into the family's
  max would mix two incomparable scales and let a family climb the ranking by
  owning more rows — the exact bias max-oriented aggregation exists to prevent.
* **Authorization is re-resolved.** Expansion is a new route into a response, and a
  route that bypasses the generators must not also bypass their check.
* **The cost bound is reported.** An expanded instance has no retrieval evidence,
  so each costs a full cross-encoder pass; `max_expanded_instances` caps it and
  `instances_dropped_by_cap` records what the cap, rather than the catalog, hid.

Expanded instances are masked `None` against every generator, never 0.0 — the same
contract the union enforces. They were not rejected by the retrievers; the
retrievers never voted.

The system invariant: **no single retriever, score, rewrite or hand-weighted sum
decides what reaches the cross-encoder.** A report nominated by any generator enters
the union and is eligible for the shortlist.

## Measured results

120 v2 validation queries, real models, real 4000-row estate
(`artifacts/candidate_survival_v2.json`). These describe the plain fused shortlist
fill, which is what serves. They predate the dense tie-determinism fix, whose
effect is bounded: on the 265-query fusion evaluation it moved
`instance_recall@1_within_family` from 0.8947 to 0.8684, about one query out of
38 measurable ones. Reproduce with:

```bash
uv run python scripts/measure_survival.py --config configs/legacy_generators.yaml \
    --relevance-root data/relevance_v2 --split validation --queries 120 \
    --output artifacts/candidate_survival_v2.json
```

**Recall by stage**

| Stage | Recall |
|---|---|
| generated / union | 0.624 |
| shortlist / rerank / fusion | 0.542 |
| **after family expansion** | **0.576** |
| final (top-10), all 120 queries | 0.046 |
| final (top-10), the 88 answered queries | 0.062 |

**Family expansion recovers +0.034 recall.** That is the first direct evidence it
does anything. Under the v1 labels it was unmeasurable *in principle* — not one of
their queries graded two instances of the same family — and the earlier finding
that it "changed no metric" was a fact about those labels, not about expansion.

**Recall by generator, and what each uniquely contributed**

| Generator | Recall | Unique relevant contributed |
|---|---|---|
| bm25f | 0.540 | 11 |
| splade | 0.350 | 10 |
| dense_schema | 0.302 | 3 |
| dense_identity | 0.202 | 1 |
| prototype | 0.183 | 2 |
| dense_purpose | 0.031 | **4** |

Every generator contributes candidates nothing else found. Under v1 only bm25f (5)
and splade (1) did, and every dense view contributed zero — which is the reading
that nearly got the purpose view deleted. It has the lowest recall here and the
third-highest unique contribution; its "0 unique" was an artifact of labels that
could not pose the queries it exists for.

Latency, warm, ONNX backend: **p50 3.7 s / p95 7.3 s** over 10 mixed queries, of
which **rerank is 85%**. The survival run's own p50 was 1.9 s on shorter queries.

### How to read these numbers honestly

**They are not comparable to the v1 figures.** v2 queries are harder by
construction: the vague ones are *enforced* to share no content token with their
target's title, which is precisely what the old labels could not produce. Union
recall of 0.624 against v1's 0.872 is a change of question, not a regression.

**They measure agreement with a generator.** No human has judged any of this. The
v2 generator builds each query from a chosen report's metadata so its grades are
true by construction — but only under its own assumptions.

**The bottleneck is ranking, and it is severe.** 54% of relevant instances reach
the cross-encoder; 4.6% appear in the final top 10. The reranker is general-domain
and unadapted to this vocabulary, and fine-tuning it is the largest piece of
unstarted work.

**Neither learned model is served.** Both were trained on v2, evaluated against the
fallback each would replace, and refused by the approval gate: the decision head
loses to the deterministic policy (0.6139 vs 0.7151 macro per-class recall, and
0.677 vs **1.000** on NO_CONFIDENT_MATCH), and the fusion model loses to RRF
(0.0931 vs 0.1064 family nDCG@5). An earlier run had the fusion model ahead by
+0.0031; that did not survive regenerating the feature pass, which is the correct
reading of a delta that small on 265 queries. RRF and the deterministic policy
still serve, and still say so in every response.

## Component notes

### Source floors, then family-diverse remainder

The shortlist is filled in two parts, and only the second changed.

**The exclusive floors are untouched.** Each source keeps its reserved slots for
candidates only it found, unclaimed reservations still widen the fused fill, and
a purpose-exclusive candidate still takes its slot even when its family is
already represented. That is the whole point of the shortlist and nothing here
negotiates with it.

**The fused remainder now prefers unrepresented families.** It walks the same
fused ordering twice: the first pass admits candidates whose family nothing has
admitted yet and defers siblings of families already present; the second gives
whatever depth is left to those deferred siblings. Depth is never exceeded, no
instance is admitted twice, and the final ordering is still by fused rank.

The reason is that aggregation is max-oriented over families. A second copy of a
family already in the shortlist can only improve *which copy* that family offers;
a family with no representative at all cannot be returned however good it is. And
a deferred sibling is not lost — family-first expansion cross-encodes every
authorized instance of a leading family anyway, so deferral moves work from the
shortlist to expansion rather than discarding it. That hand-off is visible in the
tests: a sibling that used to be selected with a retrieval rank is now selected
with the "no fused rank" sentinel, having arrived through expansion.

`shortlist.diversify_families: false` restores the plain fused fill exactly, as an
ablation. Telemetry reports `family_diversity_enabled`, `distinct_families`,
`sibling_entries` and `deferred_then_admitted`.

**No recall gain is claimed for this.** The survival numbers below predate it and
are labelled as such; re-measuring under the new fill is a run of
`scripts/measure_survival.py`, not an assumption.

### Cross-encoder pair deduplication

Within one query the model input is `(query, text)`, so two candidates whose
authoritative text is byte-identical are the same forward pass computed twice.
One shared helper — used by both the shortlist rerank and the expansion rerank —
scores each distinct text once in first-seen order and fans the results back to
every candidate that shares it. A scorer returning anything but one score per
unique input raises rather than being padded or truncated into alignment, because
a misalignment attaches one report's score to another and looks exactly like a
correct answer.

Scores are unchanged by construction: identical text implies an identical logit,
so this is cost and telemetry only, and no evaluation or cache is invalidated.

Telemetry separates candidates from work: `cross_encoder_pairs_submitted`,
`cross_encoder_model_pairs`, `cross_encoder_deduplicated_pairs`, and the same trio
for expansion, plus `expansion_non_finite_scores` (previously filtered without
being counted, so a batch that scored nothing was indistinguishable from one that
never ran).

**No latency improvement is claimed.** Reranking is 85% of measured query latency,
which is why the duplicate work is worth removing, but the size of the saving
depends on how much duplicate text real shortlists actually contain. These
counters are the instrument for measuring that, not the measurement.

### The purpose view has almost no recall and is kept anyway

`Description` holds **8 distinct values across 4000 rows** and `Area Where Used` is
68% empty, so the purpose view carries little signal — 0.031 recall, the lowest of
any generator, while consuming a 10-slot shortlist quota.

It was slated for removal on the v1 reading of "0 unique contributions". On v2 it
contributes **4** unique relevant candidates — third-highest of the six — so that
reading was an artifact of labels which could not pose the vague, outcome-shaped
queries a purpose view exists to answer. Low recall and high uniqueness is exactly
the profile source preservation is built to protect: a generator that only ever
finds what others already found is the one worth cutting, and this is not that.

The **interface** view *is* auto-disabled: distinct-text ratio 0.079, below
`retrieval.dense.min_distinct_text_ratio`, recorded as `degenerate_low_entropy`
with `fallback: generator_disabled`. Note that ratio does not predict usefulness —
purpose sits at 0.202, twice the floor, and still has the worst recall.

### Family identity is the normalized title

Not title-plus-field-set: that distinguishes 4000 rows into 3999 families, making
aggregation a no-op. `title_key` gives **3280 families, up to 7 instances each**.

### The cross-encoder runs on ONNX Runtime

Parity-gated, not assumed: against float64 torch over 200 real (query, report)
pairs, max |logit delta| 7e-06, **zero** ordering changes, **zero** non-finite
scores. The last of those is the reason it is the default — see below.

`rerank.backend` selects it and `torch` remains the fallback. Re-run
`scripts/export_cross_encoder_onnx.py --verify` on any new host: the NaN this
avoids is a property of one torch build on one platform, not of the weights.

### Cross-encoder numerical stability

On this machine (torch 2.11.0, arm64 macOS 13, mkldnn unavailable) the MiniLM
cross-encoder returns **NaN in float32 for any input beyond ~24 tokens** — which is
every real report. `TorchCrossEncoder` probes once at load and escalates to float64,
recording `cross_encoder_dtype_escalated` in telemetry. float64 is exact and ~2.3×
slower, which is most of the observed latency. On a machine where float32 is sound
the probe passes and nothing is paid.

This was only visible because reranking scores were checked; a silent NaN makes the
reranker appear to run while scoring nothing, and every downstream margin becomes
meaningless.

### Fusion orders by reranker, ties broken by RRF

Adding the cross-encoder into the RRF sum does not work: rank fusion discards score
magnitude, so a decisively better candidate gains only one rank — worth 1/61 − 1/62
≈ 0.0003, while two extra agreeing retrievers are worth 0.033. The reranker would be
outvoted by retrievers that never read the query and the report together, and no
weight fixes it robustly. The fallback therefore orders by cross-encoder rank and
uses the RRF sum to break ties. Still purely rank-based; no logit is compared to an
RRF score. `RrfFuser(cross_encoder_priority=False)` preserves the rank-sum variant
as a measurable ablation.

## What is not trained, and what that means

| Component | State | Active fallback |
|---|---|---|
| Fusion model | not trained | `rrf_with_cross_encoder_priority` |
| Decision head | not trained, not calibrated | `deterministic_three_way_policy` |
| Late interaction | implemented, no artifact | `generator_not_constructed` |
| Authorization | no ACL source exists | `allow_all_dev_default` |
| views.interface | degenerate | `generator_disabled` |

Every one of these is reported in `Result.fallbacks`, in the bundle manifest's
`active_fallbacks`, and in the rendered CLI output. Nothing uncalibrated is
described as a probability: fusion scores are labelled
`unnormalized_ranking_score` and `decision_calibrated` is `false`.

`reportfinder-train fusion|decision` both run, but write artifacts with
`approved: false`, which the loaders refuse to serve. That is deliberate — the
training path should be exercisable long before anyone is entitled to promote its
output.

## Commands

```bash
uv run reportfinder-bundle build   --config configs/legacy_generators.yaml -v
uv run reportfinder-bundle verify  --config configs/legacy_generators.yaml
uv run reportfinder-bundle inspect --config configs/legacy_generators.yaml

uv run python -m reportfinder --mode legacy_single_file \
    --retrieval-mode generators "why are we losing people"

uv run reportfinder-train decision --relevance-root data/relevance
uv run reportfinder-train fusion   --config configs/legacy_generators.yaml

# Candidate survival at every pruning boundary
uv run python scripts/measure_survival.py --config configs/legacy_generators.yaml \
    --queries 60 --output artifacts/candidate_survival_generators_expansion.json

# The HTTP service (needs the `api` extra: uv sync --extra api)
uv run uvicorn reportfinder.api.service:app --port 8080
curl -XPOST localhost:8080/v1/search -H 'content-type: application/json' \
     -H 'x-principal-id: alice' -d '{"query":"headcount by organization"}'

# Lint, types, tests -- the same three commands CI runs
uv run ruff check src tests scripts app.py demo.py calibrate.py
uv run mypy
uv run pytest tests/ -q -ra
```

Building the bundle takes ~4.5 minutes on this hardware (SPLADE encoding of 4000
documents dominates). Rebuilds reuse unchanged view vectors by content hash, and
the bundle id is keyed on corpus content plus *index* configuration only — tuning a
shortlist depth or risk threshold does not invalidate 4000 encoded documents.
