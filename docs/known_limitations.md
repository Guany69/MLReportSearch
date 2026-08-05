# Known limitations

Everything below is measured on `data/Reports.xlsx` (4000 instances, 3280
families) on one CPU host. Where a number changed from a previously documented
one, the old value and the reason are stated — a metric that moves without
explanation is worse than no metric.

## The labels

- **No human has judged anything.** Zero human relevance, answerability, or
  adjudicated judgements exist. Every metric here measures agreement with a
  *generator*, not production relevance. The v2 generator builds each query from a
  chosen report's metadata, so its grades are true by construction — but only
  under its own assumptions (a normalized title identifies a family; carrying a
  field means being able to answer about it).
- **v1's labels were circular and are no longer used for evaluation.** They were
  produced by matching generated queries against report text — the same operation
  BM25F performs — so measuring lexical retrieval against them was self-fulfilling
  (BM25F alone reached 0.864 union recall against the full union's 0.872). They
  also contained **zero** queries grading two instances of one family, which is
  why family expansion and instance selection could not be measured at all.
- **v2 and v1 numbers are not comparable.** v2's queries are harder by
  construction — vague queries are *enforced* to share no content token with their
  target's title. Union recall is 0.624 on v2 against 0.872 on v1; that is a
  change of question, not a regression.

## Retrieval

Measured on 120 v2 validation queries (`artifacts/candidate_survival_v2.json`).

These describe the plain fused shortlist fill, which is what serves.

| stage | recall |
|---|---|
| generated / union | 0.624 |
| shortlist / rerank / fusion | 0.542 |
| **after family expansion** | **0.576** |
| final (top-10, all queries) | 0.046 |
| final (top-10, the 88 answered queries) | 0.062 |

- **Family expansion is now measurably positive: +0.034 recall.** It recovers
  relevant instances no retriever nominated. This is the first direct evidence for
  it; under v1's labels it was unmeasurable in principle, and the earlier report
  that it "changed no metric" was a statement about those labels, not about
  expansion.
- **Every generator contributes unique relevant candidates** — the claim source
  preservation exists to make good on:

  | generator | recall | unique relevant contributed |
  |---|---|---|
  | bm25f | 0.540 | 11 |
  | splade | 0.350 | 10 |
  | dense_schema | 0.302 | 3 |
  | dense_identity | 0.202 | 1 |
  | prototype | 0.183 | 2 |
  | dense_purpose | 0.031 | **4** |

  Under v1 only bm25f (5) and splade (1) contributed anything unique and every
  dense view contributed zero. **The purpose view was previously slated for
  removal on that basis and has been kept**: it has the lowest recall of any
  generator and the third-highest unique contribution. Its "0 unique" reading was
  an artifact of labels that could not pose the queries it exists for.

- **Ranking, not retrieval, is the bottleneck, and severely so.** 54% of relevant
  instances reach the cross-encoder and 4.6% appear in the final top 10. The
  cross-encoder is general-domain and unadapted; fine-tuning it on this vocabulary
  remains the highest-value unstarted work.

## The learned models — neither is served

Both were trained on v2, evaluated against the fallback each would replace on a
split neither was fitted on, and **both were refused by the approval gate**:

| model | primary metric | learned | fallback | outcome |
|---|---|---|---|---|
| decision head | macro per-class recall | 0.6139 | **0.7151** | refused: loses by 0.101 |
| fusion MLP | family nDCG@5 | 0.0931 | **0.1064** | refused: loses by 0.013 |

- The deterministic policy reaches **1.000 recall on NO_CONFIDENT_MATCH** where the
  learned head reaches 0.677. It is better at the one thing the head exists for.
- **The fusion model's earlier +0.0031 "gain" did not survive re-measurement.** On
  a regenerated feature pass it scores 0.0931 against RRF's 0.1064 — it now loses
  outright rather than winning the headline while regressing a slice. A +0.0031
  delta on 265 queries was noise, and this is what noise looks like when you
  measure it twice.
- Both eval reports now carry per-archetype slices (the decision report shipped
  `slices: {}` before, so the gate could not have seen a per-class regression in
  the one model whose entire purpose is per-class behaviour) and a
  `metric_directions` block.
- So **RRF and the deterministic three-way policy are still what serve**, and are
  still disclosed in every response. `human_validated` is false on both artifacts
  and would stay false even if they were approved: beating a fallback on synthetic
  labels is not human validation.

## A change that was measured and rejected

- **Family-diverse shortlist fill is implemented, switchable, and off.** The idea
  was to spend the fused remainder on unrepresented report families before second
  copies of families already present. An isolated A/B on identical code over 265
  v2 validation queries, comparing the deterministic serving arm, measured it
  losing on every metric:

  | metric | on | off | delta |
  |---|---:|---:|---:|
  | family_ndcg@5 | 0.1026 | 0.1064 | −0.0038 |
  | family_mrr | 0.1143 | 0.1194 | −0.0051 |
  | family_recall@10 | 0.1679 | 0.1755 | −0.0076 |
  | instance_recall@1_within_family | 0.7750 | 0.8684 | **−0.0934** |

  Deferring a sibling out of the shortlist makes the choice of which copy to
  return depend on family expansion, which is bounded by `max_expanded_instances`
  — so the sibling that would have won sometimes never reaches the cross-encoder.
  "Expansion recovers it anyway" held only up to a cost bound this pushed past.
  `shortlist.diversify_families` therefore defaults to `false`. The same standard
  the approval gate applies to learned models: a measured regression does not ship
  on the strength of its rationale.

## The decision thresholds

- `weak_rerank_logit = -9.0` and `clarify_margin = 1.0` are **no longer "set from
  six probe queries"**. A 273-point grid was swept on the calibration split and
  confirmed on validation. The best calibration point (−10.5 / 2.75) scored 0.7902
  there but only 0.7220 on validation against the shipped 0.7151 — a +0.0069 delta
  that does not survive out of sample, and it trades RETURN_RESULTS recall down.
  Re-run on the regenerated feature passes; the conclusion did not move.
  **The shipped values were kept.** That is weak positive evidence, not
  calibration.
- The policy abstains on 27% of v2 validation queries. Whether that is right is
  still unknown: it needs human no-answer labels.
- Nothing in the system presents a score as a probability. No calibration artifact
  ships.

## What is disclosed rather than fixed

- **Serving configuration can drift from build configuration.** A bundle's id
  keys on corpus content plus *index* config, so a shortlist or threshold change
  reuses the encoded documents — correct, and it means one bundle id can serve
  materially different behaviour. Both hashes are now carried and compared, and a
  mismatch surfaces as a warning in search results, telemetry, `/v1/model-info`
  and every feedback record. It is a disclosure, not an error: the vectors stay
  valid, but reproducing a recorded result needs the build-time config.

## Cross-encoder and latency

- **ONNX Runtime is the default backend, on measured parity.** Against float64
  torch over 200 real (query, report) pairs: max |logit delta| 7e-06, **zero**
  ordering changes, **zero** non-finite scores.
- **That last number is why it matters.** torch 2.11 on arm64 macOS 13 returns NaN
  in float32 for any realistic cross-encoder input, so the torch path escalates to
  float64 — correct, and slow. ONNX Runtime does not reproduce the NaN, so it is
  both exact and faster here. **This is host-specific**: re-run
  `scripts/export_cross_encoder_onnx.py --verify` before trusting it elsewhere,
  and set `rerank.backend: torch` if it fails.
- Latency, warm, ONNX, 10 queries, one CPU host: **p50 3.7 s, p95 7.3 s**, of which
  **rerank is 85%**. The survival run's p50 was 1.9 s on its shorter queries.
- **Cross-encoder pair deduplication is a cost change with no measured saving.**
  Identical authoritative text within one query is now scored once and fanned
  back out, which cannot change any score (same text, same logit) and so
  invalidates no evaluation. How much work it removes depends on how much
  duplicate text real shortlists contain; the new
  `cross_encoder_pairs_submitted` / `_model_pairs` / `_deduplicated_pairs`
  counters are the instrument for measuring that, and no percentage is claimed.
- **A previously documented 4× concurrency speedup was an artifact and is
  withdrawn.** "generate 4469 ms → 1104 ms" was measured in one process without
  warm-up, so it timed lazy model loading, not concurrency. Re-measured in
  isolated processes with warm-up: `generate` is ~210 ms and concurrency gives
  **1.09× on that stage, 1.05× end to end**, with 10/10 identical result orderings.
  Concurrency is retained — it is free and provably order-preserving — but it is
  not a meaningful speedup once caches are warm.

## Still absent

- **No production ACL.** `AllowAllResolver` grants the whole estate and is
  disclosed as a development default in every response and trace. `acl_key` and
  `ExplicitAclResolver` are the hooks; neither has seen real entitlement data.
- **The HTTP service authenticates nobody.** It resolves the principal from an
  `x-principal-id` header set by a fronting proxy and refuses requests without one.
  It must not be exposed directly.
- **Late interaction is implemented but not served.** Index format, MaxSim scoring,
  generator, persistence and tests are real; the production ColBERT token encoder
  is not written. Enabling it raises rather than silently substituting another
  dense search.
- **Prototypes are catalog-derived seeds, not user language** — generated
  deterministically from titles and fields, marked `catalog_seed` / `unreviewed`.
- **The Phase 2 estate now runs, and has no evaluation at all.** Both workbooks
  are in the tree and `configs/phase2_generators.yaml` builds and serves against
  them (4,368 reports → 4,299 families, 75,689 links). What does not exist is any
  relevance label, any survival measurement, or any trained artifact for that
  corpus. Its ingest counts are import counts — they say what was read and
  linked, never whether a linked field is the right field — and none of its
  numbers may be compared with the legacy estate's, which is a different corpus
  with a different field reconstruction.
- **274 Phase 2 report references remain unresolved**, and 1,023 links are
  ambiguous by title. An unrecognized `Where_Used` type prefix falls into the
  unmatched count deliberately rather than being skipped, so a new Workday object
  type shows up as a diagnostic instead of silently deleting links.
- **mypy covers 10 packages, not the repository.** `ingest`, `cli`, `ranking`,
  `confidence` and the top-level modules lean on untyped pandas and are separate
  work.

## Pre-existing

- The answerability score is heuristic and uncalibrated.
- The lexicon improves known HR vocabulary gaps but cannot cover every
  organization-specific idiom.
- Report-to-field links reconstructed through duplicate titles can be ambiguous.
  Explanations disclose this rather than presenting inferred linkage as verified.
- Time, comparison, and complex boolean filtering remain deliberately shallow
  compared with field/topic interpretation.
- Dense inference is CPU-bound in the verified environment. `--dense-mode off`
  provides a deterministic lexical/LSA fallback.
- Predicate qrels establish metadata compatibility, not whether a report's runtime
  output truly answers a user's business question.
