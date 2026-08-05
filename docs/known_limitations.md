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

Measured on 120 v2 validation queries (`artifacts/candidate_survival_v2.json`):

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
| decision head | macro per-class recall | 0.621 | **0.713** | refused: loses |
| fusion MLP | family nDCG@5 | 0.1095 | 0.1064 | refused: regresses instance selection (0.868 vs 0.895) |

- The deterministic policy reaches **1.000 recall on NO_CONFIDENT_MATCH** where the
  learned head reaches 0.710. It is better at the one thing the head exists for.
- The fusion model's +0.0031 headline gain is noise on 265 queries, and it is
  worse on the metric v2 was specifically built to measure. The gate refuses a
  model that improves the headline while regressing anything else; that trade is a
  person's call, not an automatic approval.
- So **RRF and the deterministic three-way policy are still what serve**, and are
  still disclosed in every response. `human_validated` is false on both artifacts
  and would stay false even if they were approved: beating a fallback on synthetic
  labels is not human validation.

## The decision thresholds

- `weak_rerank_logit = -9.0` and `clarify_margin = 1.0` are **no longer "set from
  six probe queries"**. A 273-point grid was swept on the calibration split and
  confirmed on validation. The best calibration point (−10.5 / 2.75) scored 0.790
  there but only 0.720 on validation against the shipped 0.713 — a +0.007 delta
  that does not survive out of sample, and it trades RETURN_RESULTS recall down.
  **The shipped values were kept.** That is weak positive evidence, not
  calibration.
- The policy abstains on 27% of v2 validation queries. Whether that is right is
  still unknown: it needs human no-answer labels.
- Nothing in the system presents a score as a probability. No calibration artifact
  ships.

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
- **The Phase 2 dual-file estate does not exist here.** That ingest path is real
  code with synthetic-fixture coverage, but no real dual-file data exercises it.
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
