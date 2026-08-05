# Path to production accuracy

Production accuracy is an evidence standard, not a model flag. The repository now has a strong candidate generator, but no real user labels. Therefore it must not be described as production-accurate yet.

## Minimum evidence program

1. Log consented, privacy-reviewed production queries with timestamp, tenant/context fields that are safe to retain, clicked/launched report, reformulations, and explicit “none of these” feedback. Never treat clicks alone as truth; position and familiarity bias are substantial.
2. Sample at least 500 unique queries across frequent, tail, typo, acronym, multi-intent, long, and no-answer slices. Oversample failures and low-agreement queries without losing a representative holdout.
3. Pool top candidates from the current hybrid plus diverse challenger systems. Blind model identity and rank before judgment. Two domain experts independently label `0=irrelevant`, `1=useful`, `2=best`; adjudicate disagreements. Record answerability separately.
4. Split by stable query ID, user/session and time where available—never by query/report pair. Keep the final temporal test set sealed. Near-duplicate queries must remain in one split.
5. Tune fusion/LTR on train, model selection on validation, and report exactly once on test. Fit answerability calibration only on held-out validation predictions. Measure confidence intervals and every critical slice.
6. Shadow deploy. Compare against the current system using interleaving or a randomized experiment, with guardrails for no-result rate, reformulation, time-to-report, report launch, and explicit failure reports. Roll out gradually with rollback.

## Acceptance gate

`reportfinder.evaluation.production.acceptance_gate` refuses a production claim unless judgments are explicitly sourced from production queries, include at least 500 unique queries, reach 0.75 pairwise relevance agreement, and clear initial test thresholds: nDCG@10 ≥ .75, MRR@10 ≥ .70, no-answer precision ≥ .90, recall ≥ .80, and ECE ≤ .08. These are starting service-level targets and should be made stricter or slice-specific with stakeholders.

No aggregate may hide a critical slice regression. At minimum, publish bootstrap confidence intervals for each metric and separate results for rare fields, acronyms, typos, multi-intent, no-answer, short/long, new reports, and duplicate-title/ambiguous-link cases.

## Model changes after labels exist

- Train grouped LambdaMART on query groups using the shipped feature schema. Add cross-encoder score as a feature only if it improves the sealed validation set within the latency budget.
- Mine hard negatives from high-ranked irrelevant results, not random reports.
- Calibrate answerability with logistic or isotonic regression on out-of-fold predictions. Choose abstention thresholds from the business cost of a wrong recommendation versus asking for clarification.
- Keep the deterministic hybrid as fallback for missing/incompatible artifacts and monitor feature drift, score drift, calibration, slice quality, and latency.
- Retrain on a scheduled, versioned dataset only after data-quality checks; never learn directly and immediately from raw clicks.

Until this program clears the gate, the honest release label is “production-oriented architecture, weak-label evaluated,” not “production accurate.”
