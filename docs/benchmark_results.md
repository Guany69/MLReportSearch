# Benchmark results

Run on 2026-07-17 against six checked-in weak/synthetic queries: three weak title labels, two weak field-name labels, and one synthetic no-answer query. The sample is intentionally a smoke/regression set, not production accuracy and not evidence sufficient to train LTR or a calibrator.

| metric | legacy weighted-logit | deterministic hybrid |
|---|---:|---:|
| Recall@10 | 0.500 | 0.667 |
| Recall@20 | 0.667 | 0.667 |
| Recall@50 | 0.833 | 0.833 |
| Hit@1 | 0.333 | 0.500 |
| MRR@10 | 0.361 | 0.556 |
| nDCG@5 | 0.333 | 0.583 |
| nDCG@10 | 0.393 | 0.583 |
| P@5 | 0.067 | 0.133 |
| MAP | 0.375 | 0.559 |
| p50 latency | 69 ms | 166 ms |
| p95 latency | 92 ms | 270 ms |

Warm-up excludes one-time model/index initialization. The hybrid improves every early-ranking metric in this tiny weak set, at roughly 2.4× p50 latency. Recall@50 is capped at 0.833 because aggregate ranking metrics include the no-answer query, which intentionally has no relevant document.

ECE/Brier/NLL are emitted for observability, but the answerability fallback is explicitly uncalibrated; values on six synthetic/weak examples do not establish calibration. After adding the OOD lexical-support guard, the synthetic no-answer query abstains; the checked-in JSON artifact is the authoritative rerunnable output.

The cross-encoder was not enabled: the deterministic hybrid already clears the weak regression gate and the plan requires measured incremental gain within a latency budget before enabling it. LTR was likewise left disabled because training it on title/field generators would mainly learn label-generation artifacts.
