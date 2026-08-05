# Engineering report: closing plain-language search gaps

## Recorded baseline

The pre-change snapshot is `artifacts/baseline_v2_capture.json`. The supplied warm-cache benchmark baseline was:

| Signal | dev (18) | held-out v1 (15) | transfer v1 (12) |
|---|---:|---:|---:|
| Hit@1 / Hit@5 / MRR@10 | .889 / .889 / .889 | .667 / .733 / .700 | .750 / .750 / .750 |
| `ANSWERABLE` emitted | 0 | 0 | 0 |
| `NEEDS_CLARIFICATION` | 17 | 14 | 11 |
| latency p50 | 204 ms | 188 ms | 218 ms |

These are synthetic/weak development signals, not production accuracy. A warm `ReportFinder` produced the latency figures; a cold convenience `find()` also pays representation loading and is reported separately when measured.

## Root causes and implemented decisions

- Normalized candidate-share entropy was nearly one for every query, making the answerable gate unreachable. It was replaced by deterministic intent specificity, top-five category dispersion, and a raw rank-score gap.
- Hard-coded answerability thresholds and coefficients moved to validated `Config` fields. Every decision branch now provides a reason, including parser ambiguity, unknown fields, impossible field combinations, and top-candidate missing fields.
- Exact literal canonical fields now become required. High-confidence inferred fields require direct request syntax; broad, typo-derived, morphological, and low-confidence concepts remain soft. Boundary trimming is symmetric.
- The lexicon now covers salary ranges, termination/turnover, hires, mobility, learning, succession, requisitions, applications, benefits, leave, payroll, and performance. Emissions are corpus-resolved and linted.
- Semantic replacement is rule-driven: `Start Date` → `Hire Date` in hiring context, `Manager` → `Previous Manager` for prior-boss language, and grouped `Organization` → `Supervisory Organization`.
- Typo safety protects lexicon/domain words such as `training`; distance-two correction is stricter. Acronym casing and `TA` recruiting/time-away context are explicit.
- Ranking feature schema v4 bounds missing-field and percentile values, fixes vacuous negation, derives prompt/filter/granularity compatibility, and offers `legacy` versus field-first `improved` presets.
- Legacy retrieval degrades to a neutral dense expert when dense inference is unavailable. Cache v5 records local dense availability and treats missing legacy metadata as fallback.
- Evaluation now scores expected statuses, reports confusion and slice/version metrics, supports config ablations and a regression gate, and round-trips the complete qrel schema.

## Ranking preset

The improved deterministic preset prioritizes mandatory coverage (3.40), exact-field evidence (2.40), optional coverage (1.55), original-title overlap (3.00), interpreted-concept/title coverage (4.00), and full-evidence intent coverage (1.00). The interpreted-title signal ignores generic organization/worker/manager concepts and normalizes plurals so it rewards discriminating report intent. Retrieval signals follow: BM25 (.80), dense (.70), RRF (.65), agreement (.40/.20), LSA (.20), and character evidence (.18). Compatibility and operational priors are deliberately smaller. The mandatory gate remains bounded with floor .15 and exponent 1.5; a hard negation/exclusion violation costs 10.

## Decision journal

1. Preserve uncalibrated wording: no probability claim is made without human answerability labels.
2. Prefer general, auditable rules to benchmark-phrase exceptions.
3. Keep held-out v1 immutable and append documented v2 coverage gaps.
4. Ship challengers as ablation controls only; do not introduce heavyweight training dependencies or claim an untrained LTR gain.
5. Stop behavior tuning when remaining misses are corpus ambiguity or label limitations, and retain per-query evidence in artifacts.

## Verification

The suite, compiler pass, behavior probes, three benchmark slices, label audit, and ablation artifacts are the acceptance evidence. Final dense-mode measurements are:

| Signal | dev v1 / v2 | held-out v1 / v2 | transfer v1 / v2 |
|---|---:|---:|---:|
| Hit@1 | .944 / .864 | .733 / .684 | .750 / .688 |
| Hit@5 | .944 / .864 | .800 / .737 | .750 / .688 |
| MRR@10 | .944 / .864 | .767 / .711 | .760 / .695 |
| labelled status accuracy | 1.000 | 1.000 | 1.000 |
| no-answer precision / recall | 1.000 / 1.000 | 1.000 / 1.000 | 1.000 / 1.000 |
| latency p50 / p95 | 198 / 300 ms | 215 / 341 ms | 242 / 300 ms |

The system emitted 4/9/5 `ANSWERABLE`, 16/8/9 `NEEDS_CLARIFICATION`, and 2/2/2 `NO_SATISFACTORY_REPORT` decisions on dev/held-out/transfer v2 respectively. Unlike the baseline, the status classes now discriminate exact, broad, and unsupported requests.

Dev ablations confirm that expansion is material and that the optional challengers do not justify changing the default:

| Ablation | Hit@1 | Hit@5 | MRR@10 | p50 |
|---|---:|---:|---:|---:|
| expansion off | .636 | .682 | .645 | 94 ms |
| typo correction off | .864 | .864 | .864 | 229 ms |
| legacy ranking preset | .818 | .818 | .826 | 206 ms |
| dense off | .864 | .864 | .864 | 67 ms |
| cross-encoder on, dense off | .864 | .864 | .864 | 250 ms |

The typo slice is too small for an aggregate change when correction is disabled, so the targeted safety/correction tests remain the evidence for that component. Dense and the cross-encoder produced no aggregate gain on this synthetic dev set; they remain optional, and LTR remains untrained.

The remaining aggregate misses are chiefly intentional no-answer rows with no relevant report, ambiguity rows whose predicates are empty, and a frozen v1 typo label that requires generic `Organization` even though the accepted semantic replacement correctly requests `Supervisory Organization`. These remain visible in per-query artifacts rather than being hidden with benchmark-specific rules. All numbers remain synthetic/weak engineering measurements.
