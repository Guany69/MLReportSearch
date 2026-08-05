# Final architecture

```text
original query
  -> span-preserving normalization and trie matching
  -> context, specificity, typo, acronym, and requirement resolution
  -> original / expanded / subquery forms
  -> BM25F + dense + LSA + char + structured exact-field retrieval
  -> reciprocal-rank candidate union
  -> bounded structured features and mandatory-coverage gate
  -> optional normalized cross-encoder rerank
  -> semantic support + breadth + required-field answerability
  -> structured explanation and linkage warnings
```

Query forms share a channel-wide normalization scale. Their top-two bounded aggregate rewards corroboration without noisy-or saturation. Exact-field scores are requested IDF-mass coverage in `[0,1]`.

Dense modes are `auto`, `local`, and `off`. If the package/model is unavailable under `auto`, the dense channel is omitted and LSA appears once; it is never duplicated as a fake dense vote. `local` fails with an actionable error. Cache version `v5-expansion-hybrid` includes `corpus_granularity`, configured dense mode, dependency markers, and a local-only fingerprint of resolved model availability. A cache without a recorded dense state loads as LSA fallback, never as a native one-column dense matrix. Query-time lexicon/ranking hashes are recorded without forcing a matrix rebuild.

Answerability ambiguity is intent-specific rather than entropy over a fixed candidate set. It combines the number of high-confidence concepts, presence of a specific field/measure, top-five category dispersion, and the top-two gap on the raw ranking scale. Thresholds and coefficients live in `Config`; the result remains explicitly uncalibrated.

The default `improved` deterministic ranking preset follows this order: mandatory field coverage, exact-field evidence, optional concept coverage, original-title overlap, interpreted-intent coverage, then retrieval corroboration and operational priors. The `legacy` preset preserves the previous coefficients for ablation. Constant compatibility features were replaced by prompt, filter, and granularity evidence derived from report prompts/fields.

Feature inspection is available with `--explain-features`. Results preserve interpreted/suppressed concepts, corrections, required/optional field evidence, retriever ranks, ranking features, answerability reasons, and ambiguous linkage warnings.
