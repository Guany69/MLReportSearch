# Architecture decision

The production default is a deterministic, corpus-aware hybrid. Query understanding is data-driven: a token trie unifies literal workbook vocabulary and reviewed YAML rules, preserving all nested spans before contextual and structural suppression. Conservative typo correction uses bounded OSA Damerau-Levenshtein; generated acronyms require acronym-like input. Every interpretation retains evidence, origin, confidence, requirement tier, and suppressed alternatives.

Original, expanded, and focused forms are scored on one shared scale. The bounded top-two aggregate gives expansions recall without letting expansion-only evidence overpower a strong original match. Exact-field retrieval accepts resolved required/optional fields and returns an IDF-mass coverage fraction in `[0,1]`; the historical raw-string API remains only for compatibility.

RRF remains the candidate-union method because channel scales differ and no human labels exist. Ranking terms are bounded, with explicit priority coefficients and a smooth mandatory-coverage gate. Cross-encoder reranking remains disabled by default and process-cached when enabled.

Answerability has three public outcomes: `ANSWERABLE`, `NEEDS_CLARIFICATION`, and `NO_SATISFACTORY_REPORT`. It grounds meaningful original-query terms through literal, morphological, acronym, typo, or reviewed expansion evidence. A low ranking score alone does not cause abstention, and one partially unmet requirement causes clarification rather than an unconditional veto.

The answerability score is an uncalibrated engineering heuristic, not a probability. Retriever agreement is overlapping retrieval support, not independent confirmation. LambdaMART is intentionally excluded because there are no human judgments and LightGBM is not operational in the verified environment.
