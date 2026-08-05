# Migration notes

- `Config.retrieval_mode` defaults to `hybrid`; use `legacy_weighted_logit` to reproduce the previous scoring path.
- `Candidate.retrieval_share` is the preferred name. `probability` and `confidence_pct` remain compatibility aliases but are not calibrated confidence.
- `Result.answerability` contains the new three-way outcome and reasons.
- Cache version `v5-expansion-hybrid` invalidates older representations and fingerprints input SHA-256 hashes, schemas, Python, dependencies, representation config, and resolved local dense-model availability.
- `Config.ranking_preset="legacy"` preserves the previous deterministic coefficients. The default is now `improved`.
- Literal canonical-field mentions are required constraints. High-confidence inferred phrases become required only when directly requested; broad inferred concepts remain preferred or mentioned.
- Cross-encoder and learned rankers remain off unless explicitly configured.
