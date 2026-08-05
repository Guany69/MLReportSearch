# Model improvement decision record

All evaluation currently uses weak/synthetic labels, so this document makes no production-accuracy claim.

| approach | decision | rationale |
|---|---|---|
| BM25F | adopt | preserves fielded evidence and avoids dense truncation |
| BGE bi-encoder | adopt as one channel | semantic recall, but not sufficient alone |
| RRF hybrid | adopt/default | robust to incomparable score scales without labels |
| Exact-field index | adopt | strong, auditable evidence for requested fields |
| Character n-grams | adopt | typo and abbreviation resilience |
| LambdaMART/GBDT | ship interface, disabled | weak labels risk learning generator artifacts |
| Cross-encoder | ship config-gated, disabled | possible precision gain with material CPU latency |
| SPLADE | defer | another model/deployment burden before human qrels |
| ColBERT | defer | storage and serving complexity is premature at this estate size |
| Isotonic/logistic calibration | defer artifact | needs independent answerability labels |
| Conformal prediction | defer | exchangeability and a trustworthy calibration set are absent |
