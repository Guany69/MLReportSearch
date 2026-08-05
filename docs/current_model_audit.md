# Current model audit

The legacy path creates one repeated-zone document per report family, embeds it with BGE, creates a TF-IDF → SVD vector, and combines both similarities. Its former “product of experts” name overstated the behavior. If

`log P_d(i) = s_d(i)/T_d - log Z_d` and `log P_l(i) = s_l(i)/T_l - log Z_l`,

then renormalizing `α log P_d(i) + (1-α) log P_l(i)` cancels both query-constant normalizers. The result is exactly

`softmax(α s_d/T_d + (1-α) s_l/T_l)`.

Run `python scripts/check_fusion_algebra.py` for a numeric check. The implementation is retained as `retrieval_mode: legacy_weighted_logit`.

## Lineage and information loss

| Source | Canonical column | Retrieval use |
|---|---|---|
| Catalog title | `title` | BM25F, dense, LSA, char |
| Report description | `description` | BM25F |
| Area where used | `area_where_used` | BM25F and ranking |
| Landing/worklet/chart/owner/creator/dates | same-name columns | preserved; operational/display features |
| Dictionary fields | `fields`, `field_meta` | BM25F, exact-field, dense, structured coverage |
| Link provenance | `field_link_confidence`, `ambiguous_link_fraction` | structured ranking and explanation |
| Authorized usage | `field_meta.authorized_usage` | preserved but not scored because the supplied data is non-discriminative |

Before this overhaul, the catalog metadata above was dropped at enrichment. It is now additive in both ingestion modes, with empty typed defaults in legacy mode. `report_id` is the stable source row index; titles are not identifiers.

## Measured truncation and ambiguity

Measured on the supplied **synthetic** Phase 2 estate with the real BGE tokenizer on 2026-07-17:

> The real Phase 2 workbooks are now in the tree and produce different figures —
> 4,368 reports, 4,299 families, 75,689 links, 1,023 ambiguous. See
> `artifacts/phase2_ingest_validation.json` and the README's Phase 2 section. The
> counts in this section describe the synthetic estate this audit was performed
> against and are kept as the record of that audit, not as current facts.

| statistic | tokens |
|---|---:|
| median | 405 |
| p90 | 495 |
| p95 | 525 |
| p99 | 593 |
| max | 895 |

6.72% of 3,882 documents exceed BGE's 512-token limit. Because field metadata is appended after repeated core zones, the lost tail is primarily field descriptions, business objects, domains, categories, prompts, and related objects. This confirms that a single vector cannot be the sole retrieval channel; BM25F and exact-field indexing retain those zones without transformer truncation. A future benchmark may justify separate zone vectors.

The same import produced 50,464 report-field links, including 1,936 ambiguous links, and 533 duplicate report-title identities. Permissive linking remains recall-oriented, but ambiguity now lowers structured rank and is disclosed.

## Confidence finding

The old top-1 softmax share was not calibrated and depended on corpus size and score temperature. It is now called retrieval share. Answerability is a separate three-way decision. The shipped deterministic answerability score is explicitly marked uncalibrated until human or independently adjudicated labels exist.
