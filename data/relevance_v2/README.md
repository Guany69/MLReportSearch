# Relevance labels v2 (synthetic, generative)

1771 scenarios / 6047 judgements,
generated from `/Users/gyansaxsna/Desktop/Axis/ML Report Search/data/Reports.xlsx`
(4000 instances,
3280 families) with seed
20260804.

## Why this exists

v1's labels were produced by matching generated queries back against report text.
That is the same operation BM25F performs, so measuring lexical retrieval against
them was circular -- and it showed: BM25F alone reached 0.864 union recall against
the full union's 0.872.

Worse, **zero** v1 queries graded two instances of the same family. Family
expansion and best-instance selection therefore could not be measured at all.

v2 builds each query *from* a chosen target's metadata, so the grade is true by
construction. 450 scenarios exist specifically to distinguish siblings.

## Archetypes

| archetype | count |
|---|---|
| acronym | 150 |
| ambiguous_clarification | 139 |
| misspelling | 150 |
| multi_intent_single | 60 |
| multi_intent_split | 60 |
| negation | 100 |
| no_answer_impossible_combo | 100 |
| no_answer_reserved | 150 |
| short_query | 12 |
| sibling_discriminating | 450 |
| vague_outcome | 400 |

## Splits

| split | scenarios |
|---|---|
| calibration | 293 |
| test | 255 |
| train | 958 |
| validation | 265 |

Groups (target family, plus near-duplicate query clusters) are assigned whole, so
no family's scenarios straddle a split boundary.

## Limitations

- Synthetic. No human judged any of these. Metrics computed against them measure agreement with this generator, not production relevance.
- Grades hold only under the generator's assumptions: that a normalized title identifies a report family, and that carrying a field means being able to answer about it.
- v2 removes v1's lexical-matching circularity -- v1 built labels by matching titles and fields, which is the operation BM25F performs, so it favoured lexical retrieval by construction. v2 still shares corpus vocabulary by necessity.
- Not comparable across label versions. A v2 number is not a better or worse v1 number; it is a different measurement.
- Any model approved against this set carries approval_basis 'synthetic_v2_evaluation' and must be re-earned against human judgements before any production-accuracy claim.
