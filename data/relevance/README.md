# AXIS Report Search Relevance Data

## Source of truth

All report identities in this directory are based on `Phase2_Report_Catalog_No_Fields.xlsx`.
Report keys use the catalog's original Excel row number: row 4 is `R0004`, row 4003 is `R4003`.

Report fields are linked from `Phase2_Field_Dictionary.xlsx` through the `Where_Used` column.
The existing files in the parent `data/` directory were preserved unchanged.

## Directory

```text
relevance/
├── raw/
│   ├── axis_report_search_scenarios_10000.csv
│   └── axis_report_search_seed_judgments.jsonl
├── annotation/
│   ├── candidate_pool.csv
│   ├── human_judgments.csv
│   └── adjudicated_judgments.csv
├── processed/
│   ├── query_plans.parquet
│   ├── ranking_features.parquet
│   └── answerability_features.parquet
├── splits/
│   ├── train_queries.txt
│   ├── validation_queries.txt
│   └── test_queries.txt
├── dataset_manifest.json
└── validation_report.json
```

## How to use the files

1. Use `candidate_pool.csv` to create blinded annotation assignments.
2. Reviewers enter relevance grades and answerability labels in `human_judgments.csv`.
3. Resolve disagreements in `adjudicated_judgments.csv`.
4. Join adjudicated grades to `ranking_features.parquet` by `scenario_id` and `report_key`.
5. Train ranking models only on adjudicated labels.
6. Join query-level answerability labels to `answerability_features.parquet` by `scenario_id`.
7. Use the provided split text files and do not move test queries into training.

## Candidate construction

The candidate pool combines:

- title BM25
- report-field/profile BM25
- character n-gram TF-IDF
- exact required-field postings
- all synthetic positive reports
- all synthetic hard negatives
- duplicate-title variants

Synthetic source and relevance columns are intentionally omitted from the human-review template.

## Validation result

`validation_report.json` records 21 passing integrity checks, including:

- 10,000 unique scenarios
- zero missing report keys
- zero report-title mismatches
- complete field coverage for all 9,050 scenarios with explicit required concepts
- 950 intentional no-answer or clarification scenarios without explicit fields
- identical query-report pairs across candidate, human-review, adjudication, and ranking-feature files
- disjoint train, validation, and test query IDs
- zero positive report-key overlap between splits
- successful Parquet readability and matching row counts

## Warning

The relevance labels are synthetic weak supervision. They should seed candidate pooling and reviewer workflows, not be treated as verified Axis employee judgments.
