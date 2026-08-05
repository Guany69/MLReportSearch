# Relevance dataset

The immutable source bundle is `data/relevance/`. Schema version 1 covers the 10,000-scenario CSV, JSONL seed judgments, candidate/annotation CSVs, three processed Parquet files, and locked train/validation/test ID lists. Catalog identities are physical rows: spreadsheet source row 4 is `R0004`; normalized titles are `title_key` and are not identities.

Labels resolve in this order: adjudicated (weight 1.0), human (1.0), synthetic seed (0.6), unjudged (`None`, weight 0). Unjudged pairs are never negatives. Loaders reject malformed IDs, duplicate pairs, invalid grades, missing columns, title mismatches, split overlap, and feature-version drift with file/row context.

The test split is sealed. Feature fitting, training, tuning, calibration, early stopping, comparison, and label-driven alias mining reject test IDs. Release evaluation requires an explicit release flag and matching split hash. Synthetic results must be labeled “synthetic benchmark — not human-validated.”

Human templates remain blank and unchanged. Generate blinded assignments and import completed work into `data/relevance/annotation/revisions/<version>/`; retain reviewer identity, independent judgments, disagreements, adjudication, and revision history. Never fabricate judgments or replace source templates.

Run `reportfinder-data validate --config <your-config>.yaml` after adding reports or judgments. New files and derived outputs belong in a new versioned directory; never overwrite an existing artifact.
