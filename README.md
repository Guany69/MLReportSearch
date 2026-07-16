# reportfinder

Type a plain-English request, get back the Workday-style custom report definition that answers it.

Retrieval is done by an **unsupervised dual-space product-of-experts model** over ~4000 report
definitions. There are no labels, no training signal, and no hand-weighted keyword scorer — two
independent representations of the corpus each induce a probability distribution over the reports,
and the answer is their geometric mixture, renormalized into a proper posterior.

Everything runs locally. The only network access is the one-time sentence-transformer weight
download; after that it is fully offline.

---

## Quick start

```bash
uv sync   # uv manages the venv and Python 3.11+

# --- Phase 1 (legacy, single file with a Fields column) — the default ---
#     ./data/Reports.xlsx
uv run python -m reportfinder "show me terminated workers by supervisory organization"

# --- Phase 2 (report catalog with NO Fields + field dictionary) ---
#     ./data/Phase2_Report_Catalog_No_Fields.xlsx
#     ./data/Phase2_Field_Dictionary.xlsx
uv run python -m reportfinder --mode phase2_dual_file "worker location, company and supervisory organization"

# UI (mode selectable in the sidebar)
uv run streamlit run app.py

# Acceptance demo — 6 example queries end to end, either mode
uv run python demo.py --mode phase2_dual_file
```

The first run builds the index (parse → encode → fit LSA) and caches it to `./.cache/`.
Subsequent queries load the cache and answer in well under a second. Force a rebuild with
`--rebuild`.

Useful flags:

```bash
uv run python -m reportfinder "monthly turnover by organization" -v          # posterior + import diagnostics
uv run python -m reportfinder "gross pay by pay group" --alpha 0.3           # lean on the LSA expert
uv run python -m reportfinder "workers with termination date" --field-expert # enable the 3rd expert
uv run python -m reportfinder --rebuild                                      # rebuild index, no query
uv run python -m reportfinder "..." --data /path/to/other.xlsx               # different Phase 1 workbook

# Phase 2 ingestion controls
uv run python -m reportfinder --mode phase2_dual_file \
    --catalog data/Phase2_Report_Catalog_No_Fields.xlsx \
    --field-dictionary data/Phase2_Field_Dictionary.xlsx "..."
uv run python -m reportfinder --mode phase2_dual_file --ambiguity-policy strict "..."
uv run python -m reportfinder --mode phase2_dual_file --no-composite "..."   # disable BO corroboration
uv run python -m reportfinder --mode phase2_dual_file --fuzzy "..."          # opt-in fuzzy names
```

---

## Phase 2: two-file ingestion

In Phase 2 the catalog **no longer has a `Fields` column**. The relationship arrives inverted: each
row of the *field dictionary* lists, in `Where_Used`, the reports that use that field. Ingestion
reconstructs report → fields from it.

Both modes produce the **same internal corpus**, so the ML pipeline never learns which spreadsheets
produced it. `frame["fields"]` remains a `list[str]` of field names — exactly its Phase 1 meaning —
and Phase 2 metadata arrives in additive columns that are empty in legacy mode.

### Matching precedence

Report titles are **not unique** (533 titles occur on more than one catalog row), so ~30% of
`Where_Used` entries are ambiguous by name alone. Matching runs in order, and every link records the
method used:

| Method | Meaning |
|---|---|
| `exact_name` | unique exact title match |
| `normalized_name` | unique match after cosmetic normalization |
| `composite_business_object` | name + `Business Object == Data Source` narrows to exactly 1 |
| `ambiguous_multi` | still >1 candidate — see policy below |
| `fuzzy` | **off by default**; candidate-blocked, never a Cartesian scan |

**Normalization** folds only cosmetic differences (NFKC, case, whitespace runs, dash glyphs) and
never touches display values. Meaningful tokens are preserved: `Report`, `Report v2`, `Report - Copy`,
`Report - Summary`, `Report - Current`, `Report - Audit` stay six distinct reports.

**Field identity** is `business_object | field_name`, so `Worker|Status` and `Job Requisition|Status`
never merge. Repeated rows merge compatible metadata, keep all provenance, and **flag conflicting
non-empty values rather than overwriting them**.

`Where_Used` splits on `;` and line breaks — **never on commas**, which legitimately occur inside
report names ("Headcount by Region, Country and Company").

### The ambiguity trade-off (measured)

1,936 links (227 reports) share *both* a title and a data source, so nothing in the data can decide
them. Measured against Phase 1's authored `Fields` as ground truth:

| Policy | Recall | Precision | Field-blind reports |
|---|---|---|---|
| `strict` — withhold undecidable links | 0.943 | 0.943 | **227** |
| `permissive` (default) — attach to all candidates | **1.000** | 0.984 | **0** |

**Default is `permissive`**, chosen deliberately: `strict` leaves 227 reports (5.7%) with no fields
at all, blinding them to the core ML signal. The trade is real and is *not* hidden — every such link
is recorded `AMBIGUOUS` with its full candidate list, counted in the import summary, and disclosed in
the result explanation ("…could not be uniquely resolved — this report shares its name with others").
Switch with `--ambiguity-policy strict`.

Note the granularity: `Where_Used` says "field F is used by a report *named* N" — it cannot say which
duplicate-titled row. **At title level, reconstruction is exact (precision 1.000, recall 1.000
against real ground truth)**, which is also the granularity family collapse operates at.

### Import diagnostics

Every run reports counts (never field values) — `-v` on the CLI, an expander in the UI:

```
Import summary [phase2_dual_file]
  reports: 4000 read -> 4000 valid | 533 duplicate identities over 1253 rows
  fields:  3234 rows read -> 3234 valid -> 3219 unique identities
  links:   50464 created from 48390 Where_Used entries
           exact=33825 normalized=0 composite=12629 fuzzy=0 ambiguous=1936
           0 duplicate links removed
  gaps:    0 unmatched Where_Used | 0 reports with zero fields
           | 0 fields matched to no report | 0 fields with empty Where_Used
  quality: 0 metadata conflicts | 0 row errors
  corpus:  3882 families | 1.1s
```

### Known Phase 2 behaviour change

Phase 2 yields **3882 families vs Phase 1's 3999**. Reports sharing a title *and* data source get
identical reconstructed fields and therefore collapse into one family. `Where_Used` genuinely cannot
distinguish them — Phase 1 could, because their authored field lists differed. This is family
collapse working as designed (representative = most-run copy; `family_size` shows the count) and is
reported rather than hidden.

---

## The data, and its quirks

The workbook is parsed exactly one way, and the parser asserts it rather than guessing:

- **The real header is on the third physical row** (`pandas.read_excel(path, header=2)`). Above it
  sit a banner row and an "End Date" row.
- `Fields` and `Report Prompts` are `;`-separated lists — split, stripped, empties dropped.
- **`Description` is ignored for matching.** It has only 7 distinct values across 4000 rows, so it
  carries no discriminating signal. Including it would add noise to every document equally.
- **Nulls in `Last Run Date` / `Last Updated Date` are never zero-filled.** A report that has never
  run is not a report that ran on the epoch. Nulls are preserved as `NaT`, flagged with
  `last_run_missing` / `last_updated_missing`, and surfaced as "not recorded".
- **Exact-duplicate definitions are collapsed into families.** Identity is `title + field set`
  (order-insensitive, case-folded). Without this, near-identical copies flood the top-k with the
  same answer repeated. The representative is the most-run copy (ties broken by most recent run) —
  the one people actually use — and `family_size` records how many copies exist.

## The model

### 1. Field-weighted document

Each report family becomes one document, with the discriminating zones repeated so they dominate
the representation:

```
doc(r) = title×3 + fields×3 + prompts×2 + category + data_source + report_type + tags
```

The repetition is a *representation* choice, not a scoring rule: emphasis is baked into the vector
once, and both experts then read the same document independently.

### 2. Two independent spaces

| | Space | Built by | What it's good at |
|---|---|---|---|
| **A** | Dense semantic, `E` (n×d) | local sentence-transformer (`BAAI/bge-small-en-v1.5`, falling back to `all-MiniLM-L6-v2`) | paraphrase — "people who left" ≈ "terminated workers" |
| **B** | Latent lexical, `U` (n×200) | `TfidfVectorizer` (uni+bigrams) → `TruncatedSVD(200)` | exact jargon and co-occurrence — "supervisory organization", "gross to net" |

Both matrices are L2-normalized row-wise, so a dot product *is* cosine similarity. Both are fit
unsupervised on the corpus itself.

The two spaces fail in different directions, which is the whole point: the dense encoder blurs
domain jargon into generic HR-speak, and LSA can't see a paraphrase it has no term overlap with.
Independent errors are what make the product meaningful.

### 3. Query → posterior

The same transforms map a query `q` into `e_q` and `u_q`, giving two similarity vectors over all
reports:

```
s_dense = E · e_q          s_lsa = U · u_q
```

Each becomes a distribution via a tempered softmax, and the two are fused as a **product of
experts** — a geometric mixture, renormalized over the whole corpus:

```
P_dense(r|q) = softmax(s_dense / T_d)
P_lsa(r|q)   = softmax(s_lsa   / T_l)

P(r|q) ∝ P_dense(r|q)^α · P_lsa(r|q)^(1-α)
```

**Why a product and not a sum.** A weighted sum lets a single confident expert drag a candidate to
the top on its own. A product requires *both* experts to find the candidate plausible — either one
can veto by assigning low probability. That is the behaviour you want from a retrieval model whose
two views are supposed to corroborate each other, and it's what makes the resulting number a
posterior rather than an uncalibrated score.

Computed in log space (`α·log P_dense + (1-α)·log P_lsa`, then log-normalized) for numerical
stability.

### 4. Confidence and the decision rule

The payoff of having a real posterior is that its *shape* is informative:

- **p1** — top-1 probability
- **p2** — runner-up, giving the margin `p1 − p2`
- **H(P)** — Shannon entropy over all reports, in bits (reported normalized by `log₂ n`)

```
if p1 ≥ τ and (p1 − p2) ≥ δ:  return ONE confident report
else:                          return top-5, labelled "ambiguous — did you mean one of these?"
```

A peaked posterior means the corpus really does contain one answer. A flat one (high entropy, thin
margin) means the query is ambiguous *against this corpus* — which is a finding, not a failure, and
the model says so instead of bluffing a top-1.

### 5. Optional third expert

`--field-expert` adds `P_field` as another PoE factor: it mines the field vocabulary from the corpus
and detects multi-word field names the user asked for verbatim, forming a smoothed distribution over
the reports carrying them. Additive smoothing keeps every report in the support — a hard zero would
let a string match veto a report outright, which is too strong a claim for a heuristic. Off by
default; it's deliberately thin, and not a rule engine.

---

## Config knobs

All defaults live in [`src/reportfinder/config.py`](src/reportfinder/config.py) and are overridable
per-call from the CLI.

| Knob | Default | Effect |
|---|---|---|
| `T_d` (`--t-dense`) | `0.05` | Dense softmax temperature. Lower ⇒ peakier, more confident. |
| `T_l` (`--t-lsa`) | `0.05` | LSA softmax temperature. |
| `α` (`--alpha`) | `0.6` | Mixture weight. `1.0` = dense only, `0.0` = LSA only. |
| `τ` (`--tau`) | `0.15` | Min top-1 probability to return a single answer. |
| `δ` (`--delta`) | `0.03` | Min margin `p1 − p2` to return a single answer. |
| `top_k` (`--top-k`) | `5` | Candidates shown when ambiguous. |
| `use_field_expert` (`--field-expert`) | `False` | Enable `P_field`. |

Phase 2 ingestion knobs:

| Knob | Default | Effect |
|---|---|---|
| `ingest_mode` (`--mode`) | `legacy_single_file` | `legacy_single_file` or `phase2_dual_file`. |
| `ambiguity_policy` (`--ambiguity-policy`) | `permissive` | `permissive` attaches undecidable links to all candidates; `strict` withholds. |
| `enable_composite_match` (`--no-composite`) | `True` | Corroborate ambiguous names with `Business Object == Data Source`. |
| `enable_fuzzy_match` (`--fuzzy`) | `False` | Constrained fuzzy report-name fallback. |
| `fuzzy_threshold` | `0.93` | Similarity cutoff when fuzzy is on. |

Phase 2 representation zones — all default to **weight 1** so new signals enrich `doc(r)` without
displacing the Phase 1 zones ranking is tuned around: `w_field_description`, `w_business_object`,
`w_domain`, `w_field_categories`, `w_builtin_prompts`, `w_related_business_object`, and
`w_field_type` (default **0**; only 9 values across 3234 rows, so it's weak).

**`Authorized Usage` is deliberately excluded from scoring.** It has exactly **one** distinct value
("Default Areas") across all 3234 dictionary rows, so it cannot discriminate between reports — the
same trap as Phase 1's `Description` (7 distinct values). It is kept as displayable metadata but
never enters `doc(r)`. The brief's "authorized-usage filtering/penalties" is not implementable as
signal on this data.

Tuning intuition: **T** controls how sharp each expert's opinion is (it sets what counts as a
meaningful cosine gap); **α** controls who to believe when they disagree; **τ/δ** control how willing
the system is to commit to one answer. Raising τ trades coverage for precision — more queries come
back as "ambiguous" rather than a confident-but-wrong top-1.

---

## Calibrating τ and δ on your corpus

`τ` cannot be chosen in the abstract, because the posterior's sharpness depends on the **semantic
redundancy of the corpus**. A corpus of genuinely distinct reports concentrates mass; one with deep
clusters of near-synonymous reports spreads it across true near-duplicates. That is the model
behaving correctly — but it moves where τ should sit.

`calibrate.py` picks the thresholds empirically, with **no labels**:

```bash
uv run python calibrate.py --n 300
```

It uses each report's own title as a query — the correct answer is then known by construction, so
the supervision comes from the corpus's own structure. It reports `hit@1`, the `p1` distribution,
and a τ sweep of coverage vs precision.

**This probe is leaky and optimistic**: titles are part of `doc(r)` at ×3 weight, so it measures an
upper bound, not real-world accuracy on paraphrased queries. Use it to compare knob settings and to
put a ceiling on τ — a production τ should sit meaningfully below whatever looks ideal here.

On the real workbook (`data/Reports.xlsx`, 4000 rows → 3999 families) it produces:

```
hit@1: 60.5%           p1 median 0.097

   tau   coverage   precision
  0.05     53.5%      74.3%
  0.10     39.2%      76.4%
  0.15     21.8%      78.2%   <- default
  0.20     11.2%      77.8%
  0.30      2.2%      88.9%
  0.40+     0.0%       never fires
```

**Precision rises with τ** (74% → 76% → 78% → 89%) even though absolute confidence is much lower
here than a naive first guess — that's the evidence the posterior stays calibrated: a higher `p1`
still means a likelier-correct answer, so thresholding on it buys real precision, even on a corpus
this redundant.

The lower ceiling versus a "clean" corpus is a real, explainable property of this workbook, not a
bug: titles are heavily combinatorial (e.g. 94 report titles containing "Terminat*", 132 containing
"Transfer"/"Mobility"/"Movement" — the same report stem crossed with many organizational dimensions:
by Company, by Region, by Business Unit, by Month...). That structure genuinely spreads posterior
mass across true near-duplicates for many queries, so the default τ=0.15 keeps most queries in the
"ambiguous — did you mean one of these?" branch rather than committing to a single answer. That is
the model reporting an honest "this corpus doesn't have one unambiguous answer" rather than bluffing
a confident-sounding top-1. Lower τ (e.g. 0.05–0.10) trades some precision for meaningfully higher
coverage if a single best-guess answer is preferred over an ambiguous list.

(The bundled synthetic test fixture, used by the test suite, calibrates much higher — hit@1 76%,
τ=0.15 gives 61% coverage at 88% precision — because it's less combinatorially redundant than the
real estate. Regenerate and compare with `uv run python tests/make_fixture.py && uv run python
calibrate.py --data tests/fixtures/fixture_reports.xlsx`.)

---

## What "the report" means here

The workbook contains report **definitions** (metadata), not report result-sets. So the answer to a
query is the identified definition, presented in full: title, category, data source, report type,
field list, prompts, run count, recency, confidence %, and a one-line "why it matched" showing the
overlapping terms/fields and the dense-vs-LSA evidence split.

**The system never fabricates report data rows.** It cannot tell you *who* was terminated last
quarter — it tells you which report to run to find out.

The "why it matched" line reports each expert's contribution as **lift**: `α·(log P_expert(r) +
log n)`, i.e. how much that expert raises the report above a uniform prior, in nats. When both lifts
are positive they're shown as a percentage split; when one expert actively disfavours the candidate
a share would be meaningless, so the raw lifts are shown instead.

---

## Layout

```
src/reportfinder/
  config.py      all knobs and defaults (model + ingestion)
  data.py        Phase 1 workbook parsing, ;-splitting, family collapse
  ingest/        Phase 2 ingestion — kept strictly out of ML scoring
    normalize.py       display vs match values, header aliases, multi-value splitting
    models.py          typed records, links, EnrichedField, ImportSummary
    catalog.py         ReportCatalogLoader.load(path)  -> [ReportCatalogRecord]
    field_dictionary.py FieldDictionaryLoader.load(path) -> [FieldDictionaryRecord]
    linker.py          ReportFieldLinker.link(reports, fields) -> LinkResult
    enrich.py          builds the ML-facing corpus (the seam both modes converge on)
    __init__.py        build_corpus(cfg) — mode dispatch
  represent.py   doc(r) construction, dense + LSA spaces, cache build/load
  model.py       the product-of-experts model, posterior, decision rule, explanation
  render.py      terminal formatting of report definitions
  __main__.py    CLI
app.py           Streamlit UI
demo.py          acceptance demo, ~6 example queries
calibrate.py     label-free τ/δ calibration probe
tests/
  make_fixture.py         generates a Phase 1 FIXTURE workbook (not the real data)
  phase2_fixtures.py      builders for small, explicit Phase 2 fixtures
  test_pipeline.py        parser + model tests
  test_phase2_ingest.py   loading, normalization, linking, enrichment
  test_phase2_regression.py backward compat + doc(r) regression + real-data equivalence
  test_phase2_search.py   search regression on the real Phase 2 estate
```

Run the tests with:

```bash
uv run pytest tests/ -q
```

They cover the parsing quirks (header row, `;`-splitting, null preservation, family collapse) and
the model's actual claims — that the posterior is a valid distribution, that `α=1`/`α=0` reduce to
the pure dense/LSA experts, and that the fused winner sits in the upper tail of *both* experts
(the veto property that separates a product from a sum).

## Notes

- **Cache invalidation** is by signature over the data file (path, size, mtime) plus every knob that
  changes the representation (model name, SVD components, TF-IDF settings, zone weights). Change any
  of them and the cache rebuilds itself rather than silently serving stale vectors. Query-time knobs
  (α, T, τ, δ) don't affect the cache — they're applied per-query, so you can sweep them freely.
- **`tests/make_fixture.py` is not the real dataset.** It generates a synthetic workbook with the
  same structural quirks so the pipeline can be exercised without the real file present. It's used
  by the tests; `demo.py` uses your real workbook.
