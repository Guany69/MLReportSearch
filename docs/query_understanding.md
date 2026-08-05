# Query understanding

`ExpansionEngine` tokenizes while retaining character offsets, then uses one pure-Python token trie for literal corpus phrases and YAML rule phrases. It returns all nested matches. Higher-priority context rules resolve within a span; corpus-derived specificity suppresses contained generic concepts such as `Manager` inside `Previous Manager`; declarative suppression handles semantic overrides.

Requirements are span-local:

- required: literal canonical-field mentions; high-confidence phrase fields with direct `include`/`require`/`must have` or immediately adjacent request wording;
- preferred: `by`, `per`, grouping, and filtering markers;
- mentioned: bare topical language.

A broad `show`, `with`, or `need` does not distribute mandatory status across every concept. Typo/morphological or low-confidence concepts cannot become required, boundaries are applied on both sides of a span, and at most four fields are required.

Rules live in `src/reportfinder/query/lexicon/*.yaml`. Each emission must resolve to an exact current-corpus canonical name. Rule IDs are global, suppression must be acyclic, conflicting same-priority suppressions are rejected, and reserved synthetic-negative words cannot be grounded. `tests/test_lexicon_lint.py` validates every rule against the real workbooks.

Typo correction never rewrites an existing corpus token, lexicon phrase token, protected domain/English word, acronym, mixed alphanumeric, instruction word, or recognized plural. Distance-two corrections require long tokens and strong corpus support. At most two corrections are made, every correction is exposed, and the original query is never overwritten.

Reserved acronyms have a casing policy. Common-word collisions such as `TA`, `BU`, and `CC` require uppercase unless a domain phrase supplies context; `ta partner`, for example, resolves to `Recruiter`, while an unsupported lowercase `ta` does not expand.

Semantic replacements are declarative rules with context and suppression provenance: hire-context `Start Date` becomes `Hire Date`, prior-boss wording becomes `Previous Manager`, and organization grouping becomes `Supervisory Organization`. Suppressed concepts cannot be reintroduced by typo re-probing.

Run:

```bash
uv run pytest tests/test_query_expansion.py tests/test_lexicon_lint.py -q
uv run python -m reportfinder --explain-features \
  "show me attriton by manger"
```
