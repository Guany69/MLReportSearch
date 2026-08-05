# ML search audit

The pre-change active hybrid path parsed intent through literal whole-word matching. The existing alias builder was unreachable. Consequently user vocabulary such as “boss,” “take-home pay,” and “people who left” had no dependable path to `Manager`, `Net Pay`, or termination fields.

The main reliability defects were:

- a query-global mandatory flag made a bare “show” hard-require every literal field;
- any missing mandatory field forced abstention;
- raw exact-field IDF was unbounded relative to RRF and semantic signals;
- each expanded form was independently max-normalized, erasing evidence that the original matched better;
- fifteen declared ranking features were always zero;
- dense dependency failure escaped before the fallback path;
- `corpus_granularity` was absent from the representation cache signature;
- the optional cross-encoder reloaded per query and did not globally re-sort;
- benchmark tests exercised hybrid mode while asserting legacy weighted-logit behavior.

The implementation addresses these at their owning boundaries: query interpretation, query-form aggregation, structured field retrieval, bounded feature scoring, semantic support accounting, cache policy, and explicit mode-isolated tests.

No human relevance judgments exist. The judgment CSVs are templates, so this audit makes no production-accuracy claim.
