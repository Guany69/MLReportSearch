"""Central configuration for the retrieval model.

Every knob the model exposes lives here so the CLI, the Streamlit app and demo.py
all read the same defaults.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path

# Repo root = three levels up from this file (src/reportfinder/config.py).
ROOT = Path(__file__).resolve().parents[2]

# Phase 1: single workbook, fields read from its `Fields` column.
DATA_PATH = ROOT / "data" / "Reports.xlsx"

# Phase 2: report catalog (no Fields) + supplemental field dictionary.
CATALOG_PATH = ROOT / "data" / "Phase2_Report_Catalog_No_Fields.xlsx"
FIELD_DICTIONARY_PATH = ROOT / "data" / "Phase2_Field_Dictionary.xlsx"

CACHE_DIR = ROOT / ".cache"

# Dense encoder: primary, then fallback if the primary can't be fetched/loaded.
DENSE_MODEL = "BAAI/bge-small-en-v1.5"
DENSE_MODEL_FALLBACK = "sentence-transformers/all-MiniLM-L6-v2"

# bge-* models are trained with an asymmetric retrieval prefix on the query side
# only. Applied to queries, never to documents.
BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "


@dataclass(frozen=True)
class Config:
    """Model + pipeline configuration.

    Attributes:
        t_dense: Softmax temperature for the dense expert. Lower => peakier posterior.
        t_lsa: Softmax temperature for the LSA expert.
        alpha: Geometric mixture weight. 1.0 = dense only, 0.0 = LSA only.
        tau: Minimum top-1 posterior probability to return a single confident answer.
        delta: Minimum margin (p1 - p2) to return a single confident answer.
        top_k: How many candidates to show when the posterior is ambiguous.
        use_field_expert: Toggle the optional third (field-term) expert.
        field_expert_weight: Exponent on P_field in the product of experts.
        ingest_mode: "legacy_single_file" or "phase2_dual_file". Both modes produce
            the same corpus contract; this only selects how it is assembled.
        ambiguity_policy: What to do when a Where_Used name matches several catalog
            rows. "permissive" attaches to all candidates (higher recall, may attach
            a field to a report that lacks it); "strict" withholds. Ambiguity is
            flagged and reported either way.
    """

    t_dense: float = 0.05
    t_lsa: float = 0.05
    alpha: float = 0.6

    tau: float = 0.15
    delta: float = 0.03
    top_k: int = 5

    use_field_expert: bool = False
    field_expert_weight: float = 0.25

    # Representation build
    svd_components: int = 200
    tfidf_min_df: int = 2
    tfidf_max_df: float = 0.6
    ngram_max: int = 2

    # Zone repetition for doc(r). These are representation choices (how many times
    # a zone's text is repeated into the document), not a hand-tuned score.
    w_title: int = 3
    w_fields: int = 3
    w_prompts: int = 2
    w_category: int = 1
    w_data_source: int = 1
    w_report_type: int = 1
    w_tags: int = 1

    # -- Phase 2 zones -------------------------------------------------
    # New signals default to weight 1 -- deliberately conservative, so they enrich
    # the representation without displacing the Phase 1 zones that ranking is
    # already tuned around. Legacy mode emits none of these, leaving doc(r)
    # byte-identical to Phase 1.
    #
    # Note: `Authorized Usage` is intentionally absent. It has exactly ONE distinct
    # value across all 3234 dictionary rows, so it carries zero discriminating
    # signal -- the same trap as Phase 1's `Description` (7 distinct values). It is
    # kept as displayable metadata but never enters doc(r) or scoring.
    w_field_description: int = 1
    w_business_object: int = 1
    w_domain: int = 1
    w_field_categories: int = 1
    w_field_type: int = 0  # 9 values across 3234 rows; weak, off by default
    w_builtin_prompts: int = 1
    w_related_business_object: int = 1

    # -- ingestion -----------------------------------------------------
    ingest_mode: str = "legacy_single_file"
    ambiguity_policy: str = "permissive"
    enable_composite_match: bool = True
    enable_fuzzy_match: bool = False
    fuzzy_threshold: float = 0.93

    dense_model: str = DENSE_MODEL
    data_path: Path = DATA_PATH
    catalog_path: Path = CATALOG_PATH
    field_dictionary_path: Path = FIELD_DICTIONARY_PATH
    cache_dir: Path = CACHE_DIR

    def with_overrides(self, **kwargs) -> "Config":
        """Return a copy with non-None overrides applied."""
        clean = {k: v for k, v in kwargs.items() if v is not None}
        return replace(self, **clean)


DEFAULT = Config()
