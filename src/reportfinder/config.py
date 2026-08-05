"""Central configuration for the retrieval model.

Every knob the model exposes lives here so the CLI, the Streamlit app and demo.py
all read the same defaults.
"""

from __future__ import annotations

from dataclasses import dataclass, fields, is_dataclass, replace
from difflib import get_close_matches
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

# Pinned checkpoint revisions. Resolved from the Hub on 2026-08-04 and recorded so a
# rebuild cannot silently pick up new weights: a checkpoint change alters every
# vector in the index, and an unpinned one makes that change invisible.
DENSE_MODEL_REVISION = "5c38ec7c405ec4b44b94cc5a9bb96e735b38267a"
DENSE_CHALLENGER_MODEL = "intfloat/e5-small-v2"
DENSE_CHALLENGER_REVISION = "ffb93f3bd4047442299a41ebb6fa998a38507c52"
CROSS_ENCODER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
CROSS_ENCODER_REVISION = "c5ee24cb16019beea0893ab7796b1df96625c6b8"
SPLADE_MODEL = "naver/splade-cocondenser-ensembledistil"
SPLADE_REVISION = "49cf4c7b0db5b870a401ddf5e2669993ef3699c7"
LATE_INTERACTION_MODEL = "colbert-ir/colbertv2.0"
LATE_INTERACTION_REVISION = "c1e84128e85ef755c096a95bdb06b47793b13acf"

BUNDLE_DIR = ROOT / ".bundle"

# The deterministic decision policy's two thresholds, named here so `decide()` and
# `RiskConfig` cannot drift apart. They previously differed: `decide()`'s no-config
# fallback used -5.0 while the config default was -9.0, so a caller passing
# cfg=None silently ran a different policy from the documented one.
DEFAULT_WEAK_RERANK_LOGIT = -9.0
DEFAULT_CLARIFY_MARGIN = 1.0

# Concurrent generator workers. Same reason: `SearchPipeline` read this with a
# getattr default of 1, so it disagreed with the config default of 6.
DEFAULT_GENERATOR_WORKERS = 6

# bge-* models are trained with an asymmetric retrieval prefix on the query side
# only. Applied to queries, never to documents.
BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "

# The original deterministic ranker is retained as an explicit ablation preset.
# The improved preset changes only inspectable coefficients; no learned artifact
# is implied by either choice.
LEGACY_RANKING_WEIGHTS = (
    ("mandatory_coverage", 3.00), ("exact_field_norm", 2.20),
    ("optional_concept_coverage", 1.40), ("intent_coverage", .80),
    ("title_token_jaccard", 1.10), ("dense_norm", .90), ("bm25_norm", .85),
    ("rrf_norm", .70), ("retriever_agreement_strength", .45),
    ("retriever_agreement_count_n", .25), ("subquery_coverage", .40),
    ("subquery_count_satisfied", .20), ("description_token_jaccard", .30),
    ("prompt_compatibility", .30), ("filter_compatibility", .25),
    ("lsa_norm", .25), ("char_norm", .20), ("time_match", .20),
    ("granularity_match", .20), ("usage_prior_n", .08),
    ("freshness_prior", .05), ("shared_status_prior", .03),
    ("field_link_confidence", .15),
)
IMPROVED_RANKING_WEIGHTS = (
    ("mandatory_coverage", 3.40), ("exact_field_norm", 2.40),
    ("optional_concept_coverage", 1.55), ("intent_coverage", 1.00),
    ("title_token_jaccard", 3.00), ("interpreted_title_coverage", 4.00),
    ("prompt_compatibility", .45),
    ("filter_compatibility", .35), ("granularity_match", .30),
    ("bm25_norm", .80), ("dense_norm", .70), ("rrf_norm", .65),
    ("retriever_agreement_strength", .40),
    ("retriever_agreement_count_n", .20), ("subquery_coverage", .35),
    ("subquery_count_satisfied", .15), ("description_token_jaccard", .25),
    ("lsa_norm", .20), ("char_norm", .18), ("time_match", .20),
    ("usage_prior_n", .05), ("freshness_prior", .04),
    ("shared_status_prior", .02), ("field_link_confidence", .15),
)
RANKING_BONUSES = (
    ("category_match", .30), ("data_source_match", .25),
    ("business_object_match", .20),
)
RANKING_PENALTIES = (("ambiguous_link_fraction", .35),)


# --- generator-architecture configuration -----------------------------------
#
# These are nested rather than flat because there are ~70 of them and prefix-encoded
# names (`dense_retriever_dense_k_per_view`) stop being readable well before that.
# Each object validates itself; `Config.__post_init__` only checks invariants that
# span more than one object. YAML nests to match; flat legacy keys still work.
#
# Every default here is a *starting* value, not a measured winner. Depths, quotas
# and thresholds are stated so they can be ablated, and anything that changes
# tokenization, indexing, candidate depth, features or decisions is versioned into
# the bundle manifest.


@dataclass(frozen=True)
class Bm25fConfig:
    """Field-aware lexical generator. Independent -- never a gate on the others."""

    enabled: bool = True
    k: int = 100

    def __post_init__(self) -> None:
        if self.k < 1:
            raise ValueError("retrieval.bm25f.k must be at least 1")


@dataclass(frozen=True)
class SpladeConfig:
    """Learned sparse generator (term expansion in vocabulary space)."""

    enabled: bool = True
    checkpoint: str = SPLADE_MODEL
    revision: str = SPLADE_REVISION
    k: int = 100
    max_length: int = 256
    batch_size: int = 32
    # Terms below this weight contribute nothing but cost; pruning keeps the
    # postings list small. Applied at index build and recorded in the manifest.
    min_term_weight: float = 0.01

    def __post_init__(self) -> None:
        if self.k < 1:
            raise ValueError("retrieval.splade.k must be at least 1")
        if self.min_term_weight < 0:
            raise ValueError("retrieval.splade.min_term_weight must be non-negative")


@dataclass(frozen=True)
class DenseRetrieverConfig:
    """Multi-view bi-encoder retrieval. Each view is searched independently."""

    checkpoint: str = DENSE_MODEL
    revision: str = DENSE_MODEL_REVISION
    challenger_checkpoint: str = DENSE_CHALLENGER_MODEL
    challenger_revision: str = DENSE_CHALLENGER_REVISION
    views: tuple[str, ...] = ("identity", "purpose", "schema", "interface")
    k_per_view: int = 50
    max_length: int = 512
    batch_size: int = 64
    # A view whose text is near-constant returns the same arbitrary top-k for every
    # query. Below this ratio of distinct non-empty texts the index build records a
    # degenerate status and the generator is not constructed.
    min_distinct_text_ratio: float = 0.10

    def __post_init__(self) -> None:
        # YAML yields lists; the field is a tuple so the config stays frozen and
        # hashable. Coerced here so the type does not depend on where it came from.
        object.__setattr__(self, "views", tuple(self.views))
        if self.k_per_view < 1:
            raise ValueError("retrieval.dense.k_per_view must be at least 1")
        if not self.views:
            raise ValueError("retrieval.dense.views must not be empty")
        unknown = set(self.views) - {"identity", "purpose", "schema", "interface"}
        if unknown:
            raise ValueError(f"unknown dense views: {sorted(unknown)}")
        if not 0.0 <= self.min_distinct_text_ratio <= 1.0:
            raise ValueError("retrieval.dense.min_distinct_text_ratio must be in [0, 1]")


@dataclass(frozen=True)
class PrototypeConfig:
    """Family query-prototype retrieval: user-language examples per family."""

    enabled: bool = True
    k: int = 50

    def __post_init__(self) -> None:
        if self.k < 1:
            raise ValueError("retrieval.prototypes.k must be at least 1")


@dataclass(frozen=True)
class LateInteractionConfig:
    """Token-level fallback for high-recall-risk queries.

    Ships disabled: the module, loader and tests are real, but no approved artifact
    has been built or measured on this estate. `disabled_reason` is surfaced in
    telemetry so the state is visible rather than assumed.
    """

    enabled: bool = False
    checkpoint: str = LATE_INTERACTION_MODEL
    revision: str = LATE_INTERACTION_REVISION
    k: int = 50
    max_length: int = 256
    batch_size: int = 16
    doc_dim: int = 128
    trigger_on_high_risk: bool = True
    disabled_reason: str = "shipped disabled pending latency and quality evidence"

    def __post_init__(self) -> None:
        if self.k < 1:
            raise ValueError("retrieval.late_interaction.k must be at least 1")


@dataclass(frozen=True)
class RetrievalConfig:
    """The independent candidate generators and how their ranks are fused."""

    bm25f: Bm25fConfig = None  # type: ignore[assignment]
    splade: SpladeConfig = None  # type: ignore[assignment]
    dense: DenseRetrieverConfig = None  # type: ignore[assignment]
    prototypes: PrototypeConfig = None  # type: ignore[assignment]
    late_interaction: LateInteractionConfig = None  # type: ignore[assignment]
    rrf_constant: int = 60
    # Generators share no mutable state and torch releases the GIL during
    # inference, so they can run concurrently. Results are still collected in
    # submission order, so this changes latency and nothing else. 1 disables it.
    max_generator_workers: int = DEFAULT_GENERATOR_WORKERS

    def __post_init__(self) -> None:
        # Defaults are filled here rather than via default_factory so that a YAML
        # mapping can override one nested object without restating the others.
        for name, factory in (
            ("bm25f", Bm25fConfig), ("splade", SpladeConfig),
            ("dense", DenseRetrieverConfig), ("prototypes", PrototypeConfig),
            ("late_interaction", LateInteractionConfig),
        ):
            if getattr(self, name) is None:
                object.__setattr__(self, name, factory())
        if self.rrf_constant < 1:
            raise ValueError("retrieval.rrf_constant must be at least 1")
        if self.max_generator_workers < 1:
            raise ValueError("retrieval.max_generator_workers must be at least 1")


@dataclass(frozen=True)
class RiskConfig:
    """Recall risk -- how likely the right report is to be missed.

    Deliberately separate from answer confidence. A vague query is a reason to
    search *wider*, not to abstain; abstention is decided later, on evidence.
    """

    short_query_tokens: int = 3
    min_generator_overlap: float = 0.30
    high_source_exclusive_ratio: float = 0.50
    many_families: int = 25
    # Entropy of the fused scores across the contending head, not the whole union:
    # a long RRF tail drives whole-union entropy to ~1 for every query.
    high_entropy: float = 0.98
    # Relative margin ((top - second) / top), so the threshold does not depend on
    # the number of generators or on the RRF constant.
    low_margin: float = 0.05
    min_schema_facet_coverage: float = 0.25
    # How many HIGH-risk signals must fire before the expanded shortlist is used.
    medium_signal_threshold: int = 1
    high_signal_threshold: int = 3

    # -- deterministic decision-policy thresholds ----------------------
    # Used only while no trained decision head exists. Both are on the
    # cross-encoder's logit scale, which is the only score in the pipeline whose
    # magnitude means "how good a match is this" rather than "where did it land".
    #
    # ms-marco cross-encoders emit strongly negative logits for unrelated pairs,
    # so a best-candidate score below this is evidence that nothing in the catalog
    # answers the request.
    #
    # Set by inspecting six probe queries on the real estate: genuinely unanswerable
    # requests bottomed out around -11.4, while a misspelled query whose correct
    # answer *was* retrieved peaked at -7.1. -9.0 sits between them. Six queries is
    # an observation, not a validation -- this needs human no-answer labels before
    # it can be called calibrated.
    weak_rerank_logit: float = DEFAULT_WEAK_RERANK_LOGIT
    # How far ahead the top family must be before results are returned without a
    # clarifying question. Unmeasured: it needs human decision labels to set.
    clarify_margin: float = DEFAULT_CLARIFY_MARGIN

    def __post_init__(self) -> None:
        if self.high_signal_threshold < self.medium_signal_threshold:
            raise ValueError("risk.high_signal_threshold must be >= medium_signal_threshold")


# Quota floors for the source-preserving shortlist. These are *floors*, not a
# partition: whatever they do not consume is filled from the fused ordering, so a
# generator that finds nothing exclusive never shrinks the shortlist.
DEFAULT_SHORTLIST_QUOTAS = (
    ("fused", 50),
    ("bm25_exclusive", 15),
    ("splade_exclusive", 15),
    ("schema_exclusive", 10),
    ("purpose_exclusive", 10),
    ("prototype_exclusive", 10),
    ("alternate_query_exclusive", 10),
)


@dataclass(frozen=True)
class ShortlistConfig:
    """How the union is reduced to what the cross-encoder scores.

    Not by one global weighted score: that is exactly the single-gate failure the
    architecture exists to remove. Quotas reserve room for candidates only one
    generator found.
    """

    standard_rerank_depth: int = 120
    high_risk_rerank_depth: int = 200
    preserve_source_exclusive_candidates: bool = True
    quotas: tuple[tuple[str, int], ...] = DEFAULT_SHORTLIST_QUOTAS
    # HIGH risk widens the exclusive floors by this factor.
    high_risk_quota_scale: float = 1.5
    # Spend the *fused remainder* on families not yet represented before spending
    # it on a second copy of one already in the shortlist. Source-exclusive floors
    # are untouched either way.
    #
    # Implemented, switchable, and **off by default because it was measured and it
    # lost.** Isolated A/B on the same code over 265 v2 validation queries,
    # comparing the deterministic serving arm:
    #
    #     metric                            on        off      delta
    #     family_ndcg@5                     0.1026    0.1064   -0.0038
    #     family_mrr                        0.1143    0.1194   -0.0051
    #     family_recall@10                  0.1679    0.1755   -0.0076
    #     instance_recall@1_within_family   0.7750    0.8684   -0.0934
    #
    # The mechanism is the last row. Deferring a sibling out of the shortlist
    # makes the choice of *which copy* to return depend on family expansion, which
    # is capped by max_expanded_instances -- so the sibling that would have won
    # sometimes never reaches the cross-encoder at all. The hoped-for family-level
    # gain did not appear either: at depth 120 the plain fused remainder already
    # reaches enough distinct families that reordering it only costs.
    #
    # Kept rather than deleted because the reasoning behind it is sound for a
    # narrower shortlist or an uncapped expansion, and a switch with a recorded
    # measurement is worth more than a deleted branch. Turn it on with
    # `shortlist.diversify_families: true` and re-measure before trusting it.
    diversify_families: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "quotas", tuple((str(n), int(v)) for n, v in self.quotas)
        )
        if self.standard_rerank_depth < 1:
            raise ValueError("shortlist.standard_rerank_depth must be at least 1")
        if self.high_risk_rerank_depth < self.standard_rerank_depth:
            raise ValueError(
                "shortlist.high_risk_rerank_depth must be >= standard_rerank_depth: "
                "a risky query must never receive a narrower pool"
            )
        if any(count < 0 for _, count in self.quotas):
            raise ValueError("shortlist quotas must be non-negative")
        total = sum(count for name, count in self.quotas if name != "fused")
        if total > self.standard_rerank_depth:
            raise ValueError(
                f"exclusive shortlist quotas ({total}) exceed standard_rerank_depth "
                f"({self.standard_rerank_depth})"
            )

    @property
    def quota_map(self) -> dict[str, int]:
        return dict(self.quotas)

    def depth_for(self, risk_band: str) -> int:
        return (
            self.high_risk_rerank_depth
            if risk_band == "HIGH"
            else self.standard_rerank_depth
        )


@dataclass(frozen=True)
class RerankConfig:
    """Cross-encoder scoring of the full shortlist."""

    enabled: bool = True
    checkpoint: str = CROSS_ENCODER_MODEL
    revision: str = CROSS_ENCODER_REVISION
    max_length: int = 256
    batch_size: int = 32
    max_report_chars: int = 1200
    truncation: str = "longest_first"
    # "auto" probes float32 once at load and escalates to float64 if this machine
    # cannot be trusted with it. See TorchCrossEncoder: torch 2.11 on arm64
    # macOS 13 returns NaN in float32 for any realistic sequence length, and a
    # silent NaN is far worse than a slower correct answer.
    dtype: str = "auto"
    # Which runtime executes the cross-encoder. `torch` is the default and stays
    # so until the ONNX parity gate passes on a given host: a faster backend that
    # ranks differently is not a faster backend, it is a different system.
    backend: str = "torch"
    onnx_path: Path | None = None

    def __post_init__(self) -> None:
        if self.dtype not in {"auto", "float32", "float64"}:
            raise ValueError("rerank.dtype must be 'auto', 'float32' or 'float64'")
        if self.backend not in {"torch", "onnx"}:
            raise ValueError("rerank.backend must be 'torch' or 'onnx'")
        if self.onnx_path is not None:
            object.__setattr__(self, "onnx_path", Path(self.onnx_path))
        if self.max_length < 16:
            raise ValueError("rerank.max_length must be at least 16")
        if self.truncation not in {"longest_first", "only_second"}:
            raise ValueError("rerank.truncation must be 'longest_first' or 'only_second'")


@dataclass(frozen=True)
class FusionConfig:
    """Listwise neural fusion, with RRF as the declared fallback.

    `artifact_path=None` is the shipped state: no trained fusion model exists, so
    the pipeline uses RRF and says so in telemetry. Random weights are never
    presented as trained.
    """

    architecture: str = "pytorch_mlp_64_32_1"
    artifact_path: Path | None = None
    fallback: str = "rrf"
    hidden_1: int = 64
    hidden_2: int = 32
    dropout: float = 0.1

    def __post_init__(self) -> None:
        if self.fallback != "rrf":
            raise ValueError("fusion.fallback must be 'rrf'")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("fusion.dropout must be in [0, 1)")


@dataclass(frozen=True)
class DecisionConfig:
    """Three-way decision head, with a deterministic policy as the fallback."""

    architecture: str = "pytorch_mlp_32_16_3"
    artifact_path: Path | None = None
    calibration_path: Path | None = None
    fallback: str = "deterministic_three_way_policy"
    hidden_1: int = 32
    hidden_2: int = 16

    def __post_init__(self) -> None:
        if self.fallback != "deterministic_three_way_policy":
            raise ValueError("decision.fallback must be 'deterministic_three_way_policy'")
        if self.calibration_path is not None and self.artifact_path is None:
            raise ValueError(
                "decision.calibration_path without decision.artifact_path: there is "
                "nothing to calibrate"
            )


@dataclass(frozen=True)
class AggregationConfig:
    """Family-first retrieval: expand a leading family's other instances.

    Retrieval nominates *instances*, but 533 titles on this estate exist on more
    than one row with different field sets, prompts or data sources. Aggregating
    only the instances that happened to survive the shortlist means the copy a user
    should actually run can be invisible simply because a sibling ranked higher.

    So once the leading families are known, every authorized instance inside them is
    pulled in and scored, whether or not any retriever found it.

    `max_expanded_instances` is a real cost bound, not a tidiness knob: an expanded
    instance carries no retrieval evidence, so it can only be judged by a full
    cross-encoder pass, and this machine runs that in float64.
    """

    enabled: bool = True
    family_expansion_k: int = 20
    max_expanded_instances: int = 60

    def __post_init__(self) -> None:
        if self.family_expansion_k < 1:
            raise ValueError("aggregation.family_expansion_k must be at least 1")
        if self.max_expanded_instances < 0:
            raise ValueError("aggregation.max_expanded_instances must be non-negative")
        if self.enabled and self.max_expanded_instances == 0:
            raise ValueError(
                "aggregation.enabled=true with max_expanded_instances=0 would expand "
                "nothing; set enabled=false to turn family expansion off explicitly"
            )


@dataclass(frozen=True)
class AuthConfig:
    """Entitlement resolution. Applied before any candidate is exposed.

    `allow_all` is a development default for an estate with no ACL source. It must
    be opted into explicitly, and it is disclosed in every response and trace, so
    a deployment cannot inherit an open door by accident.
    """

    resolver: str = "allow_all"
    allow_dev_default: bool = True
    fail_closed_on_acl_error: bool = True

    def __post_init__(self) -> None:
        if self.resolver not in {"allow_all", "explicit_acl"}:
            raise ValueError("auth.resolver must be 'allow_all' or 'explicit_acl'")
        if self.resolver == "allow_all" and not self.allow_dev_default:
            raise ValueError(
                "auth.resolver='allow_all' requires auth.allow_dev_default=true; "
                "refusing to search the whole estate without an explicit opt-in"
            )


@dataclass(frozen=True)
class SecurityConfig:
    trust_remote_code: bool = False


@dataclass(frozen=True)
class BundleConfig:
    """Where index components live and which of them block readiness."""

    root: Path = BUNDLE_DIR
    required: tuple[str, ...] = (
        "views.identity", "views.schema", "splade", "cross_encoder",
    )

    def __post_init__(self) -> None:
        object.__setattr__(self, "root", Path(self.root))
        object.__setattr__(self, "required", tuple(self.required))


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

    # Retrieval/ranking architecture, in order of age:
    #
    #   legacy_weighted_logit -- the original pre-overhaul behavior.
    #   hybrid                -- candidate retrieval + RRF + a deterministic ranker.
    #                            Deprecated: retained for ablation and for the tests
    #                            that pin its documented behavior, not for serving.
    #   generators            -- the default. Independent authorized candidate
    #                            generators, source-preserving union and shortlist,
    #                            cross-encoder reranking over the full shortlist,
    #                            family-first expansion, three-way decision.
    #
    # `generators` is the default because the alternative is a split-brain product:
    # a documented architecture nobody's entrypoint actually runs. It requires
    # corpus_granularity='report_row' -- family aggregation needs distinct instances.
    retrieval_mode: str = "generators"
    corpus_granularity: str = "report_row"
    fusion_method: str = "rrf"
    candidate_k: int = 100
    rrf_k: int = 60
    use_bm25f: bool = True
    use_dense: bool = True
    use_lsa: bool = True
    use_exact_field: bool = True
    use_char_ngrams: bool = True
    char_ngram_min: int = 3
    char_ngram_max: int = 5
    use_ltr: bool = False
    ltr_model_path: Path | None = None
    use_query_expansion: bool = True
    expansion_weight: float = 0.85
    subquery_weight: float = 0.75
    expansion_aggregate_lambda: float = 0.30
    expansion_min_confidence: float = 0.70
    use_typo_correction: bool = True
    typo_max_distance: int = 2
    max_typo_corrections: int = 2
    max_mandatory_fields: int = 4
    dense_mode: str = "auto"
    use_cross_encoder: bool = False
    cross_encoder_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    cross_encoder_top_k: int = 20
    cross_encoder_weight: float = 0.15
    share_temperature: float = 1.0
    answerability_model_path: Path | None = None
    use_calibrated_answerability: bool = False
    answerability_high_threshold: float = 0.72
    answerability_insufficient_threshold: float = 0.35
    min_support_ratio: float = 0.50
    hard_support_floor: float = 0.20
    breadth_threshold: float = 0.60
    answerability_min_margin: float = 0.06
    answerability_min_high_support: float = 0.70
    answerability_min_field_coverage: float = 0.80
    answerability_intercept: float = -1.60
    answerability_top_score_weight: float = 1.70
    answerability_margin_weight: float = 1.60
    answerability_margin_scale: float = 0.15
    answerability_field_coverage_weight: float = 1.30
    answerability_retriever_agreement_weight: float = 0.80
    answerability_parser_confidence_weight: float = 0.50
    answerability_support_weight: float = 1.10
    answerability_ambiguity_weight: float = 1.20
    # Deterministic ranking controls. ``legacy`` reproduces the original values;
    # ``improved`` follows the documented field/intent-first priority order.
    ranking_preset: str = "improved"
    ranking_gate_floor: float = 0.15
    ranking_gate_exponent: float = 1.5
    ranking_hard_violation_penalty: float = 10.0
    ranking_legacy_weights: tuple[tuple[str, float], ...] = LEGACY_RANKING_WEIGHTS
    ranking_improved_weights: tuple[tuple[str, float], ...] = IMPROVED_RANKING_WEIGHTS
    ranking_bonuses: tuple[tuple[str, float], ...] = RANKING_BONUSES
    ranking_penalties: tuple[tuple[str, float], ...] = RANKING_PENALTIES

    # BM25F zone weights.
    bm25_title: float = 4.0
    bm25_description: float = 1.6
    bm25_fields: float = 3.5
    bm25_field_descriptions: float = 1.4
    bm25_prompts: float = 1.8
    bm25_business_objects: float = 1.4
    bm25_category: float = 1.0
    bm25_tags: float = 1.0
    bm25_data_source: float = 1.2
    bm25_area_where_used: float = 1.4

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

    # -- generator architecture ----------------------------------------
    # Appended last so `Result`/`Config` positional construction elsewhere is
    # unaffected. Defaults are filled in __post_init__ rather than by
    # default_factory so a YAML mapping can override one nested object without
    # having to restate its siblings.
    retrieval: RetrievalConfig = None  # type: ignore[assignment]
    risk: RiskConfig = None  # type: ignore[assignment]
    shortlist: ShortlistConfig = None  # type: ignore[assignment]
    rerank: RerankConfig = None  # type: ignore[assignment]
    fusion: FusionConfig = None  # type: ignore[assignment]
    decision: DecisionConfig = None  # type: ignore[assignment]
    aggregation: AggregationConfig = None  # type: ignore[assignment]
    auth: AuthConfig = None  # type: ignore[assignment]
    security: SecurityConfig = None  # type: ignore[assignment]
    bundle: BundleConfig = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        for name, factory in (
            ("retrieval", RetrievalConfig), ("risk", RiskConfig),
            ("shortlist", ShortlistConfig), ("rerank", RerankConfig),
            ("fusion", FusionConfig), ("decision", DecisionConfig),
            ("aggregation", AggregationConfig),
            ("auth", AuthConfig), ("security", SecurityConfig),
            ("bundle", BundleConfig),
        ):
            if getattr(self, name) is None:
                object.__setattr__(self, name, factory())

        # YAML has no Path type, so a config file supplies these as strings while
        # the annotation and every consumer expect a Path. Coerced here rather
        # than at each use site, where one missed `.name` or `.exists()` is an
        # AttributeError on a path that is perfectly valid.
        for name in ("data_path", "catalog_path", "field_dictionary_path", "cache_dir"):
            value = getattr(self, name, None)
            if isinstance(value, str):
                object.__setattr__(self, name, Path(value))

        if self.retrieval_mode not in {"hybrid", "legacy_weighted_logit", "generators"}:
            raise ValueError(f"Unsupported retrieval_mode: {self.retrieval_mode}")
        if self.retrieval_mode == "generators" and self.corpus_granularity != "report_row":
            raise ValueError(
                "retrieval_mode='generators' requires corpus_granularity='report_row': "
                "family aggregation needs distinct instances to aggregate"
            )
        if self.corpus_granularity not in {"report_row", "family"}:
            raise ValueError(f"Unsupported corpus_granularity: {self.corpus_granularity}")
        if self.dense_mode not in {"auto", "local", "off"}:
            raise ValueError(f"Unsupported dense_mode: {self.dense_mode}")
        if self.dense_mode == "local" and not self.use_dense:
            raise ValueError("dense_mode='local' requires use_dense=True")
        if not 0.0 <= self.expansion_weight <= 1.0:
            raise ValueError("expansion_weight must be in [0, 1]")
        if not 0.0 <= self.subquery_weight <= 1.0:
            raise ValueError("subquery_weight must be in [0, 1]")
        if not 0.0 <= self.expansion_aggregate_lambda <= 1.0:
            raise ValueError("expansion_aggregate_lambda must be in [0, 1]")
        if self.use_typo_correction and self.typo_max_distance < 1:
            raise ValueError("typo_max_distance must be at least 1 when typo correction is enabled")
        if self.max_typo_corrections < 0:
            raise ValueError("max_typo_corrections must be non-negative")
        if self.max_mandatory_fields < 1:
            raise ValueError("max_mandatory_fields must be at least 1")
        if self.use_cross_encoder and self.cross_encoder_top_k < 1:
            raise ValueError("cross_encoder_top_k must be at least 1")
        if self.cross_encoder_weight < 0:
            raise ValueError("cross_encoder_weight must be non-negative")
        if self.share_temperature <= 0:
            raise ValueError("share_temperature must be positive")
        for name in (
            "answerability_high_threshold", "answerability_insufficient_threshold",
            "min_support_ratio", "hard_support_floor", "breadth_threshold",
            "answerability_min_margin", "answerability_min_high_support",
            "answerability_min_field_coverage",
        ):
            if not 0.0 <= getattr(self, name) <= 1.0:
                raise ValueError(f"{name} must be in [0, 1]")
        if self.answerability_margin_scale <= 0:
            raise ValueError("answerability_margin_scale must be positive")
        if self.ranking_preset not in {"legacy", "improved"}:
            raise ValueError("ranking_preset must be 'legacy' or 'improved'")
        if not 0.0 <= self.ranking_gate_floor <= 1.0:
            raise ValueError("ranking_gate_floor must be in [0, 1]")
        if self.ranking_gate_exponent <= 0:
            raise ValueError("ranking_gate_exponent must be positive")
        if self.ranking_hard_violation_penalty < 0:
            raise ValueError("ranking_hard_violation_penalty must be non-negative")

        # -- invariants spanning more than one nested object ---------------
        for band in ("LOW", "MEDIUM", "HIGH"):
            depth = self.shortlist.depth_for(band)
            if depth < 1:
                raise ValueError(f"shortlist depth for {band} must be at least 1")
        if self.retrieval.late_interaction.enabled and (
            "late_interaction" not in self.bundle.required
        ):
            raise ValueError(
                "retrieval.late_interaction.enabled=true requires 'late_interaction' "
                "in bundle.required, or a query could silently run without it"
            )
        if self.fusion.artifact_path is None and self.fusion.fallback != "rrf":
            raise ValueError("no fusion artifact and no usable fallback")

    def with_overrides(self, **kwargs) -> Config:
        """Return a copy with non-None overrides applied."""
        clean = {k: v for k, v in kwargs.items() if v is not None}
        return replace(self, **clean)

    def with_path_overrides(self, overrides: dict[str, object]) -> Config:
        """Apply dotted-path overrides, e.g. {"retrieval.dense.k_per_view": 80}.

        Every override is collected and the Config is rebuilt **once**. Applying
        them one at a time would re-run validation on each intermediate state and
        reject a valid end state -- raising rerank depth before raising the
        shortlist size, for instance.
        """
        nested: dict[str, dict] = {}
        flat: dict[str, object] = {}
        for key, value in overrides.items():
            head, _, rest = key.partition(".")
            if rest:
                _nest(nested.setdefault(head, {}), rest, value)
            else:
                flat[head] = value

        rebuilt = {
            name: _rebuild(getattr(self, name), patch, name)
            for name, patch in nested.items()
        }
        return replace(self, **flat, **rebuilt)


def _nest(target: dict, dotted: str, value: object) -> None:
    head, _, rest = dotted.partition(".")
    if rest:
        _nest(target.setdefault(head, {}), rest, value)
    else:
        target[head] = value


def _is_config_object(value: object) -> bool:
    return is_dataclass(value) and not isinstance(value, type)


def _unknown_key(key: str, known: set[str], where: str) -> ValueError:
    """Reject an unknown key, suggesting near matches rather than dumping all of them."""
    close = get_close_matches(key, sorted(known), n=3, cutoff=0.6)
    hint = f" Did you mean: {', '.join(close)}?" if close else ""
    return ValueError(f"unknown configuration key {key!r} in {where}.{hint}")


def _merge_quotas(current: tuple[tuple[str, int], ...], patch: dict) -> tuple:
    """Merge a quota mapping into the configured floors.

    Merged rather than replaced on purpose: a file that sets only `fused` would
    otherwise silently drop every source-exclusive quota, which removes the
    guarantee that a candidate only one generator found still reaches the
    cross-encoder. Dropping a quota has to be deliberate -- set it to 0.
    """
    known = {name for name, _ in current}
    for name in patch:
        if name not in known:
            raise _unknown_key(name, known, "shortlist.quotas")
    merged = dict(current)
    merged.update({name: int(value) for name, value in patch.items()})
    # Preserve the declared order so the shortlist fills quotas deterministically.
    return tuple((name, merged[name]) for name, _ in current)


def _rebuild(current: object, patch: dict, path: str = "") -> object:
    """Apply a nested mapping to a nested config object."""
    if not _is_config_object(current):
        raise ValueError(f"{path or current!r} is not a nested configuration object")
    known = {f.name for f in fields(current)}
    where = path or type(current).__name__
    updates: dict[str, object] = {}
    for key, value in patch.items():
        if key not in known:
            raise _unknown_key(key, known, where)
        existing = getattr(current, key)
        if key == "quotas" and isinstance(value, dict):
            # YAML expresses quotas as a mapping; the config stores ordered pairs
            # so it stays frozen and hashable.
            updates[key] = _merge_quotas(existing, value)
        elif isinstance(value, dict) and _is_config_object(existing):
            updates[key] = _rebuild(existing, value, f"{where}.{key}")
        else:
            updates[key] = value
    return replace(current, **updates)


def resolve_config_path(cfg: Config, dotted: str) -> tuple[object, str, object]:
    """Walk a dotted path, returning (owner, attribute, current value).

    Used by `--set a.b.c=1` so the CLI can validate a key and infer the target
    type before parsing the value.
    """
    owner: object = cfg
    parts = dotted.split(".")
    for part in parts[:-1]:
        if not hasattr(owner, part):
            raise ValueError(f"unknown configuration path: {dotted}")
        owner = getattr(owner, part)
    leaf = parts[-1]
    if not hasattr(owner, leaf):
        raise ValueError(f"unknown configuration path: {dotted}")
    return owner, leaf, getattr(owner, leaf)


def from_mapping(raw: dict, base: Config | None = None) -> Config:
    """Build a Config from a YAML mapping.

    A dict value whose key names a nested config object recurses into it; anything
    else is a flat override. Existing flat-only config files therefore load
    unchanged.
    """
    base = DEFAULT if base is None else base
    if not raw:
        return base

    known = {f.name for f in fields(base)}
    overrides: dict[str, object] = {}
    nested: dict[str, dict] = {}
    for key, value in raw.items():
        if key not in known:
            raise _unknown_key(key, known, "config root")
        if isinstance(value, dict) and _is_config_object(getattr(base, key)):
            nested[key] = value
        else:
            overrides[key] = value

    rebuilt = {
        name: _rebuild(getattr(base, name), patch, name)
        for name, patch in nested.items()
    }
    return replace(base, **overrides, **rebuilt)


DEFAULT = Config()
