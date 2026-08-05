"""Multi-channel retrieval, structured ranking, and answerability.

Two independent experts each induce a distribution over the corpus, and the
posterior is their geometric mixture:

    P_dense(r|q) = softmax(s_dense / T_d)
    P_lsa(r|q)   = softmax(s_lsa   / T_l)
    P(r|q)      ∝ P_dense(r|q)^α · P_lsa(r|q)^(1-α)

Algebraically, the legacy geometric mixture is a weighted sum of temperature-
scaled logits followed by softmax; the expert normalizers cancel. It remains as
``legacy_weighted_logit`` for reproducibility. The default hybrid path uses
BM25F, dense, LSA, exact-field and character retrieval, RRF candidate fusion,
structured ranking, and a separate answerability decision.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

import numpy as np
from scipy.special import logsumexp

from .confidence.answerability import (
    AnswerabilityDecision,
    AnswerabilityOutcome,
    decide_answerability,
    intent_ambiguity,
)
from .confidence.support import compute_support
from .config import Config
from .explain import ResultExplanation, build_result_explanation
from .query.intent import IntentParser, QueryIntent
from .ranking.features import RankingFeatures, extract_features, linear_score
from .ranking.ltr import Ranker
from .represent import Representation, encode_query_text
from .retrieval import (
    BM25FIndex,
    CharNgramIndex,
    FieldIndex,
    aggregate_top_two,
    reciprocal_rank_fusion,
    score_forms,
)

# Tokens too generic to be evidence of anything in a report corpus.
_STOPWORDS = frozenset(
    ["a", "an", "the", "of", "for", "by", "with", "to", "in", "on", "and", "or", "is", "are", "was", "were", "be", "been", "show", "me", "list", "all", "get", "find", "give", "please", "report", "reports", "i", "want", "need", "which", "what", "who", "how", "many", "that", "this", "these", "those", "from", "at", "as", "it", "its", "any", "some"]
)

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def _tokens(text: str) -> list[str]:
    return [t for t in _TOKEN_RE.findall(text.casefold()) if t not in _STOPWORDS]


@dataclass
class ExpertTrace:
    """Per-expert diagnostics for a single candidate report."""

    log_prob_dense: float
    log_prob_lsa: float
    sim_dense: float
    sim_lsa: float
    lift_dense: float  # α·(log P_dense(r) + log n)  -- evidence above uniform, in nats
    lift_lsa: float  # (1-α)·(log P_lsa(r) + log n)
    log_prob_field: float | None = None

    @property
    def dense_share(self) -> float | None:
        """Fraction of the above-uniform evidence contributed by the dense expert.

        Only meaningful when both experts actually favour the candidate; if either
        lift is negative the ratio isn't a share of anything, so we return None and
        the caller reports raw lifts instead.
        """
        if self.lift_dense <= 0 or self.lift_lsa <= 0:
            return None
        return self.lift_dense / (self.lift_dense + self.lift_lsa)


@dataclass
class FieldMatch:
    """A Phase 2 field that contributed to a report surfacing."""

    field_name: str
    business_object: str
    description: str
    exact: bool  # matched the field NAME literally vs only its description/metadata
    ambiguous: bool  # the report<-field link itself could not be uniquely determined
    match_method: str


@dataclass
class Candidate:
    """One retrieved report family with its posterior probability and evidence."""

    index: int
    probability: float
    row: object  # pandas Series for the report family
    trace: ExpertTrace
    matched_terms: list[str] = field(default_factory=list)
    matched_fields: list[str] = field(default_factory=list)
    # Phase 2 only; empty in legacy mode.
    field_matches: list[FieldMatch] = field(default_factory=list)
    field_coverage: float | None = None  # share of query concepts covered by fields
    concepts_total: int = 0
    concepts_covered: int = 0
    ranking_score: float = 0.0
    features: RankingFeatures | None = None
    retriever_ranks: dict[str, int] = field(default_factory=dict)

    # -- generator architecture (appended; all default so existing positional
    # construction and every current caller keep working unchanged).
    instance_id: str = ""
    family_id: str = ""
    generator_ranks: dict[str, int] = field(default_factory=dict)
    # None means the generator ran and did not retrieve this candidate -- a mask,
    # not a zero score.
    generator_scores: dict[str, float | None] = field(default_factory=dict)
    masked_sources: tuple[str, ...] = ()
    admitted_via: str = ""
    cross_encoder_score: float | None = None

    @property
    def retrieval_share(self) -> float:
        """Normalized share of displayed ranking mass; not confidence."""
        return self.probability

    @property
    def confidence_pct(self) -> float:
        return 100.0 * self.probability

    @property
    def exact_field_matches(self) -> list[FieldMatch]:
        return [m for m in self.field_matches if m.exact]

    @property
    def semantic_field_matches(self) -> list[FieldMatch]:
        return [m for m in self.field_matches if not m.exact]

    @property
    def has_ambiguous_links(self) -> bool:
        return any(m.ambiguous for m in self.field_matches)


@dataclass
class Result:
    """The full answer to one query."""

    query: str
    candidates: list[Candidate]
    confident: bool
    p1: float
    p2: float
    margin: float
    entropy_bits: float
    normalized_entropy: float
    n_reports: int
    field_expert_used: bool
    detected_fields: list[str] = field(default_factory=list)
    answerability: AnswerabilityDecision | None = None
    intent: QueryIntent | None = None
    explanation: ResultExplanation | None = None

    # -- generator architecture (appended, defaulted) --------------------
    outcome: object | None = None
    fallbacks: tuple[str, ...] = ()

    @property
    def decision(self) -> str:
        """The three-way decision, when the generator pipeline produced this."""
        if self.outcome is not None:
            return self.outcome.decision.value
        return self.status

    @property
    def clarification(self):
        return getattr(self.outcome, "clarification", None)

    @property
    def status(self) -> str:
        if self.answerability is not None:
            return self.answerability.outcome.value
        return "confident" if self.confident else "ambiguous"


def _log_softmax(scores: np.ndarray, temperature: float) -> np.ndarray:
    """Numerically stable log-softmax at a given temperature."""
    if temperature <= 0:
        raise ValueError(f"Temperature must be > 0, got {temperature}")
    scaled = scores.astype(np.float64) / temperature
    return scaled - logsumexp(scaled)


def _entropy_bits(log_probs: np.ndarray) -> float:
    """Shannon entropy H(P) in bits, computed from log-probabilities."""
    probs = np.exp(log_probs)
    nonzero = probs > 0
    return float(-np.sum(probs[nonzero] * log_probs[nonzero]) / np.log(2.0))


# The generator pipeline's three-way decision, expressed in the vocabulary the
# legacy `Result` uses. The two enums mean the same three things, so an older
# caller reading `result.answerability.outcome` gets the real decision rather than
# a None it will crash on.
_DECISION_TO_ANSWERABILITY = {
    "RETURN_RESULTS": AnswerabilityOutcome.ANSWERABLE,
    "ASK_CLARIFICATION": AnswerabilityOutcome.NEEDS_CLARIFICATION,
    "NO_CONFIDENT_MATCH": AnswerabilityOutcome.NO_SATISFACTORY_REPORT,
}


def _answerability_from(outcome) -> AnswerabilityDecision:
    """Adapt a `SearchOutcome` to the legacy answerability shape.

    `score` is deliberately 0.0 and `calibrated` deliberately False: the generator
    path ships no calibration artifact, and the decision head's fallback is a
    deterministic policy, not a probability. Synthesising a plausible-looking score
    here would present an uncalibrated judgement as a calibrated one, which is the
    single thing this field must never do.
    """
    detail = outcome.decision_detail
    return AnswerabilityDecision(
        outcome=_DECISION_TO_ANSWERABILITY[outcome.decision.value],
        score=0.0,
        reasons=tuple(detail.reasons) if detail is not None else (),
        calibrated=False,
    )


class ReportFinder:
    """Query interface over a built Representation."""

    def __init__(self, rep: Representation, cfg: Config):
        self.rep = rep
        self.cfg = cfg
        self._n = len(rep)
        self._intent_parser = None
        self._bm25 = None
        self._char = None
        self._field_index = None
        self._ranker = None
        self._search_pipeline = None

    # -- generator architecture -------------------------------------------

    @property
    def pipeline(self):
        """The generator pipeline, built once on first use.

        Constructed lazily so importing `ReportFinder` never loads a model, but
        eagerly enough that a missing or stale index fails on the first search
        rather than silently degrading.
        """
        if self._search_pipeline is None:
            from .pipeline.factory import build_pipeline

            self._search_pipeline = build_pipeline(self.cfg, self.rep.frame)
        return self._search_pipeline

    def search(self, request):
        """Run an authorized search and return the full outcome.

        This is the API the generator architecture is designed around: it carries
        the principal, the three-way decision, family and instance identity,
        grounded evidence and telemetry. `query()` remains as a compatibility
        adapter for the CLI, the Streamlit app and the benchmark harness.
        """
        return self.pipeline.run(request)

    def _ensure_hybrid_indexes(self) -> None:
        if self._intent_parser is not None:
            return
        frame = self.rep.frame
        self._intent_parser = IntentParser(frame, self.cfg)
        weights = {
            "title": self.cfg.bm25_title, "description": self.cfg.bm25_description,
            "fields": self.cfg.bm25_fields, "prompts": self.cfg.bm25_prompts,
            "category": self.cfg.bm25_category, "tags": self.cfg.bm25_tags,
            "data_source": self.cfg.bm25_data_source,
            "area_where_used": self.cfg.bm25_area_where_used,
        }
        # Flatten field metadata into two additional searchable zones without
        # mutating the canonical frame.
        indexed = frame.copy()
        indexed["field_descriptions"] = [" ".join(getattr(m, "description", "") for m in metas) for metas in frame.get("field_meta", [[]]*self._n)]
        indexed["business_objects"] = [" ".join(getattr(m, "business_object", "") for m in metas) for metas in frame.get("field_meta", [[]]*self._n)]
        weights.update({"field_descriptions": self.cfg.bm25_field_descriptions,
                        "business_objects": self.cfg.bm25_business_objects})
        self._bm25 = BM25FIndex(indexed, weights)
        self._char = CharNgramIndex(self.rep.docs, (self.cfg.char_ngram_min, self.cfg.char_ngram_max))
        self._field_index = FieldIndex(frame)

    # -- experts -----------------------------------------------------------

    def _dense_similarities(self, query: str) -> np.ndarray:
        encoder = self.rep.get_encoder()
        text = encode_query_text(query, self.rep.dense_model_name)
        vec = encoder.encode(
            [text], convert_to_numpy=True, normalize_embeddings=True
        )[0].astype(np.float32)
        return self.rep.dense @ vec  # rows are L2-normalized => cosine

    def _lsa_similarities(self, query: str) -> np.ndarray:
        vec = self.rep.lsa_pipeline.transform([query])[0].astype(np.float32)
        return self.rep.lsa @ vec

    def _field_log_prob(self, query: str) -> tuple[np.ndarray | None, list[str]]:
        """Optional third expert: field names the user literally asked for.

        Stays deliberately thin -- it looks for corpus field names appearing
        verbatim in the query and forms a smoothed distribution over the reports
        that carry them. When nothing is detected it returns None, and the PoE
        simply has one fewer factor (a uniform factor is a no-op anyway).
        """
        if not self.cfg.use_field_expert:
            return None, []

        query_lower = " " + " ".join(_tokens(query)) + " "
        detected: list[str] = []
        for field_name in self.rep.field_vocab:
            # Multi-word field names only: single tokens are too noisy to treat
            # as a literal field request.
            if " " not in field_name:
                continue
            normalized = " ".join(_tokens(field_name))
            if normalized and f" {normalized} " in query_lower:
                detected.append(field_name)

        if not detected:
            return None, []

        counts = np.zeros(self._n, dtype=np.float64)
        for field_name in detected:
            counts[self.rep.field_vocab[field_name]] += 1.0
        counts /= len(detected)  # fraction of requested fields present

        # Additive smoothing keeps every report in the support: a zero here would
        # veto a report outright in the product, which is too strong a claim for
        # a heuristic string match.
        smoothed = counts + 0.05
        log_probs = np.log(smoothed) - np.log(smoothed.sum())
        return log_probs, detected

    # -- explanation -------------------------------------------------------

    def _explain_overlap(self, query: str, index: int) -> tuple[list[str], list[str]]:
        """Which query terms and which report fields actually overlapped."""
        row = self.rep.frame.iloc[index]
        query_terms = set(_tokens(query))

        haystack_zones = [str(row["title"]), str(row["category"]),
                          str(row["data_source"]), str(row["report_type"]), str(row["tags"])]
        haystack = set(_tokens(" ".join(haystack_zones)))
        for item in list(row["fields"]) + list(row["prompts"]):
            haystack.update(_tokens(item))

        matched_terms = sorted(query_terms & haystack)

        matched_fields = [
            f for f in row["fields"] if query_terms & set(_tokens(f))
        ]
        return matched_terms, matched_fields[:6]

    def _explain_fields(self, query: str, index: int) -> tuple[list[FieldMatch], int, int]:
        """Which Phase 2 fields explain this report, and how much of the query they cover.

        Coverage is measured over query *concepts* (content tokens), counting a
        concept covered if any field name or description mentions it. This is an
        explanation of the retrieved result, not an input to the score -- ranking
        stays entirely with the product-of-experts posterior.
        """
        row = self.rep.frame.iloc[index]
        metas = row.get("field_meta")
        if not isinstance(metas, (list, tuple)) or not metas:
            return [], 0, 0

        query_terms = set(_tokens(query))
        if not query_terms:
            return [], 0, 0

        matches: list[FieldMatch] = []
        covered: set[str] = set()

        for meta in metas:
            name_terms = set(_tokens(meta.field_name))
            name_hit = query_terms & name_terms
            desc_hit = query_terms & set(_tokens(meta.description))

            if not name_hit and not desc_hit:
                continue
            covered |= name_hit | desc_hit
            matches.append(
                FieldMatch(
                    field_name=meta.field_name,
                    business_object=meta.business_object,
                    description=meta.description,
                    exact=bool(name_hit),
                    ambiguous=meta.is_ambiguous,
                    match_method=meta.match_method,
                )
            )

        # Exact name matches are the strongest evidence; surface them first.
        matches.sort(key=lambda m: (not m.exact, m.field_name))
        return matches, len(query_terms), len(covered & query_terms)

    # -- main --------------------------------------------------------------

    def _query_legacy(self, text: str, top_k: int | None = None) -> Result:
        """Reproduce the historical weighted-logit posterior and thresholds."""
        if not text or not text.strip():
            raise ValueError("Query is empty.")

        top_k = top_k or self.cfg.top_k
        cfg = self.cfg

        if self.cfg.use_dense and getattr(self.rep, "dense_mode", "disabled") == "native":
            s_dense = self._dense_similarities(text)
        else:
            # Legacy mode remains usable on machines without sentence-transformers.
            # A neutral expert is a no-op in the geometric mixture.
            s_dense = np.zeros(self._n, dtype=np.float64)
        s_lsa = self._lsa_similarities(text)

        log_p_dense = _log_softmax(s_dense, cfg.t_dense)
        log_p_lsa = _log_softmax(s_lsa, cfg.t_lsa)

        # Geometric mixture in log space. The per-expert log-normalizers are
        # constant across reports and vanish in the final renormalization, but we
        # keep them so log_p_dense/log_p_lsa stay interpretable on their own.
        log_joint = cfg.alpha * log_p_dense + (1.0 - cfg.alpha) * log_p_lsa

        log_p_field, detected_fields = self._field_log_prob(text)
        if log_p_field is not None:
            log_joint = log_joint + cfg.field_expert_weight * log_p_field

        log_post = log_joint - logsumexp(log_joint)  # renormalize over all reports
        posterior = np.exp(log_post)

        order = np.argsort(-posterior, kind="stable")
        p1 = float(posterior[order[0]])
        p2 = float(posterior[order[1]]) if self._n > 1 else 0.0
        margin = p1 - p2

        entropy_bits = _entropy_bits(log_post)
        max_entropy = np.log2(self._n) if self._n > 1 else 1.0
        normalized_entropy = float(entropy_bits / max_entropy)

        confident = (p1 >= cfg.tau) and (margin >= cfg.delta)
        n_show = 1 if confident else min(top_k, self._n)

        log_n = np.log(self._n)
        candidates: list[Candidate] = []
        for idx in order[:n_show]:
            idx = int(idx)
            trace = ExpertTrace(
                log_prob_dense=float(log_p_dense[idx]),
                log_prob_lsa=float(log_p_lsa[idx]),
                sim_dense=float(s_dense[idx]),
                sim_lsa=float(s_lsa[idx]),
                lift_dense=float(cfg.alpha * (log_p_dense[idx] + log_n)),
                lift_lsa=float((1.0 - cfg.alpha) * (log_p_lsa[idx] + log_n)),
                log_prob_field=(
                    float(log_p_field[idx]) if log_p_field is not None else None
                ),
            )
            matched_terms, matched_fields = self._explain_overlap(text, idx)
            field_matches, concepts_total, concepts_covered = self._explain_fields(
                text, idx
            )
            candidates.append(
                Candidate(
                    index=idx,
                    probability=float(posterior[idx]),
                    row=self.rep.frame.iloc[idx],
                    trace=trace,
                    matched_terms=matched_terms,
                    matched_fields=matched_fields,
                    field_matches=field_matches,
                    field_coverage=(
                        concepts_covered / concepts_total if concepts_total else None
                    ),
                    concepts_total=concepts_total,
                    concepts_covered=concepts_covered,
                )
            )

        return Result(
            query=text,
            candidates=candidates,
            confident=confident,
            p1=p1,
            p2=p2,
            margin=margin,
            entropy_bits=entropy_bits,
            normalized_entropy=normalized_entropy,
            n_reports=self._n,
            field_expert_used=log_p_field is not None,
            detected_fields=detected_fields,
        )

    @staticmethod
    def _top_results(scores: np.ndarray, k: int) -> list[tuple[int, float]]:
        order = np.argsort(-scores, kind="stable")[:k]
        return [(int(i), float(scores[i])) for i in order]

    def _query_hybrid(self, text: str, top_k: int | None = None) -> Result:
        if not text or not text.strip():
            raise ValueError("Query is empty.")
        self._ensure_hybrid_indexes()
        cfg = self.cfg
        k = min(cfg.candidate_k, self._n)
        intent = self._intent_parser.parse(text)
        vectors: dict[str, np.ndarray] = {}
        rankings: dict[str, list[tuple[int, float]]] = {}
        channel_results = {}
        forms = intent.expansion.query_forms

        def add_channel(name, scorer):
            form_scores = score_forms(scorer, forms)
            result = aggregate_top_two(
                form_scores, [form.weight for form in forms],
                cfg.expansion_aggregate_lambda,
            )
            channel_results[name] = result
            vectors[name] = result.combined
            rankings[name] = [
                (i, score) for i, score in self._top_results(result.combined, k)
                if score > 0
            ]

        if cfg.use_bm25f: add_channel("bm25", self._bm25.scores)
        dense_state = getattr(self.rep, "dense_mode", "native")
        if cfg.use_dense and cfg.dense_mode != "off" and dense_state == "native":
            add_channel("dense", self._dense_similarities)
        if cfg.use_lsa: add_channel("lsa", self._lsa_similarities)
        if cfg.use_char_ngrams: add_channel("char", self._char.scores)
        detected_fields: list[str] = []
        field_scores = None
        if cfg.use_exact_field:
            required = intent.required_fields()
            optional = intent.optional_fields()
            field_scores = self._field_index.scores_for_fields(required, optional)
            vectors["exact_field"] = field_scores.scores
            rankings["exact_field"] = [
                item for item in self._top_results(field_scores.scores, k)
                if item[1] > 0
            ]
            resolved = set(field_scores.resolved)
            detected_fields = [
                field for field in required + optional
                if " ".join(_TOKEN_RE.findall(field.casefold())) in resolved
            ]
            if cfg.use_field_expert:
                # Preserve the historical lowercase diagnostics contract. The
                # active exact-field retriever above remains structured.
                detected_fields = self._field_index.detected(text)

        fused = reciprocal_rank_fusion(rankings, cfg.rrf_k)
        rank_maps = {name: {doc: rank for rank, (doc, _) in enumerate(items, 1)} for name, items in rankings.items()}
        candidate_ids = sorted(fused, key=lambda i: (-fused[i], i))[:k]
        if not candidate_ids:
            # Keep the result contract intact for wholly out-of-domain requests;
            # groundedness will correctly abstain after ranking these neutral rows.
            candidate_ids = list(range(k))
            fused = {idx: 0.0 for idx in candidate_ids}
        relative = {}
        for name, values in vectors.items():
            finite = values[np.isfinite(values)]; mean = float(finite.mean()) if finite.size else 0.0
            std = float(finite.std()) if finite.size else 0.0
            ranks = rank_maps.get(name, {})
            relative[name] = {idx: {"zscore": (float(values[idx])-mean)/std if std else 0.0,
                "percentile": min(1.0, max(0.0, 1.0-(ranks.get(idx, k+1)-1)/max(1, len(ranks)))),
                "reciprocal_rank": 1.0/ranks[idx] if idx in ranks else 0.0,
                "present": float(idx in ranks)} for idx in candidate_ids}
        scored: list[tuple[float, int, RankingFeatures]] = []
        if cfg.use_ltr and self._ranker is None:
            self._ranker = Ranker.load_or_fallback(cfg.ltr_model_path) if cfg.ltr_model_path else Ranker(fallback_reason="no artifact configured")
        active_retrievers = len([name for name, items in rankings.items() if items])
        subquery_positions = [
            position for position, form in enumerate(forms) if form.kind == "subquery"
        ]
        for candidate_position, idx in enumerate(candidate_ids):
            raw = {name: float(values[idx]) for name, values in vectors.items()}
            raw["rrf"] = fused[idx]
            raw["rrf_k"] = cfg.rrf_k
            raw["n_active_retrievers"] = active_retrievers
            raw["candidate_rank_before_ltr"] = 1.0 / (1.0 + candidate_position)
            if subquery_positions:
                satisfied = 0
                for form_position in subquery_positions:
                    if any(
                        result.per_form[form_position, idx] > 0
                        for result in channel_results.values()
                    ):
                        satisfied += 1
                raw["subquery_coverage"] = satisfied / len(subquery_positions)
                raw["subquery_count_satisfied"] = min(satisfied, 4) / 4
            features = extract_features(self.rep.frame.iloc[idx], intent, raw,
                {name: by_doc[idx] for name, by_doc in relative.items()})
            score = self._ranker.score(features) if cfg.use_ltr else linear_score(features, cfg)
            scored.append((score, idx, features))
        scored.sort(key=lambda x: (-x[0], x[1]))

        # Optional cross-encoder is deliberately applied only after candidate
        # generation. Its score is a reranking signal, not confidence.
        if cfg.use_cross_encoder and scored:
            from .ranking.cross_encoder import apply_rerank, get_reranker
            ce = get_reranker(cfg.cross_encoder_model)
            ce_n = min(cfg.cross_encoder_top_k, len(scored))
            ce_scores = ce.scores(text, [self.rep.frame.iloc[i] for _, i, _ in scored[:ce_n]])
            scored = apply_rerank(
                scored, ce_scores, cfg.cross_encoder_weight, ce_n,
            )

        raw_scores = np.asarray([s for s, _, _ in scored], dtype=np.float64)
        # retrieval_share is a display normalization over the candidate set. It
        # is explicitly not used as the answerability estimate.
        scaled_scores = raw_scores / cfg.share_temperature
        shares = np.exp(scaled_scores - logsumexp(scaled_scores)) if raw_scores.size else np.array([])
        p1 = float(shares[0]) if shares.size else 0.0
        p2 = float(shares[1]) if shares.size > 1 else 0.0
        margin = p1 - p2
        if scored:
            top_values = scored[0][2].values
            mandatory_count = len(intent.required_fields())
            optional_count = len(intent.optional_fields())
            if mandatory_count:
                field_coverage = (
                    mandatory_count * top_values["mandatory_coverage"]
                    + optional_count * top_values["optional_concept_coverage"]
                ) / max(1, mandatory_count + optional_count)
            else:
                field_coverage = max(
                    top_values["optional_concept_coverage"],
                    top_values["exact_field_match_strength"],
                )
        else:
            field_coverage = 0.0
        active_rank_maps = {
            name: ranks for name, ranks in rank_maps.items()
            if rankings.get(name)
        }
        top_ranks = [
            ranks.get(scored[0][1], k + 1) for ranks in active_rank_maps.values()
        ] if scored else []
        agreement = sum(r <= 10 for r in top_ranks) / len(top_ranks) if top_ranks else 0.0
        if not raw_scores.size or not np.all(np.isfinite(raw_scores)):
            top_norm = 0.0
        elif len(raw_scores) == 1:
            surviving = scored[0][1]
            evidence = max((float(v[surviving]) for v in vectors.values()), default=0.0)
            top_norm = min(1.0, max(0.0, evidence)) if evidence > .1 else 0.0
        else:
            rest = raw_scores[1:]; std = float(rest.std())
            top_norm = float(1/(1+np.exp(-((raw_scores[0]-float(rest.mean()))/(std if std > 1e-9 else 1.0)))))
        top_fields = self.rep.frame.iloc[scored[0][1]]["fields"] if scored else ()
        support = compute_support(
            intent.expansion, set(self._bm25.idf), top_fields=top_fields,
            unknown_fields=field_scores.unknown if field_scores is not None else (),
            required_combination_exists=(
                field_scores.required_combination_exists
                if field_scores is not None else True
            ),
        )
        categories = [
            self.rep.frame.iloc[idx].get("category", "") for _, idx, _ in scored[:5]
        ]
        breadth, ambiguity_reasons = intent_ambiguity(
            intent, categories, raw_scores[:2],
        )
        if len(raw_scores) > 1:
            raw_margin = max(0.0, float(raw_scores[0] - raw_scores[1]))
        else:
            raw_margin = 1.0 if raw_scores.size else 0.0
        answerability = decide_answerability(
            top_score=top_norm, margin=raw_margin, field_coverage=field_coverage,
            retriever_agreement=agreement, parser_confidence=intent.parser_confidence,
            support=support, breadth=breadth,
            high_threshold=cfg.answerability_high_threshold,
            insufficient_threshold=cfg.answerability_insufficient_threshold,
            min_support_ratio=cfg.min_support_ratio,
            hard_support_floor=cfg.hard_support_floor,
            breadth_threshold=cfg.breadth_threshold,
            min_margin=cfg.answerability_min_margin,
            min_high_support=cfg.answerability_min_high_support,
            min_field_coverage=cfg.answerability_min_field_coverage,
            intercept=cfg.answerability_intercept,
            top_score_weight=cfg.answerability_top_score_weight,
            margin_weight=cfg.answerability_margin_weight,
            margin_scale=cfg.answerability_margin_scale,
            field_coverage_weight=cfg.answerability_field_coverage_weight,
            retriever_agreement_weight=cfg.answerability_retriever_agreement_weight,
            parser_confidence_weight=cfg.answerability_parser_confidence_weight,
            support_weight=cfg.answerability_support_weight,
            ambiguity_weight=cfg.answerability_ambiguity_weight,
            ambiguity_reasons=ambiguity_reasons,
        )
        confident = answerability.outcome is AnswerabilityOutcome.HIGH_CONFIDENCE
        # Previously this collapsed to a single result whenever the verdict was
        # ANSWERABLE, discarding ranks 2..k that had already been computed. That
        # made every offline nDCG/MRR number a measurement of the collapse rather
        # than of retrieval, and gave the user no alternatives to fall back on.
        n_show = min(top_k or cfg.top_k, len(scored))
        candidates = []
        for pos, (rank_score, idx, features) in enumerate(scored[:n_show]):
            matched_terms, matched_fields = self._explain_overlap(text, idx)
            field_matches, concepts_total, concepts_covered = self._explain_fields(text, idx)
            candidates.append(Candidate(
                index=idx, probability=float(shares[pos]), row=self.rep.frame.iloc[idx],
                trace=ExpertTrace(0.0, 0.0, float(vectors.get("dense", np.zeros(self._n))[idx]),
                                  float(vectors.get("lsa", np.zeros(self._n))[idx]), 0.0, 0.0),
                matched_terms=matched_terms, matched_fields=matched_fields,
                field_matches=field_matches,
                field_coverage=concepts_covered/concepts_total if concepts_total else None,
                concepts_total=concepts_total, concepts_covered=concepts_covered,
                ranking_score=rank_score, features=features,
                retriever_ranks={n: ranks[idx] for n, ranks in rank_maps.items() if idx in ranks},
            ))
        log_shares = np.log(shares, where=shares > 0, out=np.full_like(shares, -np.inf))
        entropy = _entropy_bits(log_shares) if shares.size else 0.0
        explanation = None
        if candidates and intent.expansion is not None:
            explanation = build_result_explanation(
                intent, candidates[0].row, candidates[0].retriever_ranks,
                candidates[0].features,
            )
        return Result(text, candidates, confident, p1, p2, margin, entropy,
                      entropy / np.log2(len(shares)) if len(shares) > 1 else 0.0,
                      self._n, bool(cfg.use_field_expert and detected_fields),
                      detected_fields if cfg.use_field_expert else [],
                      answerability=answerability, intent=intent,
                      explanation=explanation)

    def candidate_for(self, family) -> Candidate:
        """Render one `FamilyResult` as a display `Candidate`.

        The bridge between the two-level answer and the row-shaped card every
        renderer already knows how to draw. Public because the Streamlit app and
        the API both need it and neither should reimplement the position mapping.
        """
        selected = family.selected
        # Corpus position is the frame row position: `build_corpus_model` builds
        # instances in frame order and `position_of` is its inverse.
        position = self.pipeline.corpus.position_of(selected.instance_id)
        record = selected.record
        return Candidate(
            index=position,
            # A display share over the returned set, never a confidence.
            probability=0.0,
            row=self.rep.frame.iloc[position],
            trace=ExpertTrace(0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
            ranking_score=selected.score,
            retriever_ranks=dict(record.ranks),
            instance_id=selected.instance_id,
            family_id=family.family_id,
            generator_ranks=dict(record.ranks),
            generator_scores=dict(record.scores),
            masked_sources=tuple(sorted(record.masked)),
            admitted_via=selected.admitted_via,
            cross_encoder_score=selected.cross_encoder_score,
        )

    def _query_generators(self, text: str, top_k: int | None, principal) -> Result:
        """Run the generator pipeline and adapt it to the legacy `Result` shape.

        **Deprecated.** `Result` predates the family/instance separation, so it
        carries only the selected instance of each family and cannot express the
        two-level answer. `search()` is the real contract; this exists so callers
        that have not migrated keep working.

        What it must never do is *silently* discard the decision, the family
        identity, the warnings or the versions -- a caller reading a truncated
        answer with no sign that it was truncated is worse than one that breaks.
        So the full `SearchOutcome` is attached, and the fields `Result` can
        express are filled rather than nulled.
        """
        from .auth import DEVELOPMENT_PRINCIPAL, SearchRequest

        outcome = self.search(
            SearchRequest(
                raw_query=text,
                principal=principal or DEVELOPMENT_PRINCIPAL,
                top_k=top_k,
            )
        )

        candidates = [self.candidate_for(family) for family in outcome.families]
        decision = outcome.decision
        return Result(
            text, candidates,
            # `confident` is retained for older callers and means exactly
            # "the pipeline chose to return results".
            decision.value == "RETURN_RESULTS",
            0.0, 0.0, 0.0, 0.0, 0.0,
            self._n, False, [],
            answerability=_answerability_from(outcome),
            intent=self.pipeline.intent_parser.parse(text),
            explanation=None,
            outcome=outcome,
            fallbacks=tuple(outcome.active_fallbacks),
        )

    def query(self, text: str, top_k: int | None = None, principal=None) -> Result:
        """Search using the configured architecture."""
        # Alpha is a legacy weighted-logit knob. Preserve its documented pure-
        # expert behavior for callers sweeping the endpoints.
        if self.cfg.retrieval_mode == "legacy_weighted_logit":
            return self._query_legacy(text, top_k)
        if self.cfg.retrieval_mode == "generators":
            return self._query_generators(text, top_k, principal)
        if self.cfg.retrieval_mode != "hybrid":
            raise ValueError(f"Unknown retrieval_mode: {self.cfg.retrieval_mode}")
        return self._query_hybrid(text, top_k)


def why_matched(candidate: Candidate) -> str:
    """One-line explanation of why a candidate surfaced."""
    if candidate.retriever_ranks:
        retrievers = ", ".join(
            f"{name} #{rank}" for name, rank in sorted(candidate.retriever_ranks.items())
        )
        parts: list[str] = []
        if candidate.matched_terms:
            parts.append("terms: " + ", ".join(candidate.matched_terms[:6]))
        if candidate.matched_fields:
            parts.append("fields: " + "; ".join(candidate.matched_fields[:3]))
        if not parts:
            parts.append("semantic or expanded-query match")
        return " | ".join(parts) + f" | retriever support {retrievers}"

    trace = candidate.trace
    share = trace.dense_share
    if share is None:
        balance = (
            f"dense lift {trace.lift_dense:+.1f} / LSA lift {trace.lift_lsa:+.1f} nats"
        )
    else:
        balance = f"{share:.0%} dense / {1 - share:.0%} LSA"

    parts: list[str] = []
    if candidate.matched_terms:
        parts.append("terms: " + ", ".join(candidate.matched_terms[:6]))
    if candidate.matched_fields:
        parts.append("fields: " + "; ".join(candidate.matched_fields[:3]))
    if not parts:
        parts.append("semantic match only (no literal term overlap)")

    return " | ".join(parts) + f" | evidence {balance}"


def explain_fields(candidate: Candidate) -> list[str]:
    """Phase 2 field-level explanation lines.

    Deliberately worded as evidence rather than certainty: link confidence is an
    ordinal record of *how* a link was made, not a probability, so it is never
    presented as a percentage of correctness. Ambiguous links are disclosed.
    """
    if not candidate.field_matches:
        return []

    lines: list[str] = []

    if candidate.concepts_total:
        lines.append(
            f"Matched {candidate.concepts_covered} of {candidate.concepts_total} "
            f"requested field concepts."
        )

    exact = candidate.exact_field_matches
    if exact:
        names = ", ".join(m.field_name for m in exact[:3])
        more = f" (+{len(exact) - 3} more)" if len(exact) > 3 else ""
        lines.append(f"Relevant fields include {names}{more}.")

    semantic = candidate.semantic_field_matches
    if semantic:
        names = ", ".join(m.field_name for m in semantic[:2])
        lines.append(f"Related by field description: {names}.")

    objects = list(dict.fromkeys(m.business_object for m in candidate.field_matches if m.business_object))
    if objects:
        lines.append(f"Business objects: {', '.join(objects[:3])}.")

    methods = {m.match_method for m in candidate.field_matches}
    if methods == {"exact_name"}:
        lines.append("Field metadata was linked through an exact report-name match.")
    elif "ambiguous_multi" in methods:
        ambiguous_count = sum(1 for m in candidate.field_matches if m.ambiguous)
        lines.append(
            f"Note: {ambiguous_count} of these field links could not be uniquely "
            f"resolved — this report shares its name with others in the catalog, "
            f"so some fields may belong to a different copy."
        )
    elif "composite_business_object" in methods:
        lines.append(
            "Field metadata was linked through report name plus business-object "
            "corroboration."
        )
    return lines
