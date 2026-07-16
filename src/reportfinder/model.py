"""The dual-space product-of-experts retrieval model.

Two independent experts each induce a distribution over the corpus, and the
posterior is their geometric mixture:

    P_dense(r|q) = softmax(s_dense / T_d)
    P_lsa(r|q)   = softmax(s_lsa   / T_l)
    P(r|q)      ∝ P_dense(r|q)^α · P_lsa(r|q)^(1-α)

This is a product of experts, not a weighted sum of scores. The distinction
matters in practice: a sum lets one confident expert drag a candidate up on its
own, whereas a product requires *both* experts to find the candidate plausible --
either one can veto by assigning low probability. The output is a normalized
posterior over all reports, so the top-1 probability is directly interpretable as
confidence, and the shape of the distribution (margin, entropy) tells us whether
the query was answerable at all.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

import numpy as np
from scipy.special import logsumexp

from .config import Config
from .represent import Representation, encode_query_text

# Tokens too generic to be evidence of anything in a report corpus.
_STOPWORDS = frozenset(
    """
    a an the of for by with to in on and or is are was were be been show me
    list all get find give please report reports i want need which what who
    how many that this these those from at as it its any some
    """.split()
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

    @property
    def status(self) -> str:
        if self.confident:
            return "confident"
        return "ambiguous"


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


class ReportFinder:
    """Query interface over a built Representation."""

    def __init__(self, rep: Representation, cfg: Config):
        self.rep = rep
        self.cfg = cfg
        self._n = len(rep)

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

    def query(self, text: str, top_k: int | None = None) -> Result:
        """Compute the posterior over all reports and apply the decision rule."""
        if not text or not text.strip():
            raise ValueError("Query is empty.")

        top_k = top_k or self.cfg.top_k
        cfg = self.cfg

        s_dense = self._dense_similarities(text)
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

        order = np.argsort(-posterior)
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


def why_matched(candidate: Candidate) -> str:
    """One-line explanation of why a candidate surfaced."""
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
