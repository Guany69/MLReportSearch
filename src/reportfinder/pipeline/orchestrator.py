"""The search pipeline: one authorized request in, one grounded outcome out.

Everything is injected. `SearchPipeline` never constructs a model, loads a
checkpoint or reads a config file -- `factory.build_pipeline` does that. The
separation is what lets the whole pipeline be tested offline with deterministic
fakes, which the previous architecture could not do because every index was built
inside the search object.

Stage order, and why:

    authorize -> prepare -> generate -> union -> assess risk -> shortlist
    -> rerank -> fuse -> score instances -> aggregate families -> decide

Authorization is first because nothing may be retrieved that the principal cannot
see. Risk is assessed *after* the union because the best evidence for "we may have
missed it" is how much the generators disagreed. Reranking comes after shortlist
selection but reads the whole shortlist, and family aggregation comes after
instance scoring because the instance is what a user actually runs.
"""

from __future__ import annotations

import logging
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field

from ..auth import SearchRequest
from ..config import DEFAULT_GENERATOR_WORKERS
from .aggregate import (
    FamilyResult,
    aggregate_families,
    aggregation_telemetry,
    attach_expanded,
    expand_families,
    score_instances,
)
from .decide import Clarification, Decision, DecisionResult, decide
from .prepare import build_query_plan
from .rerank import rerank_shortlist
from .risk import assess_recall_risk
from .shortlist import select_shortlist
from .telemetry import SearchTelemetry
from .union import build_union

log = logging.getLogger(__name__)


@dataclass
class SearchOutcome:
    """The grounded response."""

    request_id: str
    decision: Decision
    families: list[FamilyResult] = field(default_factory=list)
    clarification: Clarification | None = None
    warnings: list[str] = field(default_factory=list)
    catalog_version: str = ""
    model_bundle_version: str = ""
    latency_ms: float = 0.0
    telemetry: SearchTelemetry | None = None
    decision_detail: DecisionResult | None = None

    @property
    def active_fallbacks(self) -> list[str]:
        return self.telemetry.active_fallbacks if self.telemetry else []


@dataclass
class RunState:
    """Everything one search produced, including what the response drops.

    `outcome` is what a user sees. The rest is what training and offline analysis
    need: the full family list before the top-k cut and before abstention
    suppression, the real `QueryPlan` rather than a stand-in, and every stage
    result that fed the decision.
    """

    outcome: SearchOutcome
    plan: object | None = None
    risk: object | None = None
    union: object | None = None
    shortlist: object | None = None
    rerank: object | None = None
    fusion: object | None = None
    # Pre-truncation, pre-suppression. Empty only when retrieval genuinely found
    # nothing, never merely because the system chose not to answer.
    families: list[FamilyResult] = field(default_factory=list)
    decision_result: DecisionResult | None = None
    universe: object | None = None


def _generator_hits(union) -> dict[str, tuple[str, ...]]:
    """Which instances each generator nominated, for candidate-survival evaluation."""
    hits: dict[str, list[str]] = {name: [] for name in union.generators_run}
    for record in union.records.values():
        for generator in record.found_by:
            hits.setdefault(generator, []).append(record.instance_id)
    return {generator: tuple(ids) for generator, ids in hits.items()}


class _Support:
    """Field-existence facts for the hard gates.

    Deliberately narrow: these are catalog facts ("no report has this field"), not
    confidence estimates, and they are what produce the specific failure messages
    users see today.
    """

    def __init__(self, unknown=(), required_combination_exists=True) -> None:
        self.unknown_requested_fields = tuple(unknown)
        self.required_combination_exists = required_combination_exists


class SearchPipeline:
    """Runs one authorized search."""

    def __init__(
        self,
        *,
        cfg,
        corpus,
        generators,
        resolver,
        intent_parser,
        reranker=None,
        fuser=None,
        decision_head=None,
        field_index=None,
        late_interaction_generator=None,
        bundle=None,
    ) -> None:
        self.cfg = cfg
        self.corpus = corpus
        self.generators = list(generators)
        self.resolver = resolver
        self.intent_parser = intent_parser
        self.reranker = reranker
        self.decision_head = decision_head
        self.field_index = field_index
        self.late_interaction_generator = late_interaction_generator
        self.bundle = bundle
        # Default matches RetrievalConfig's, so a hand-built cfg cannot silently
        # run serially while the config documents concurrency.
        self.max_generator_workers = getattr(
            cfg.retrieval, "max_generator_workers", DEFAULT_GENERATOR_WORKERS
        )

        if fuser is None:
            from .fuse import RrfFuser

            fuser = RrfFuser(constant=cfg.retrieval.rrf_constant)
        self.fuser = fuser

    # -- helpers ----------------------------------------------------------

    def _depth_for(self, generator) -> int:
        retrieval = self.cfg.retrieval
        if generator.name == "bm25f":
            return retrieval.bm25f.k
        if generator.name == "splade":
            return retrieval.splade.k
        if generator.name == "prototype":
            return retrieval.prototypes.k
        if generator.name == "late_interaction":
            return retrieval.late_interaction.k
        return retrieval.dense.k_per_view

    def _support(self, plan) -> _Support:
        if self.field_index is None or plan.intent is None:
            return _Support()
        scores = self.field_index.scores_for_fields(
            plan.intent.required_fields(), plan.intent.optional_fields()
        )
        return _Support(
            unknown=scores.unknown,
            required_combination_exists=scores.required_combination_exists,
        )

    def _generate_one(self, generator, plan, universe):
        """One generator's nomination, with its failure contained.

        A generator that raises must not take the request down: it is recorded as
        failed, excluded from the union's mask set (masking against something that
        never ran would misreport "did not retrieve" as evidence), and the risk
        policy widens to compensate.
        """
        try:
            return generator.generate(plan, universe, self._depth_for(generator))
        except Exception as error:  # noqa: BLE001 - see docstring
            from ..generators.base import GeneratorResult

            log.warning(
                "generator_failed",
                extra={"generator": generator.name, "error": type(error).__name__},
            )
            return GeneratorResult(
                generator=generator.name, query_variant="Q0",
                view_type=getattr(generator, "view_type", None),
                error=f"failed: {type(error).__name__}",
            )

    def _run_generators(self, plan, universe) -> list:
        """Run every generator, concurrently when it is worth it.

        The generators share no mutable state and torch releases the GIL during
        inference, so this is real parallelism rather than interleaving. Results are
        collected in **submission order**, not completion order: the union must not
        depend on which index happened to finish first, or two identical requests
        could tie-break differently.
        """
        if self.max_generator_workers <= 1 or len(self.generators) <= 1:
            return [
                self._generate_one(g, plan, universe) for g in self.generators
            ]

        workers = min(self.max_generator_workers, len(self.generators))
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = [
                pool.submit(self._generate_one, generator, plan, universe)
                for generator in self.generators
            ]
            return [future.result() for future in futures]

    # -- the pipeline -----------------------------------------------------

    def run(self, request: SearchRequest) -> SearchOutcome:
        """One authorized search."""
        return self._execute(request).outcome

    def run_traced(self, request: SearchRequest) -> RunState:
        """The same search, with every intermediate kept.

        Training needs what serving throws away. In particular `outcome.families`
        is empty on NO_CONFIDENT_MATCH -- correctly, because a user must not be
        shown cards the system just declared it has no confidence in -- but a
        feature generator reading only the outcome therefore produced *nothing*
        for ~40% of queries, silently training on the easy ones. `RunState` keeps
        the pre-decision family list so that abstained queries still contribute.
        """
        return self._execute(request)

    def _execute(self, request: SearchRequest) -> RunState:
        started = time.perf_counter()
        marks: dict[str, float] = {}

        def mark(stage: str, since: float) -> float:
            now = time.perf_counter()
            marks[stage] = (now - since) * 1000
            return now

        telemetry = SearchTelemetry(
            request_id=request.request_id,
            catalog_version=self.corpus.catalog_version,
            model_bundle_version=(
                self.bundle.manifest.bundle_version if self.bundle else ""
            ),
        )
        if self.bundle is not None:
            telemetry.active_fallbacks = list(self.bundle.active_fallbacks)
            telemetry.degraded_components = list(self.bundle.manifest.degraded())
            telemetry.build_config_hash = self.bundle.manifest.config_hash
            telemetry.runtime_config_hash = getattr(
                self.bundle, "runtime_config_hash", ""
            )
            telemetry.runtime_config_differs = bool(
                getattr(self.bundle, "config_drift", False)
            )
            if telemetry.runtime_config_differs:
                telemetry.warnings.append(
                    "Serving configuration differs from the configuration this "
                    "bundle was built with. The stored indexes may still be "
                    "compatible, but reproducing a recorded result requires the "
                    "runtime configuration, not just the bundle id "
                    f"(build {telemetry.build_config_hash}, "
                    f"runtime {telemetry.runtime_config_hash})."
                )

        # -- authorize ----------------------------------------------------
        cursor = time.perf_counter()
        universe = self.resolver.resolve(request.principal, self.corpus)
        telemetry.authorization = universe.telemetry()
        if universe.development_default:
            telemetry.active_fallbacks.append("authorization:allow_all_dev_default")
            telemetry.warnings.append(
                "Authorization is using the permissive development default; "
                "no entitlement source is configured."
            )
        cursor = mark("authorize", cursor)

        if universe.is_empty:
            # Fail closed: an unresolvable or empty entitlement set must never
            # widen into a full-estate search.
            telemetry.latency_ms = marks
            return RunState(outcome=SearchOutcome(
                request_id=request.request_id,
                decision=Decision.NO_CONFIDENT_MATCH,
                warnings=[
                    universe.reason or "no reports are available to this user"
                ],
                catalog_version=self.corpus.catalog_version,
                model_bundle_version=telemetry.model_bundle_version,
                latency_ms=(time.perf_counter() - started) * 1000,
                telemetry=telemetry,
            ))

        # -- prepare ------------------------------------------------------
        intent = self.intent_parser.parse(request.raw_query)
        plan = build_query_plan(
            intent,
            raw_query=request.raw_query,
            clarification_context=request.clarification_context,
        )
        telemetry.query = plan.telemetry()
        cursor = mark("prepare", cursor)

        # -- generate -----------------------------------------------------
        results = self._run_generators(plan, universe)
        cursor = mark("generate", cursor)

        union = build_union(
            results,
            family_of=lambda i: self.corpus.instance(i).family_id,
            rrf_constant=self.cfg.retrieval.rrf_constant,
        )
        telemetry.record_many(union.records, "generated")
        telemetry.record_many(union.records, "union")
        telemetry.retrieval = union.telemetry()
        telemetry.generator_hits = _generator_hits(union)
        cursor = mark("union", cursor)

        # -- assess risk --------------------------------------------------
        risk = assess_recall_risk(
            plan, union, self.cfg,
            authorized_count=len(universe),
            late_interaction_available=self.late_interaction_generator is not None,
        )

        if risk.run_late_interaction and self.late_interaction_generator is not None:
            # A widening pass: its candidates join the same union, they never
            # replace what is already there.
            extra = self.late_interaction_generator.generate(
                plan, universe, self.cfg.retrieval.late_interaction.k
            )
            union = build_union(
                [*results, extra],
                family_of=lambda i: self.corpus.instance(i).family_id,
                rrf_constant=self.cfg.retrieval.rrf_constant,
            )
            telemetry.record_many(union.records, "generated")
            telemetry.record_many(union.records, "union")
            telemetry.retrieval = union.telemetry()
            telemetry.generator_hits = _generator_hits(union)

        telemetry.risk = risk.telemetry()
        cursor = mark("risk", cursor)

        # -- shortlist ----------------------------------------------------
        shortlist = select_shortlist(union, risk, self.cfg)
        telemetry.record_many(shortlist.instance_ids, "shortlist")
        telemetry.shortlist = shortlist.telemetry()
        cursor = mark("shortlist", cursor)

        # -- rerank -------------------------------------------------------
        rerank = rerank_shortlist(
            shortlist, self.corpus, self.reranker,
            raw_query=plan.raw_query,
            max_report_chars=self.cfg.rerank.max_report_chars,
        )
        if rerank.ran:
            telemetry.record_many(rerank.scores, "rerank")
        telemetry.rerank = rerank.telemetry()
        cursor = mark("rerank", cursor)

        # -- fuse ---------------------------------------------------------
        fusion = self.fuser.fuse(shortlist, union, rerank, self.corpus, plan)
        telemetry.record_many(fusion.scores, "fusion")
        telemetry.fusion = fusion.telemetry()
        if fusion.fallback_active:
            telemetry.active_fallbacks.append(f"fusion:{fusion.method}")
        cursor = mark("fuse", cursor)

        # -- instances, then families, then the rest of those families -----
        scored = score_instances(shortlist, fusion, rerank)
        families = aggregate_families(scored, self.corpus, universe=universe)
        cursor = mark("aggregate", cursor)

        # Family-first expansion. Ranking has decided *which reports*; this decides
        # which concrete copy of each, over every authorized instance rather than
        # only the ones retrieval happened to surface.
        expansion = expand_families(
            families, scored, self.corpus, universe,
            plan=plan, cfg=self.cfg, scorer=self.reranker, union=union,
        )
        attach_expanded(families, expansion, self.corpus, plan)
        # Everything still in contention after expansion -- retrieved members and
        # expanded ones alike -- so the survival trace attributes a later loss to
        # final ranking rather than to expansion.
        telemetry.record_many(
            [m.instance_id for f in families for m in f.instances], "expansion"
        )
        telemetry.aggregation = {
            **aggregation_telemetry(families), **expansion.telemetry(),
        }
        cursor = mark("expand", cursor)

        # -- decide -------------------------------------------------------
        outcome = decide(
            families, corpus=self.corpus, plan=plan, risk=risk, union=union,
            rerank=rerank, fusion=fusion, support=self._support(plan),
            head=self.decision_head, cfg=self.cfg,
        )
        telemetry.decision = outcome.telemetry()
        if outcome.fallback_active:
            telemetry.active_fallbacks.append(f"decision:{outcome.method}")
        mark("decide", cursor)

        # Several stages can name the same fallback (the manifest records the
        # absent artifact, the fuser records that it is running). Report each once.
        telemetry.active_fallbacks = sorted(set(telemetry.active_fallbacks))

        top_k = request.top_k or self.cfg.top_k
        # NO_CONFIDENT_MATCH returns no result cards. Returning them anyway is the
        # failure mode the three-way decision exists to prevent: a user shown a
        # ranked list reads it as an answer regardless of the status beside it.
        # ASK_CLARIFICATION keeps its candidates -- the question is grounded in them.
        shown = [] if outcome.decision is Decision.NO_CONFIDENT_MATCH else families[:top_k]
        telemetry.record_many([f.selected.instance_id for f in shown], "final")
        telemetry.latency_ms = marks

        return RunState(
            outcome=SearchOutcome(
                request_id=request.request_id,
                decision=outcome.decision,
                families=shown,
                clarification=outcome.clarification,
                warnings=telemetry.warnings,
                catalog_version=self.corpus.catalog_version,
                model_bundle_version=telemetry.model_bundle_version,
                latency_ms=(time.perf_counter() - started) * 1000,
                telemetry=telemetry,
                decision_detail=outcome,
            ),
            # Pre-truncation and pre-suppression: what the ranker actually
            # produced, which is what a feature generator must see.
            plan=plan, risk=risk, union=union, shortlist=shortlist,
            rerank=rerank, fusion=fusion, families=families,
            decision_result=outcome, universe=universe,
        )
