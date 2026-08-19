"""Streamlit UI: one text box -> a three-way decision and ranked report families.

    uv run streamlit run app.py

This is on the structured `search(SearchRequest) -> SearchOutcome` contract rather
than the deprecated `query()` adapter, so the decision, the family/instance split,
the grounded clarification, the active fallbacks and the catalog and bundle versions
all reach the screen instead of being flattened away.
"""


#Even if we don't find confident matches, we still want to show the user what we did find
# only in streamlit aapp

from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st
import yaml

from reportfinder.auth import DEVELOPMENT_PRINCIPAL, SearchRequest
from reportfinder.config import from_mapping
from reportfinder.model import ReportFinder, explain_fields, why_matched
from reportfinder.represent import load_or_build

st.set_page_config(page_title="Report Finder", page_icon="🔎", layout="centered")

# The config the bundle in .bundle/ was built with, and the one every measured
# number in the project was produced against. Serving from the built-in defaults
# instead left rerank on the torch backend while the bundle recorded onnx, which
# the orchestrator reports as configuration drift: same weights, different
# execution path, so a recorded result could not be reproduced from the bundle
# id alone.
CONFIG_PATH = Path(__file__).parent / "configs" / "legacy_generators.yaml"
BASE_CONFIG = from_mapping(yaml.safe_load(CONFIG_PATH.read_text()) or {})


@st.cache_resource(show_spinner="Building index (first run only)...")
def get_finder(data_path: str, top_k: int):
    """Built once per process, keyed on the inputs that change what it returns.

    Streamlit reruns this script top to bottom on every interaction. Without the
    cache the whole representation -- and, in generator mode, every index in the
    bundle -- would be rebuilt on each keystroke.
    """
    cfg = BASE_CONFIG.with_overrides(
        data_path=Path(data_path),
        top_k=int(top_k),
    )
    finder = ReportFinder(load_or_build(cfg, rebuild=False, verbose=False), cfg)
    if cfg.retrieval_mode == "generators":
        # Construct the pipeline eagerly, inside the cache, so the first query does
        # not pay for loading every checkpoint.
        _ = finder.pipeline
    return finder, cfg


def fmt_date(value, missing: bool) -> str:
    if missing or pd.isna(value):
        return "not recorded"
    return pd.Timestamp(value).strftime("%Y-%m-%d")


def render_card(candidate, rank: int | None = None) -> None:
    row = candidate.row
    heading = f"{rank}. {row['title']}" if rank else row["title"]

    with st.container(border=True):
        left, right = st.columns([4, 1])
        with left:
            st.markdown(f"**{heading}**")
            st.caption(
                f"{row['category']} · {row['data_source']} · {row['report_type']}"
                + (f" · tags: {row['tags']}" if str(row["tags"]).strip() else "")
            )
        with right:
            if candidate.cross_encoder_score is not None:
                # A raw logit, labelled as one. It is not a probability and there is
                # no calibration artifact that would make it one.
                st.metric("reranker", f"{candidate.cross_encoder_score:+.2f}")
            elif candidate.instance_id:
                st.metric("reranker", "n/a")
            else:
                st.metric("retrieval share", f"{candidate.confidence_pct:.1f}%")

        if candidate.instance_id:
            st.caption(
                f"instance `{candidate.instance_id}` · admitted via "
                f"`{candidate.admitted_via}`"
            )
        else:
            st.progress(min(candidate.probability * 4, 1.0))

        fields = list(row["fields"])
        st.markdown(f"**Fields ({len(fields)})**")
        st.write(" · ".join(fields))

        prompts = list(row["prompts"])
        st.markdown(f"**Prompts ({len(prompts)})**" if prompts else "**Prompts**")
        st.write(" · ".join(prompts) if prompts else "_none defined_")

        runs = (
            "run count not recorded"
            if bool(row["runs_missing"]) or pd.isna(row["runs"])
            else f"run {int(row['runs'])}×"
        )
        usage = (
            f"{runs} · last run {fmt_date(row['last_run'], bool(row['last_run_missing']))}"
            f" · updated {fmt_date(row['last_updated'], bool(row['last_updated_missing']))}"
        )
        if int(row["family_size"]) > 1:
            usage += f" · {row['family_size']} identical copies in estate"
        if str(row["shared"]).strip():
            usage += f" · shared: {row['shared']}"
        st.caption(usage)

        st.info(f"**Why it matched** — {why_matched(candidate)}", icon="🔍")

        field_lines = explain_fields(candidate)
        if field_lines:
            body = "\n".join(f"- {line}" for line in field_lines)
            if candidate.has_ambiguous_links:
                st.warning(body, icon="⚠️")
            else:
                st.success(body, icon="🧩")


st.title("🔎 Report Finder")
st.caption(
    "Describe what you need in plain English. Independent retrievers nominate "
    "candidates, a cross-encoder reranks the full shortlist, and the system "
    "answers, asks, or declines."
)

with st.sidebar:
    st.header("Ingestion")
    data_path = st.text_input("Report workbook", value=str(BASE_CONFIG.data_path))

    st.header("Search")
    top_k = st.number_input("results to show", 1, 20, BASE_CONFIG.top_k)
    st.divider()
    st.caption(
        "These are report **definitions**, not result-sets. The app identifies "
        "which report answers your question; it never fabricates report data."
    )

if not Path(data_path).exists():
    st.error(f"Report workbook not found at `{data_path}`. Point the sidebar at a valid file.")
    st.stop()

try:
    finder, cfg = get_finder(data_path, int(top_k))
except Exception as exc:  # noqa: BLE001 - surface build/ingest errors in the UI
    st.error(f"Could not build the index: {exc}")
    st.stop()

if finder.rep.import_summary is not None:
    with st.expander(f"Import summary — {len(finder.rep)} report families"):
        st.code(finder.rep.import_summary.render(), language="text")

with st.form("query_form"):
    query = st.text_input(
        "What do you need?",
        placeholder="show me terminated workers by supervisory organization",
    )
    submitted = st.form_submit_button("Find report", type="primary")

if submitted and query.strip():
    if cfg.retrieval_mode != "generators":
        # Deprecated compatibility path, kept reachable for ablation.
        result = finder.query(query)
        st.warning(f"Running the deprecated `{cfg.retrieval_mode}` runtime.")
        for i, candidate in enumerate(result.candidates, 1):
            render_card(candidate, rank=i)
        st.stop()

    # `run_traced` rather than `search`, only so the abstained-on candidates are
    # reachable below. It returns the same outcome plus the pre-suppression family
    # list; the served contract is untouched.
    state = finder.pipeline.run_traced(
        SearchRequest(query, DEVELOPMENT_PRINCIPAL, top_k=int(top_k))
    )
    outcome = state.outcome

    decision = outcome.decision.value
    detail = outcome.decision_detail
    if decision == "RETURN_RESULTS":
        st.success("Answerable")
    elif decision == "ASK_CLARIFICATION":
        st.warning("Needs clarification")
    else:
        st.error("No confident match")

    for reason in (detail.reasons if detail is not None else ()):
        st.caption(reason)

    if outcome.clarification is not None:
        st.markdown(f"**{outcome.clarification.question}**")
        for option in outcome.clarification.options:
            st.markdown(f"- {option}")

    for warning in outcome.warnings:
        st.caption(f"⚠️ {warning}")

    if outcome.active_fallbacks:
        # Disclosed on every response. A fallback nobody can see is a fallback that
        # gets mistaken for a trained model.
        st.info(
            "Running with fallbacks: " + ", ".join(sorted(outcome.active_fallbacks)),
            icon="🧪",
        )

    # NO_CONFIDENT_MATCH deliberately carries no families: a ranked list reads as an
    # answer whatever status sits above it.
    for i, family in enumerate(outcome.families, 1):
        render_card(finder.candidate_for(family), rank=i)

    with st.expander("Diagnostics"):
        cols = st.columns(3)
        cols[0].metric("families", len(outcome.families))
        cols[1].metric("latency", f"{outcome.latency_ms:.0f} ms")
        cols[2].metric("risk band", outcome.telemetry.risk.get("risk_band", "?"))
        st.caption(
            f"catalog `{outcome.catalog_version}` · bundle "
            f"`{outcome.model_bundle_version or 'none'}` · request "
            f"`{outcome.request_id}`"
        )
        st.json(outcome.telemetry.as_dict(), expanded=False)
elif submitted:
    st.error("Enter a query first.")
