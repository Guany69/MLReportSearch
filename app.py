"""Minimal Streamlit UI: one text box -> ranked report-definition cards.

    uv run streamlit run app.py
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

from reportfinder.config import DEFAULT
from reportfinder.ingest.models import IngestMode
from reportfinder.model import ReportFinder, explain_fields, why_matched
from reportfinder.represent import load_or_build

st.set_page_config(page_title="Report Finder", page_icon="🔎", layout="centered")


@st.cache_resource(show_spinner="Building index (first run only)...")
def get_representation(
    mode: str, data_path: str, catalog_path: str, dictionary_path: str
):
    """Built once per process; the encoder stays resident so queries are sub-second.

    Keyed on the ingest inputs so switching mode in the sidebar rebuilds rather
    than silently reusing the other mode's index.
    """
    cfg = DEFAULT.with_overrides(
        ingest_mode=mode,
        data_path=Path(data_path),
        catalog_path=Path(catalog_path),
        field_dictionary_path=Path(dictionary_path),
    )
    return load_or_build(cfg, rebuild=False, verbose=False), cfg


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
            st.metric("confidence", f"{candidate.confidence_pct:.1f}%")

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
            ambiguous = candidate.has_ambiguous_links
            body = "\n".join(f"- {line}" for line in field_lines)
            if ambiguous:
                st.warning(body, icon="⚠️")
            else:
                st.success(body, icon="🧩")


st.title("🔎 Report Finder")
st.caption(
    "Describe what you need in plain English. Answers come from an unsupervised "
    "dual-space product-of-experts model over the report estate."
)

with st.sidebar:
    st.header("Ingestion")
    mode = st.radio(
        "Mode",
        options=[m.value for m in IngestMode],
        index=0,
        help="legacy_single_file reads the Phase 1 Fields column. "
        "phase2_dual_file reconstructs fields from the catalog + field dictionary.",
    )
    if mode == IngestMode.LEGACY_SINGLE_FILE.value:
        data_path = st.text_input("Workbook", value=str(DEFAULT.data_path))
        catalog_path, dictionary_path = str(DEFAULT.catalog_path), str(
            DEFAULT.field_dictionary_path
        )
    else:
        data_path = str(DEFAULT.data_path)
        catalog_path = st.text_input("Report catalog", value=str(DEFAULT.catalog_path))
        dictionary_path = st.text_input(
            "Field dictionary", value=str(DEFAULT.field_dictionary_path)
        )

    st.header("Model knobs")
    alpha = st.slider("α — dense vs LSA", 0.0, 1.0, DEFAULT.alpha, 0.05,
                      help="1.0 = dense only, 0.0 = LSA only")
    t_dense = st.slider("T_d — dense temperature", 0.01, 0.30, DEFAULT.t_dense, 0.01)
    t_lsa = st.slider("T_l — LSA temperature", 0.01, 0.30, DEFAULT.t_lsa, 0.01)
    tau = st.slider("τ — confidence threshold", 0.0, 0.6, DEFAULT.tau, 0.01)
    delta = st.slider("δ — margin threshold", 0.0, 0.2, DEFAULT.delta, 0.01)
    top_k = st.number_input("top-k when ambiguous", 1, 20, DEFAULT.top_k)
    field_expert = st.checkbox("Enable field-term expert", value=DEFAULT.use_field_expert)
    st.divider()
    st.caption(
        "These are report **definitions**, not result-sets. The app identifies "
        "which report answers your question; it never fabricates report data."
    )

_required = (
    [(data_path, "Workbook")]
    if mode == IngestMode.LEGACY_SINGLE_FILE.value
    else [(catalog_path, "Report catalog"), (dictionary_path, "Field dictionary")]
)
for _path, _label in _required:
    if not Path(_path).exists():
        st.error(f"{_label} not found at `{_path}`. Point the sidebar at a valid file.")
        st.stop()

try:
    rep, base_cfg = get_representation(mode, data_path, catalog_path, dictionary_path)
except Exception as exc:  # noqa: BLE001 - surface build/ingest errors in the UI
    st.error(f"Could not build the index: {exc}")
    st.stop()

cfg = base_cfg.with_overrides(
    alpha=alpha,
    t_dense=t_dense,
    t_lsa=t_lsa,
    tau=tau,
    delta=delta,
    top_k=int(top_k),
    use_field_expert=field_expert,
)
finder = ReportFinder(rep, cfg)

if rep.import_summary is not None:
    with st.expander(f"Import summary — {len(rep)} report families ({mode})"):
        st.code(rep.import_summary.render(), language="text")

with st.form("query_form"):
    query = st.text_input(
        "What do you need?",
        placeholder="show me terminated workers by supervisory organization",
    )
    submitted = st.form_submit_button("Find report", type="primary")

if submitted and query.strip():
    result = finder.query(query)

    if result.confident:
        st.success(
            f"Confident match — top-1 {result.p1:.1%} (≥ τ={cfg.tau:.0%}), "
            f"margin {result.margin:.1%} (≥ δ={cfg.delta:.0%})"
        )
        render_card(result.candidates[0])
    else:
        st.warning(
            f"**Ambiguous — did you mean one of these?** "
            f"Top-1 is {result.p1:.1%} with a {result.margin:.1%} margin, which doesn't "
            f"clear τ={cfg.tau:.0%} / δ={cfg.delta:.0%}. The posterior is spread across "
            f"several reports."
        )
        for i, candidate in enumerate(result.candidates, 1):
            render_card(candidate, rank=i)

    with st.expander("Posterior diagnostics"):
        cols = st.columns(4)
        cols[0].metric("top-1 p₁", f"{result.p1:.3f}")
        cols[1].metric("runner-up p₂", f"{result.p2:.3f}")
        cols[2].metric("margin", f"{result.margin:.3f}")
        cols[3].metric("H(P)", f"{result.entropy_bits:.2f} bits")
        st.caption(
            f"Posterior over {result.n_reports} report families · "
            f"normalized entropy {result.normalized_entropy:.3f} "
            f"(0 = one certain answer, 1 = uniform / no information) · "
            f"max entropy {result.n_reports.bit_length() - 1}–"
            f"{result.n_reports.bit_length()} bits"
        )
        if result.field_expert_used:
            st.caption(f"Field expert active — detected: {', '.join(result.detected_fields)}")
elif submitted:
    st.error("Enter a query first.")
