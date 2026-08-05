"""Terminal rendering of results.

The workbook holds report *definitions*, not result-sets, so "the report" here is
the definition: title, where it lives, what it selects, how it's filtered, and how
much it's used. Nothing in this module invents report data rows.
"""

from __future__ import annotations

import pandas as pd

from .model import Candidate, Result, explain_fields, why_matched


def _fmt_date(value, missing: bool) -> str:
    if missing or pd.isna(value):
        return "not recorded"
    return pd.Timestamp(value).strftime("%Y-%m-%d")


def _fmt_runs(row) -> str:
    """Usage line. A missing run count is reported as missing, never as zero."""
    if bool(row["runs_missing"]) or pd.isna(row["runs"]):
        runs = "run count not recorded"
    else:
        runs = f"run {int(row['runs'])}x"

    last_run = _fmt_date(row["last_run"], bool(row["last_run_missing"]))
    updated = _fmt_date(row["last_updated"], bool(row["last_updated_missing"]))
    return f"{runs} | last run {last_run} | updated {updated}"


def format_candidate(candidate: Candidate, rank: int | None = None) -> str:
    row = candidate.row
    prefix = f"{rank}. " if rank is not None else ""
    lines: list[str] = []

    if candidate.instance_id:
        # Generator architecture: report the reranker's judgement and which
        # instance was selected. "Retrieval share" is a legacy display
        # normalization and means nothing here.
        score = (
            f"cross-encoder {candidate.cross_encoder_score:+.2f}"
            if candidate.cross_encoder_score is not None
            else "not reranked"
        )
        lines.append(f"{prefix}{row['title']}  [{score}]")
        lines.append(f"   instance {candidate.instance_id} | admitted via {candidate.admitted_via}")
    else:
        lines.append(f"{prefix}{row['title']}  [{candidate.confidence_pct:.1f}% retrieval share]")

    meta = f"   {row['category']} | {row['data_source']} | {row['report_type']}"
    if str(row["tags"]).strip():
        meta += f" | tags: {row['tags']}"
    lines.append(meta)

    fields = list(row["fields"])
    shown = "; ".join(fields[:10])
    if len(fields) > 10:
        shown += f"  (+{len(fields) - 10} more)"
    lines.append(f"   Fields ({len(fields)}): {shown}")

    prompts = list(row["prompts"])
    if prompts:
        lines.append(f"   Prompts ({len(prompts)}): {'; '.join(prompts)}")
    else:
        lines.append("   Prompts: none defined")

    usage = _fmt_runs(row)
    if int(row["family_size"]) > 1:
        usage += f" | {row['family_size']} identical copies in estate"
    if str(row["shared"]).strip():
        usage += f" | shared: {row['shared']}"
    lines.append(f"   {usage}")

    lines.append(f"   Why: {why_matched(candidate)}")
    for line in explain_fields(candidate):
        lines.append(f"        {line}")
    return "\n".join(lines)


def format_result(result: Result, show_diagnostics: bool = False) -> str:
    out: list[str] = []
    out.append(f'Query: "{result.query}"')

    if result.outcome is not None:
        # Generator architecture. The decision, the grounded clarification and any
        # active fallback are all surfaced -- a fallback that is not visible is a
        # fallback that gets mistaken for a trained model.
        label = result.decision.replace("_", " ").title()
        out.append(f"{label}\n")
        for reason in result.outcome.decision_detail.reasons:
            out.append(f"   {reason}")
        clarification = result.clarification
        if clarification is not None:
            out.append(f"\n   {clarification.question}")
            for option in clarification.options:
                out.append(f"     - {option}")
        if result.fallbacks:
            out.append("\n   Running with fallbacks: " + ", ".join(sorted(result.fallbacks)))
        for warning in result.outcome.warnings:
            out.append(f"   ! {warning}")
        out.append("")

    if result.outcome is None and result.answerability is not None:
        label = result.answerability.outcome.value.replace("_", " ").title()
        qualification = "calibrated" if result.answerability.calibrated else "uncalibrated"
        out.append(f"{label} (answerability score {result.answerability.score:.2f}, {qualification})\n")
        for reason in result.answerability.reasons:
            out.append(f"   {reason}")
        if result.explanation is not None:
            concepts = result.explanation.interpreted_concepts
            if concepts:
                rendered = ", ".join(
                    f"{item.evidence!r} → {item.canonical}"
                    + (" (required)" if item.mandatory else "")
                    for item in concepts
                )
                out.append(f"   Interpreted: {rendered}")
            for typo in result.explanation.typo_corrections:
                out.append(
                    f"   {typo.surface!r} looks like a typo for {typo.corrected!r}; "
                    "results include the correction."
                )
            for warning in result.explanation.linkage_warnings:
                out.append(f"   Metadata warning: {warning}")
            for reason in result.explanation.ambiguity_reasons:
                out.append(f"   Ambiguity: {reason}")
        for i, candidate in enumerate(result.candidates, 1):
            out.append(format_candidate(candidate, None if result.confident else i))
            out.append("")
    elif result.outcome is not None:
        # Generator architecture: always ranked, never collapsed to one answer.
        if not result.candidates:
            out.append("No reports to show.")
        for i, candidate in enumerate(result.candidates, 1):
            out.append(format_candidate(candidate, i))
            out.append("")
    elif result.confident:
        out.append(
            f"Legacy threshold match (retrieval share={result.p1:.1%} >= tau, "
            f"margin={result.margin:.1%}):\n"
        )
        out.append(format_candidate(result.candidates[0]))
    else:
        out.append(
            f"Legacy ambiguous result - did you mean one of these? "
            f"(top-1 {result.p1:.1%}, margin {result.margin:.1%})\n"
        )
        for i, candidate in enumerate(result.candidates, 1):
            out.append(format_candidate(candidate, rank=i))
            out.append("")

    if show_diagnostics:
        out.append("")
        out.append(
            f"   normalized ranking mass over candidates from {result.n_reports} report families | "
            f"H(P)={result.entropy_bits:.2f} bits "
            f"(normalized {result.normalized_entropy:.3f}) | "
            f"p1={result.p1:.4f} p2={result.p2:.4f} margin={result.margin:.4f}"
        )
        if result.field_expert_used:
            out.append(f"   field expert active: {', '.join(result.detected_fields)}")

    return "\n".join(out)
