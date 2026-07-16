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

    lines.append(f"{prefix}{row['title']}  [{candidate.confidence_pct:.1f}% confidence]")

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

    if result.confident:
        out.append(
            f"Confident match (p1={result.p1:.1%} >= tau, "
            f"margin={result.margin:.1%}):\n"
        )
        out.append(format_candidate(result.candidates[0]))
    else:
        out.append(
            f"Ambiguous - did you mean one of these? "
            f"(top-1 {result.p1:.1%}, margin {result.margin:.1%})\n"
        )
        for i, candidate in enumerate(result.candidates, 1):
            out.append(format_candidate(candidate, rank=i))
            out.append("")

    if show_diagnostics:
        out.append("")
        out.append(
            f"   posterior over {result.n_reports} report families | "
            f"H(P)={result.entropy_bits:.2f} bits "
            f"(normalized {result.normalized_entropy:.3f}) | "
            f"p1={result.p1:.4f} p2={result.p2:.4f} margin={result.margin:.4f}"
        )
        if result.field_expert_used:
            out.append(f"   field expert active: {', '.join(result.detected_fields)}")

    return "\n".join(out)
