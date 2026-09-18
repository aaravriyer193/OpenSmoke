"""Three ways out: a terminal summary, Markdown for an issue or PR, JSON for machines."""

from __future__ import annotations

import json
from typing import Any

import httpx
from rich.console import Console
from rich.table import Table
from rich.text import Text

from .pipeline import CLEAN, DISCLOSED, RECOVERED, SILENT, RunResult, ScanReport

STATUS_STYLE = {SILENT: "bold red", DISCLOSED: "yellow", RECOVERED: "cyan", CLEAN: "green"}


def _counts(report: ScanReport) -> dict[str, int]:
    counts = {s: 0 for s in (SILENT, DISCLOSED, RECOVERED, CLEAN)}
    for r in report.results:
        counts[r.status] += 1
    counts["dismissed"] = sum(r.dismissed for r in report.results)
    return counts


def _excerpt(r: RunResult, width: int = 160) -> str:
    if not r.flagged:
        return ""
    top = max(r.flagged, key=lambda f: f["env_broken"])
    step = r.trace.steps[top["index"]]
    text = " ".join(step.content.split())
    if len(text) <= width:
        return text
    # Tool output ends in the error; an agent says what went wrong up front.
    return text[:width] + "…" if step.kind == "assistant" else "…" + text[-width:]


def to_dict(report: ScanReport) -> dict[str, Any]:
    return {
        "summary": {
            "runs": len(report.results),
            "steps_judged": report.steps_judged,
            **_counts(report),
            "judge": report.judge,
            "judge_cost_usd": round(report.judge_cost_usd, 6),
            "llm_cost_usd": round(report.llm_cost_usd, 6),
            "notes": report.notes,
        },
        "incidents": [
            {
                "category": c.category,
                "missing": c.fingerprint,
                "runs": [r.trace.id for r in c.runs],
                "silent": c.silent,
                "fix": next((r.diagnosis.fix for r in c.runs if r.diagnosis and r.diagnosis.fix), ""),
            }
            for c in report.clusters
        ],
        "runs": [
            {
                "id": r.trace.id,
                "status": r.status,
                "category": r.category,
                "fingerprint": r.fingerprint,
                "flagged_steps": r.flagged,
                "signals": vars(r.signals) if r.signals else None,
                "diagnosis": r.diagnosis.to_dict() if r.diagnosis else None,
                "errors": r.errors,
            }
            for r in report.results
        ],
    }


def print_report(report: ScanReport, console: Console | None = None, verbose: bool = False) -> None:
    console = console or Console()
    c = _counts(report)
    console.print(
        f"\n[bold]OpenSmoke[/] scanned [bold]{len(report.results)}[/] runs "
        f"({report.steps_judged} steps) with [bold]{report.judge}[/]: "
        f"[bold red]{c[SILENT]} silent[/] · [yellow]{c[DISCLOSED]} disclosed[/] · "
        f"[cyan]{c[RECOVERED]} recovered[/] · [green]{c[CLEAN]} clean[/]"
        + (f" · {c['dismissed']} dismissed by diagnosis" if c["dismissed"] else "")
    )
    console.print(
        f"[dim]cost: judge ${report.judge_cost_usd:.5f} · diagnosis ${report.llm_cost_usd:.4f}[/]"
    )
    for note in report.notes:
        console.print(Text(f"note: {note}", style="yellow"))
    console.print()

    if report.clusters:
        t = Table(title="Incidents", title_justify="left", show_lines=False)
        t.add_column("category")
        t.add_column("missing / broken")
        t.add_column("runs", justify="right")
        t.add_column("silent", justify="right")
        t.add_column("fix")
        for cl in report.clusters:
            fix = next((r.diagnosis.fix for r in cl.runs if r.diagnosis and r.diagnosis.fix), "")
            t.add_row(cl.category, cl.fingerprint, str(len(cl.runs)), str(cl.silent), fix)
        console.print(t)

    for r in report.results:
        if r.status == CLEAN and not verbose:
            continue
        style = STATUS_STYLE[r.status]
        console.print(f"\n[{style}]● {r.status.upper()}[/] [bold]{r.trace.id}[/]  {r.category}")
        for f in r.flagged:
            step = r.trace.steps[f["index"]]
            console.print(
                f"  step {f['index']:>3} {step.label:<24} env={f['env_broken']:.2f} "
                f"workaround={f['workaround']:.2f}  {f['category']}"
            )
        if r.flagged:
            # Trace text is data: never let it be parsed as markup.
            console.print(Text("  " + _excerpt(r), style="dim"), highlight=False)
        if r.diagnosis:
            d = r.diagnosis
            console.print(f"  [bold]{d.title}[/] ({d.verdict}, {d.confidence:.0%})")
            if d.root_cause:
                console.print(f"  cause: {d.root_cause}")
            if d.fix:
                console.print(f"  fix:   {d.fix}")
        for e in r.errors:
            console.print(f"  [dim red]error: {e}[/]")


def to_markdown(report: ScanReport) -> str:
    c = _counts(report)
    lines = [
        "# OpenSmoke report",
        "",
        (
            f"Scanned **{len(report.results)}** runs ({report.steps_judged} steps) with `{report.judge}`: "
            f"**{c[SILENT]} silent**, {c[DISCLOSED]} disclosed, {c[RECOVERED]} recovered, {c[CLEAN]} clean."
        ),
        "",
    ]
    if report.clusters:
        lines += ["## Incidents", "", "| Category | Missing / broken | Runs | Silent | Fix |", "|---|---|---|---|---|"]
        for cl in report.clusters:
            fix = next((r.diagnosis.fix for r in cl.runs if r.diagnosis and r.diagnosis.fix), "")
            lines.append(f"| {cl.category} | `{cl.fingerprint}` | {len(cl.runs)} | {cl.silent} | {fix} |")
        lines.append("")
    smoky = [r for r in report.results if r.status in (SILENT, DISCLOSED)]
    if smoky:
        lines += ["## Runs", ""]
    for r in smoky:
        lines.append(f"### {r.status.upper()}: `{r.trace.id}` ({r.category})")
        if r.diagnosis:
            d = r.diagnosis
            lines += [f"**{d.title}** ({d.verdict}, {d.confidence:.0%})", "", d.what_happened, ""]
            if d.fix:
                lines += [f"**Fix:** {d.fix}", ""]
        steps = ", ".join(f"{f['index']} ({f['category']}, {f['env_broken']:.2f})" for f in r.flagged)
        lines += [f"Flagged steps: {steps}", "", "```", _excerpt(r, 600), "```", ""]
    return "\n".join(lines)


def post_webhook(url: str, report: ScanReport) -> None:
    """Slack-compatible: a `text` field, plus the full JSON for anything else listening."""
    c = _counts(report)
    lines = [f"*OpenSmoke*: {c[SILENT]} silent, {c[DISCLOSED]} disclosed of {len(report.results)} runs"]
    for cl in report.clusters[:10]:
        lines.append(f"• {cl.category} `{cl.fingerprint}`: {len(cl.runs)} runs ({cl.silent} silent)")
    httpx.post(url, json={"text": "\n".join(lines), "opensmoke": to_dict(report)}, timeout=30).raise_for_status()


def dumps(report: ScanReport) -> str:
    return json.dumps(to_dict(report), indent=2, ensure_ascii=False)
