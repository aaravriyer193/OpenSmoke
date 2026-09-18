"""opensmoke scan | eval"""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

from rich.console import Console

from . import __version__
from .diagnose import Diagnoser, DiagnosisError
from .evaluate import evaluate
from .judges import JudgeError, make_judge
from .loaders import load
from .pipeline import DISCLOSED, SILENT, ScanReport, Settings, scan
from .report import dumps, post_webhook, print_report, to_markdown

console = Console(stderr=True)


def _common(p: argparse.ArgumentParser) -> None:
    p.add_argument("paths", nargs="+", help="trace files, directories, or an OpenBoffin .db")
    p.add_argument("--judge", choices=["jev", "heuristic"], default="jev",
                   help="jev needs TYPESAFE_API_KEY; heuristic is an offline regex baseline")
    p.add_argument("--model", default="jev-1.13", help="TypeSafe model (default jev-1.13)")
    p.add_argument("--threshold", type=float, default=0.7, help="env_broken probability that flags a step")
    p.add_argument("--concurrency", type=int, default=8)
    p.add_argument("--max-steps", type=int, default=200, help="newest steps judged per run")
    p.add_argument("--runs", help="OpenBoffin run ids, comma separated")


async def _scan(args: argparse.Namespace, diagnose: bool) -> ScanReport:
    run_ids = [int(x) for x in args.runs.split(",")] if args.runs else None
    traces = load(args.paths, run_ids)
    if not traces:
        raise SystemExit("no traces found")
    judge = make_judge(args.judge, args.model, args.concurrency)
    diagnoser = None
    if diagnose:
        try:
            diagnoser = Diagnoser(model=getattr(args, "llm_model", None))
        except DiagnosisError as e:
            console.print(f"[yellow]diagnosis off:[/] {e}")
    settings = Settings(threshold=args.threshold, max_steps=args.max_steps, diagnose=diagnoser is not None)
    console.print(f"[dim]judging {len(traces)} runs with {judge.name}…[/]")
    try:
        return await scan(traces, judge, settings, diagnoser)
    finally:
        await judge.aclose()
        if diagnoser:
            await diagnoser.aclose()


def cmd_scan(args: argparse.Namespace) -> int:
    report = asyncio.run(_scan(args, diagnose=not args.no_diagnose))
    print_report(report, console, verbose=args.verbose)
    if args.json:
        Path(args.json).write_text(dumps(report), encoding="utf-8")
        console.print(f"[dim]wrote {args.json}[/]")
    if args.markdown:
        Path(args.markdown).write_text(to_markdown(report), encoding="utf-8")
        console.print(f"[dim]wrote {args.markdown}[/]")
    if args.webhook:
        post_webhook(args.webhook, report)
    fail = {"never": (), "silent": (SILENT,), "any": (SILENT, DISCLOSED)}[args.fail_on]
    return 1 if any(r.status in fail and not r.dismissed for r in report.results) else 0


def cmd_eval(args: argparse.Namespace) -> int:
    report = asyncio.run(_scan(args, diagnose=args.diagnose))
    ev = evaluate(report)
    labeled = ev.tp + ev.fp + ev.fn + ev.tn
    console.print(
        f"\n[bold]{report.judge}[/] on {labeled} labeled runs: "
        f"precision [bold]{ev.precision:.2f}[/] · recall [bold]{ev.recall:.2f}[/] · F1 [bold]{ev.f1:.2f}[/]"
    )
    console.print(f"  tp {ev.tp}  fp {ev.fp}  fn {ev.fn}  tn {ev.tn}")
    if ev.silent_total:
        console.print(f"  silent vs disclosed right on {ev.silent_right}/{ev.silent_total} caught runs")
    if ev.category_total:
        console.print(f"  category right on {ev.category_right}/{ev.category_total} caught runs")
    console.print(f"  cost: judge ${report.judge_cost_usd:.5f} · diagnosis ${report.llm_cost_usd:.4f}")
    for m in ev.misses:
        console.print(f"  [yellow]•[/] {m}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="opensmoke",
        description="Find agent runs that broke because their environment did.",
    )
    parser.add_argument("--version", action="version", version=f"opensmoke {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    s = sub.add_parser("scan", help="judge traces and report incidents")
    _common(s)
    s.add_argument("--no-diagnose", action="store_true", help="skip the LLM second opinion")
    s.add_argument("--llm-model", help="diagnosis model (default z-ai/glm-5.2 via OpenRouter)")
    s.add_argument("--json", help="write the full report as JSON")
    s.add_argument("--markdown", help="write a Markdown report")
    s.add_argument("--webhook", help="POST a Slack-compatible summary to this URL")
    s.add_argument("--fail-on", choices=["never", "silent", "any"], default="never",
                   help="exit 1 when runs with this status are found (for CI)")
    s.add_argument("-v", "--verbose", action="store_true", help="also list clean runs")
    s.set_defaults(func=cmd_scan)

    e = sub.add_parser("eval", help="score a judge against labeled traces")
    _common(e)
    e.add_argument("--diagnose", action="store_true", help="let the LLM dismiss false alarms too")
    e.add_argument("--llm-model", help="diagnosis model")
    e.set_defaults(func=cmd_eval)

    args = parser.parse_args(argv)
    try:
        return args.func(args)
    except JudgeError as e:
        console.print(f"[red]{e}[/]")
        return 2


if __name__ == "__main__":
    sys.exit(main())
