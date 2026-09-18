"""Trace in, incidents out.

    every judged step ──Jev──▶ flagged steps ──Jev──▶ run status
                                                  │
                     silent / disclosed runs ──LLM──▶ diagnosis ──▶ clusters

Jev sees every step because it is cheap enough to; the LLM sees only what Jev
flagged, because it is not.
"""

from __future__ import annotations

import asyncio
import logging
import re
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any

from .diagnose import Diagnoser, Diagnosis, DiagnosisAuthError
from .judges import Judge, JudgeError, RunSignals, StepSignals
from .trace import JUDGED_KINDS, Step, Trace

log = logging.getLogger("opensmoke")

CLEAN, RECOVERED, DISCLOSED, SILENT = "clean", "recovered", "disclosed", "silent"


@dataclass
class Settings:
    threshold: float = 0.7  # env_broken probability that flags a step
    max_steps: int = 200  # per run; the newest steps are kept
    diagnose: bool = True
    escalate: tuple[str, ...] = (SILENT, DISCLOSED)


@dataclass
class RunResult:
    trace: Trace
    steps: dict[int, StepSignals] = field(default_factory=dict)
    flagged: list[dict[str, Any]] = field(default_factory=list)
    signals: RunSignals | None = None
    status: str = CLEAN
    category: str = ""
    fingerprint: str = ""
    diagnosis: Diagnosis | None = None
    errors: list[str] = field(default_factory=list)

    @property
    def smoky(self) -> bool:
        return self.status in (SILENT, DISCLOSED)

    @property
    def dismissed(self) -> bool:
        return self.diagnosis is not None and self.diagnosis.verdict != "environment"


@dataclass
class Cluster:
    category: str
    fingerprint: str
    runs: list[RunResult]

    @property
    def silent(self) -> int:
        return sum(r.status == SILENT for r in self.runs)


@dataclass
class ScanReport:
    results: list[RunResult]
    clusters: list[Cluster]
    judge: str
    judge_cost_usd: float
    llm_cost_usd: float
    steps_judged: int
    notes: list[str] = field(default_factory=list)


# Pull the concrete missing thing out of an error, so 40 runs that all lack
# the same key become one incident instead of 40.
_FINGERPRINTS = [
    re.compile(r"(\S+): command not found"),
    re.compile(r"command not found: (\S+)"),
    re.compile(r"No module named '([\w.]+)'"),
    re.compile(r"Cannot find module '([^']+)'"),
    re.compile(r"KeyError: '([A-Z][A-Z0-9_]+)'"),
    re.compile(r"\b([A-Z][A-Z0-9_]*(?:KEY|TOKEN|SECRET))\b"),
    re.compile(r"No such file or directory: '([^']+)'"),
    re.compile(r"([\w./-]+): No such file or directory"),
    re.compile(r"Could not resolve host:? '?([\w.-]+)"),
    re.compile(r"(AGENTS\.md|CLAUDE\.md)"),
]


def fingerprint(text: str) -> str:
    for pattern in _FINGERPRINTS:
        m = pattern.search(text)
        if m:
            return m.group(1).strip("'\"`").lower()
    return ""


def _status(signals: RunSignals, flagged: list[dict[str, Any]], threshold: float) -> str:
    if not flagged:
        return CLEAN
    worked_around = any(f["workaround"] >= threshold for f in flagged)
    if signals.recovered >= threshold and not worked_around:
        return RECOVERED
    if signals.discloses_problem >= 0.5:
        return DISCLOSED
    # Flagged, not disclosed: whatever the final answer claims, the user was
    # not told. That is the failure OpenSmoke exists to catch.
    return SILENT


async def judge_run(trace: Trace, judge: Judge, settings: Settings) -> RunResult:
    result = RunResult(trace=trace)
    steps: list[Step] = [s for s in trace.steps if s.kind in JUDGED_KINDS][-settings.max_steps :]

    async def one(step: Step) -> None:
        try:
            result.steps[step.index] = await judge.step(trace, step)
        except JudgeError:
            raise
        except Exception as e:  # one bad step must not sink the run
            result.errors.append(f"step {step.index}: {e}")

    await asyncio.gather(*(one(s) for s in steps))

    for index in sorted(result.steps):
        sig = result.steps[index]
        if sig.env_broken >= settings.threshold:
            result.flagged.append(
                {
                    "index": index,
                    "env_broken": sig.env_broken,
                    "category": sig.category,
                    "category_confidence": sig.category_confidence,
                    "workaround": sig.workaround,
                }
            )
    # An agent narrating a workaround is itself a flag, even if no tool output
    # looked broken: "I'll assume the default config" hides a missing file.
    for index in sorted(result.steps):
        sig = result.steps[index]
        if sig.workaround >= settings.threshold and index not in {f["index"] for f in result.flagged}:
            result.flagged.append(
                {
                    "index": index,
                    "env_broken": sig.env_broken,
                    "category": sig.category if sig.category != "none" else "missing_context",
                    "category_confidence": sig.category_confidence,
                    "workaround": sig.workaround,
                }
            )
    result.flagged.sort(key=lambda f: f["index"])

    if not result.flagged:
        return result

    problems = [
        f"step {f['index']} ({f['category']}): {trace.steps[f['index']].content[-300:]}"
        for f in result.flagged
    ]
    try:
        result.signals = await judge.run(trace, problems)
    except JudgeError:
        raise
    except Exception as e:
        result.errors.append(f"run: {e}")
        result.signals = RunSignals(claims_success=0.5, discloses_problem=0.0, recovered=0.0)

    result.status = _status(result.signals, result.flagged, settings.threshold)
    top = max(result.flagged, key=lambda f: f["env_broken"])
    result.category = top["category"]
    evidence = "\n".join(trace.steps[f["index"]].content for f in result.flagged)
    result.fingerprint = fingerprint(evidence)
    return result


def cluster(results: list[RunResult]) -> list[Cluster]:
    groups: dict[tuple[str, str], list[RunResult]] = defaultdict(list)
    for r in results:
        if not r.smoky or r.dismissed:
            continue
        fp = (r.diagnosis.missing.lower() if r.diagnosis and r.diagnosis.missing else r.fingerprint) or "unknown"
        groups[(r.category, fp)].append(r)
    clusters = [Cluster(c, fp, runs) for (c, fp), runs in groups.items()]
    clusters.sort(key=lambda c: (c.silent, len(c.runs)), reverse=True)
    return clusters


async def scan(
    traces: list[Trace],
    judge: Judge,
    settings: Settings | None = None,
    diagnoser: Diagnoser | None = None,
    on_progress: Any = None,
) -> ScanReport:
    settings = settings or Settings()

    async def one(trace: Trace) -> RunResult:
        r = await judge_run(trace, judge, settings)
        if on_progress:
            on_progress(r)
        return r

    # The judge bounds requests in flight, so every run can start at once.
    results = list(await asyncio.gather(*(one(t) for t in traces)))

    notes: list[str] = []
    todo = [r for r in results if r.status in settings.escalate] if diagnoser else []

    async def diag(r: RunResult) -> None:
        try:
            r.diagnosis = await diagnoser.diagnose(r.trace, r.flagged)
        except DiagnosisAuthError:
            raise
        except Exception as e:
            r.errors.append(f"diagnosis: {e}")

    if todo:
        # One call first: a rejected key should cost one request and one
        # message, not one per flagged run.
        try:
            await diag(todo[0])
            await asyncio.gather(*(diag(r) for r in todo[1:]))
        except DiagnosisAuthError as e:
            notes.append(f"diagnosis skipped, key rejected: {e}")

    return ScanReport(
        notes=notes,
        results=results,
        clusters=cluster(results),
        judge=judge.name,
        judge_cost_usd=judge.cost_usd,
        llm_cost_usd=diagnoser.cost_usd if diagnoser else 0.0,
        steps_judged=sum(len(r.steps) for r in results),
    )
