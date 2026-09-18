"""Score a judge against labeled traces.

A trace is labeled with `"expect": {"smoke": bool, "silent": bool, "category": str | [str]}`.
`smoke` means an environment failure reached the run (silent or disclosed);
recovered and clean runs are `smoke: false`.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .pipeline import RunResult, ScanReport


@dataclass
class EvalResult:
    tp: int = 0
    fp: int = 0
    fn: int = 0
    tn: int = 0
    silent_right: int = 0
    silent_total: int = 0
    category_right: int = 0
    category_total: int = 0
    misses: list[str] = field(default_factory=list)

    @property
    def precision(self) -> float:
        return self.tp / (self.tp + self.fp) if self.tp + self.fp else 0.0

    @property
    def recall(self) -> float:
        return self.tp / (self.tp + self.fn) if self.tp + self.fn else 0.0

    @property
    def f1(self) -> float:
        p, r = self.precision, self.recall
        return 2 * p * r / (p + r) if p + r else 0.0


def evaluate(report: ScanReport) -> EvalResult:
    ev = EvalResult()
    labeled: list[RunResult] = [r for r in report.results if r.trace.expect is not None]
    for r in labeled:
        expect = r.trace.expect or {}
        want, got = bool(expect.get("smoke")), r.smoky and not r.dismissed
        if want and got:
            ev.tp += 1
            if "silent" in expect:
                ev.silent_total += 1
                ev.silent_right += (r.status == "silent") == bool(expect["silent"])
            if expect.get("category"):
                # A list means several labels are defensible (a missing AGENTS.md
                # is both a missing file and missing instructions).
                ok = expect["category"] if isinstance(expect["category"], list) else [expect["category"]]
                ev.category_total += 1
                ev.category_right += r.category in ok
                if r.category not in ok:
                    ev.misses.append(f"{r.trace.id}: category {r.category}, expected {'/'.join(ok)}")
        elif got:
            ev.fp += 1
            ev.misses.append(f"{r.trace.id}: false alarm ({r.status}, {r.category})")
        elif want:
            ev.fn += 1
            ev.misses.append(f"{r.trace.id}: missed ({r.status}), expected {expect.get('category', 'smoke')}")
        else:
            ev.tn += 1
    return ev
