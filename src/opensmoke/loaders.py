"""Turn what agents log into `Trace`s.

Supported:
  * OpenSmoke JSON   {"id", "task", "steps": [...], "final_output"}
  * chat messages    {"messages": [...]} in OpenAI or Anthropic shape
  * OpenBoffin       its SQLite database plus the experiments/ folder beside it

A file may hold one trace, a list of them, {"traces": [...]}, or JSONL.
"""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from .trace import Step, Trace


def _text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        # Anthropic/OpenAI content blocks: keep the text, serialize the rest.
        out = []
        for block in value:
            if isinstance(block, dict) and block.get("type") == "text":
                out.append(str(block.get("text", "")))
            elif isinstance(block, dict) and "content" in block:
                out.append(_text(block["content"]))
            else:
                out.append(json.dumps(block, ensure_ascii=False))
        return "\n".join(out)
    return json.dumps(value, ensure_ascii=False)


def from_steps(obj: dict[str, Any], default_id: str) -> Trace:
    steps = []
    for i, raw in enumerate(obj.get("steps", [])):
        kind = raw.get("type") or raw.get("kind") or "log"
        if kind not in ("user", "assistant", "tool_call", "tool_result", "log"):
            kind = "log"
        steps.append(
            Step(
                index=i,
                kind=kind,
                content=_text(raw.get("content", raw.get("output", raw.get("input", "")))),
                name=str(raw.get("name") or raw.get("tool") or ""),
                exit_code=raw.get("exit_code"),
                meta={k: v for k, v in raw.items() if k not in ("type", "kind", "content", "name", "tool", "exit_code")},
            )
        )
    final = obj.get("final_output")
    if final is None:
        last = next((s for s in reversed(steps) if s.kind == "assistant"), None)
        final = last.content if last else ""
    return Trace(
        id=str(obj.get("id", default_id)),
        task=_text(obj.get("task", "")),
        steps=steps,
        final_output=_text(final),
        meta=obj.get("meta", {}),
        expect=obj.get("expect"),
    )


def from_messages(obj: dict[str, Any], default_id: str) -> Trace:
    """OpenAI chat messages (tool_calls / role=tool) or Anthropic blocks (tool_use / tool_result)."""
    steps: list[Step] = []
    tool_names: dict[str, str] = {}
    task = _text(obj.get("task", ""))

    def add(kind: str, content: str, name: str = "", exit_code: int | None = None) -> None:
        steps.append(Step(len(steps), kind, content, name, exit_code))  # type: ignore[arg-type]

    for m in obj.get("messages", []):
        role, content = m.get("role"), m.get("content")
        if role == "system":
            continue
        if role == "tool":  # OpenAI tool result
            name = m.get("name") or tool_names.get(m.get("tool_call_id", ""), "")
            add("tool_result", _text(content), name)
            continue
        blocks = content if isinstance(content, list) else [{"type": "text", "text": content or ""}]
        for b in blocks:
            if not isinstance(b, dict):
                continue
            t = b.get("type")
            if t == "text" and str(b.get("text", "")).strip():
                if role == "user":
                    task = task or str(b["text"])
                    add("user", str(b["text"]))
                else:
                    add("assistant", str(b["text"]))
            elif t == "tool_use":
                tool_names[b.get("id", "")] = b.get("name", "")
                add("tool_call", _text(b.get("input")), b.get("name", ""))
            elif t == "tool_result":
                add(
                    "tool_result",
                    _text(b.get("content")),
                    tool_names.get(b.get("tool_use_id", ""), ""),
                    1 if b.get("is_error") else None,
                )
        for call in m.get("tool_calls") or []:  # OpenAI tool calls
            fn = call.get("function", {})
            tool_names[call.get("id", "")] = fn.get("name", "")
            add("tool_call", _text(fn.get("arguments")), fn.get("name", ""))

    final = obj.get("final_output")
    if final is None:
        last = next((s for s in reversed(steps) if s.kind == "assistant"), None)
        final = last.content if last else ""
    return Trace(
        id=str(obj.get("id", default_id)),
        task=task,
        steps=steps,
        final_output=_text(final),
        meta=obj.get("meta", {}),
        expect=obj.get("expect"),
    )


def from_dict(obj: dict[str, Any], default_id: str) -> Trace:
    if "messages" in obj and "steps" not in obj:
        return from_messages(obj, default_id)
    return from_steps(obj, default_id)


def _from_json_file(path: Path) -> Iterable[Trace]:
    text = path.read_text(encoding="utf-8")
    if path.suffix == ".jsonl":
        for n, line in enumerate(text.splitlines()):
            if line.strip():
                yield from_dict(json.loads(line), f"{path.stem}:{n}")
        return
    data = json.loads(text)
    if isinstance(data, dict) and "traces" in data:
        data = data["traces"]
    if isinstance(data, list):
        for n, obj in enumerate(data):
            yield from_dict(obj, f"{path.stem}:{n}")
    else:
        yield from_dict(data, path.stem)


# ------------------------------------------------------------------ OpenBoffin


def _read(path: Path, limit: int = 20_000) -> str:
    try:
        return path.read_text(encoding="utf-8", errors="replace")[:limit]
    except OSError:
        return ""


def from_openboffin(db_path: Path, run_ids: list[int] | None = None) -> list[Trace]:
    """One trace per OpenBoffin run.

    Event rows keep only a ~100 character summary, so experiment attempts are
    rebuilt from experiments/exp<N>/try<K>/ beside the database: the code the
    agent wrote, and the stdout/stderr the sandbox handed back.
    """
    con = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    con.row_factory = sqlite3.Row
    exp_dir = db_path.parent / "experiments"
    traces = []
    try:
        runs = con.execute("SELECT * FROM runs ORDER BY id").fetchall()
        for run in runs:
            if run_ids and run["id"] not in run_ids:
                continue
            steps: list[Step] = []

            def add(kind: str, content: str, name: str = "", exit_code: int | None = None, steps: list[Step] = steps, **meta: Any) -> None:
                steps.append(Step(len(steps), kind, content, name, exit_code, meta))  # type: ignore[arg-type]

            experiments = {
                e["id"]: e for e in con.execute("SELECT * FROM experiments WHERE run_id=?", (run["id"],))
            }
            final = ""
            for ev in con.execute("SELECT * FROM events WHERE run_id=? ORDER BY id", (run["id"],)):
                kind, detail = ev["kind"], ev["detail"]
                payload = json.loads(ev["payload"] or "{}")
                exp = experiments.get(payload.get("exp_id"))
                if kind == "experiment" and exp is not None:
                    add("assistant", f"Hypothesis to test: {exp['hypothesis']}", event=ev["id"])
                    tries = sorted((exp_dir / f"exp{exp['id']}").glob("try*"), key=lambda p: int(p.name[3:] or 0))
                    for t in tries:
                        add("tool_call", _read(t / "experiment.py", 6_000), "run_experiment", attempt=t.name)
                        out, err = _read(t / "stdout.txt"), _read(t / "stderr.txt")
                        result = out + (f"\n--- stderr ---\n{err}" if err.strip() else "")
                        add("tool_result", result, "run_experiment", 1 if err.strip() else 0, attempt=t.name)
                elif kind == "exp_done" and exp is not None:
                    add("assistant", f"Verdict: {exp['verdict']}. {exp['finding']}", event=ev["id"])
                elif kind == "exp_retry":
                    continue  # the attempt it summarizes is already a step
                elif kind == "done":
                    report = Path(detail)
                    final = _read(report, 8_000) if report.is_file() else detail
                else:
                    add("log", f"[{ev['phase']}/{kind}] {detail}", "openboffin", event=ev["id"])
            if not final:
                final = f"(run {run['status']}; no report)"
            traces.append(
                Trace(
                    id=f"openboffin-run-{run['id']}",
                    task=run["question"],
                    steps=steps,
                    final_output=final,
                    meta={"source": "openboffin", "status": run["status"], "cost_usd": run["cost_usd"]},
                )
            )
    finally:
        con.close()
    return traces


# ----------------------------------------------------------------------- entry


def load(paths: list[str | Path], run_ids: list[int] | None = None) -> list[Trace]:
    traces: list[Trace] = []
    for p in map(Path, paths):
        if p.is_dir():
            files = sorted(f for f in p.rglob("*") if f.suffix in (".json", ".jsonl"))
            for f in files:
                traces.extend(_from_json_file(f))
        elif p.suffix in (".db", ".sqlite", ".sqlite3"):
            traces.extend(from_openboffin(p, run_ids))
        elif p.suffix in (".json", ".jsonl"):
            traces.extend(_from_json_file(p))
        else:
            raise ValueError(f"don't know how to read {p} (want .json, .jsonl, a directory, or an OpenBoffin .db)")
    return traces
