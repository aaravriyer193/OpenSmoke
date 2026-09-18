import json
import sqlite3
from pathlib import Path

from opensmoke.loaders import load

EXAMPLES = Path(__file__).parent.parent / "examples" / "traces"


def test_examples_all_load_with_labels():
    traces = load([EXAMPLES])
    assert len(traces) >= 12
    assert all(t.expect is not None for t in traces)


def test_openai_messages():
    [t] = load([EXAMPLES / "disclosed-git-push-403.openai.json"])
    kinds = [s.kind for s in t.steps]
    assert kinds == ["user", "tool_call", "tool_result", "assistant"]
    assert t.task == "Push the fix branch and open a PR."
    assert t.steps[2].name == "bash"  # resolved from tool_call_id
    assert "403" in t.steps[2].content
    assert t.final_output.startswith("I committed the fix")


def test_anthropic_blocks():
    [t] = load([EXAMPLES / "silent-empty-knowledge-base.anthropic.json"])
    assert [s.kind for s in t.steps] == ["user", "tool_call", "tool_result", "assistant"]
    assert t.steps[2].name == "search_policies"
    assert '"results": []' in t.steps[2].content


def test_jsonl(tmp_path):
    f = tmp_path / "runs.jsonl"
    f.write_text("\n".join(json.dumps({"task": f"t{i}", "steps": []}) for i in range(3)))
    assert [t.id for t in load([f])] == ["runs:0", "runs:1", "runs:2"]


def test_openboffin(tmp_path):
    db = tmp_path / "boffin.db"
    con = sqlite3.connect(db)
    con.executescript(
        """
        CREATE TABLE runs (id INTEGER PRIMARY KEY, question TEXT, status TEXT, cost_usd REAL);
        CREATE TABLE events (id INTEGER PRIMARY KEY, run_id INT, ts REAL, phase TEXT, kind TEXT, detail TEXT, payload TEXT);
        CREATE TABLE experiments (id INTEGER PRIMARY KEY, run_id INT, hypothesis TEXT, verdict TEXT, finding TEXT);
        INSERT INTO runs VALUES (1, 'Does X hold?', 'done', 0.1);
        INSERT INTO experiments VALUES (1, 1, 'X holds', 'inconclusive', 'The output is truncated.');
        INSERT INTO events VALUES (1, 1, 0, 'read', 'read', '6 papers', '{}');
        INSERT INTO events VALUES (2, 1, 0, 'experiment', 'experiment', 'X holds', '{"exp_id": 1}');
        INSERT INTO events VALUES (3, 1, 0, 'experiment', 'exp_done', 'inconclusive', '{"exp_id": 1}');
        """
    )
    con.commit()
    con.close()
    attempt = tmp_path / "experiments" / "exp1" / "try1"
    attempt.mkdir(parents=True)
    (attempt / "experiment.py").write_text("print('hi')")
    (attempt / "stdout.txt").write_text("hi")
    (attempt / "stderr.txt").write_text("")

    [t] = load([db])
    assert t.id == "openboffin-run-1" and t.task == "Does X hold?"
    assert [s.kind for s in t.steps] == ["log", "assistant", "tool_call", "tool_result", "assistant"]
    assert t.steps[3].exit_code == 0
    assert "truncated" in t.steps[4].content
