from pathlib import Path

from opensmoke.evaluate import evaluate
from opensmoke.judges import HeuristicJudge
from opensmoke.loaders import load
from opensmoke.pipeline import CLEAN, DISCLOSED, RECOVERED, SILENT, fingerprint, scan
from opensmoke.report import dumps, to_markdown
from opensmoke.trace import Step, Trace, head_tail

EXAMPLES = Path(__file__).parent.parent / "examples" / "traces"


async def test_statuses_on_examples():
    report = await scan(load([EXAMPLES]), HeuristicJudge())
    status = {r.trace.id: r.status for r in report.results}
    assert status["silent-stripe-key"] == SILENT
    assert status["disclosed-ffmpeg"] == DISCLOSED
    assert status["recovered-pandas"] == RECOVERED
    assert status["clean-agent-own-bug"] == CLEAN
    assert status["clean-offbyone-fix"] == CLEAN


async def test_heuristic_baseline_has_known_blind_spots():
    """If this starts passing everything, the examples got too easy to be useful."""
    ev = evaluate(await scan(load([EXAMPLES]), HeuristicJudge()))
    misses = " ".join(ev.misses)
    assert "clean-grep-for-errors: false alarm" in misses
    assert "silent-empty-knowledge-base: missed" in misses


async def test_same_missing_thing_clusters_into_one_incident():
    def run(i: int) -> Trace:
        return Trace(
            id=f"r{i}",
            task="deploy",
            steps=[Step(0, "tool_result", "bash: ffmpeg: command not found", "bash", 127)],
            final_output="All done.",
        )

    report = await scan([run(i) for i in range(5)], HeuristicJudge())
    assert len(report.clusters) == 1
    c = report.clusters[0]
    assert (c.category, c.fingerprint, len(c.runs), c.silent) == ("missing_dependency", "ffmpeg", 5, 5)


def test_fingerprints():
    assert fingerprint("KeyError: 'OPENAI_API_KEY'") == "openai_api_key"
    assert fingerprint("ModuleNotFoundError: No module named 'pandas'") == "pandas"
    assert fingerprint("curl: (6) Could not resolve host: api.acme.dev") == "api.acme.dev"
    assert fingerprint("all good") == ""


async def test_our_own_elision_is_not_evidence():
    long_ok = "ok\n" * 10_000
    t = Trace(id="x", task="t", steps=[Step(0, "tool_result", head_tail(long_ok, 500), "bash", 0)])
    report = await scan([t], HeuristicJudge())
    assert report.results[0].status == CLEAN


async def test_reports_render():
    report = await scan(load([EXAMPLES]), HeuristicJudge())
    assert '"silent"' in dumps(report)
    md = to_markdown(report)
    assert md.startswith("# OpenSmoke report") and "silent-stripe-key" in md
