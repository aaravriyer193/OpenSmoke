"""JevJudge against the real SDK with a fake server: what we send, and how we read it."""

import json

import httpx
import httpx2
import pytest

from opensmoke.diagnose import Diagnoser
from opensmoke.judges import JevJudge, JudgeError
from opensmoke.pipeline import SILENT, Settings, scan
from opensmoke.trace import Step, Trace


def trace() -> Trace:
    return Trace(
        id="t",
        task="Add Stripe checkout",
        steps=[
            Step(0, "tool_call", "python check.py", "bash"),
            Step(1, "tool_result", "KeyError: 'STRIPE_SECRET_KEY'", "bash", 1),
            Step(2, "assistant", "I'll use a placeholder key."),
        ],
        final_output="Done! Checkout works.",
    )


def fake_typesafe(sent: list[dict]):
    def handler(request: httpx2.Request) -> httpx2.Response:
        body = json.loads(request.content)
        sent.append(body)
        q = body["questions"]
        if "env_broken" in q:
            broken = "KeyError" in body["state"]["step"]["content"]
            answers = {
                "env_broken": {"type": "noul", "noul": 0.93 if broken else 0.08},
                "category": {
                    "type": "choice",
                    "choice": "missing_credential" if broken else "none",
                    "confidence": 0.9,
                    "probabilities": {c: 0.01 for c in q["category"]["criteria"]}
                    | {("missing_credential" if broken else "none"): 0.9},
                },
                "workaround": {"type": "noul", "noul": 0.9 if "placeholder" in body["state"]["step"]["content"] else 0.05},
            }
        else:
            answers = {
                "claims_success": {"type": "noul", "noul": 0.95},
                "discloses_problem": {"type": "noul", "noul": 0.04},
                "recovered": {"type": "noul", "noul": 0.03},
            }
        return httpx2.Response(200, json={"model": body["model"], "usage": {"input_tokens": 500, "output_tokens": 0}, "answers": answers})

    return handler


async def test_jev_request_shape_and_verdict():
    sent: list[dict] = []
    judge = JevJudge(api_key="test", transport=httpx2.MockTransport(fake_typesafe(sent)))
    report = await scan([trace()], judge, Settings(threshold=0.7))
    await judge.aclose()

    step_calls = [b for b in sent if "env_broken" in b["questions"]]
    assert len(step_calls) == 2  # tool_result and assistant; the tool_call is context only
    assert all(b["model"] == "jev-1.13" for b in sent)
    result_call = next(b for b in step_calls if b["state"]["step"]["kind"] == "tool_result")
    assert result_call["state"]["caused_by"]["input"] == "python check.py"
    assert set(result_call["questions"]["category"]["criteria"]) >= {"missing_credential", "none"}

    r = report.results[0]
    assert r.status == SILENT
    assert r.category == "missing_credential"
    assert r.fingerprint == "stripe_secret_key"
    assert report.judge_cost_usd == pytest.approx(3 * 500 * 0.042 / 1e6)


async def test_bad_key_is_fatal_not_per_step_noise():
    judge = JevJudge(api_key="bad", transport=httpx2.MockTransport(lambda r: httpx2.Response(401, json={"error": "invalid key"})))
    with pytest.raises(JudgeError):
        await scan([trace()], judge)
    await judge.aclose()


async def test_diagnoser_reads_fenced_json_and_records_cost():
    reply = '```json\n{"verdict": "environment", "title": "Stripe key missing in sandbox", "missing": "STRIPE_SECRET_KEY", "fix": "Add STRIPE_SECRET_KEY to the sandbox secrets", "evidence_steps": [1], "confidence": 0.9}\n```'
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen.update(json.loads(request.content))
        return httpx.Response(200, json={"choices": [{"message": {"content": reply}}], "usage": {"cost": 0.0012}})

    d = Diagnoser(api_key="k", transport=httpx.MockTransport(handler))
    out = await d.diagnose(trace(), [{"index": 1, "env_broken": 0.93, "category": "missing_credential"}])
    await d.aclose()
    assert out.verdict == "environment" and out.missing == "STRIPE_SECRET_KEY"
    assert d.cost_usd == pytest.approx(0.0012)
    assert "KeyError: 'STRIPE_SECRET_KEY'" in seen["messages"][1]["content"]


async def test_rejected_llm_key_costs_one_call_and_one_note():
    from opensmoke.judges import HeuristicJudge

    calls = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(1)
        return httpx.Response(401, json={"error": {"message": "User not found."}})

    d = Diagnoser(api_key="dead", transport=httpx.MockTransport(handler))
    report = await scan([trace(), trace(), trace()], HeuristicJudge(), diagnoser=d)
    await d.aclose()
    assert len(calls) == 1
    assert len(report.notes) == 1 and "key rejected" in report.notes[0]
    assert not any(r.errors for r in report.results)
