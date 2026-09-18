<p align="center">
  <img src="assets/logo.svg" width="112" alt="OpenSmoke logo: smoke rising from an ember">
</p>

<h1 align="center">OpenSmoke</h1>

<p align="center">
  <b>Find the agent runs that broke because their environment did.</b><br>
  Missing keys, missing tools, missing files, blocked permissions, dead network,<br>
  empty instructions, truncated output: the failures a human caused and the agent quietly worked around.
</p>

---

An agent in production hits `KeyError: 'STRIPE_SECRET_KEY'`, says *"I'll use a
placeholder key for now"*, and tells the user **"Done! Checkout is integrated. ✅"**

Nothing crashed. Every dashboard is green. The agent knew, and nobody else did.

These failures are not the model's fault, and no prompt change fixes them.
Someone forgot a secret, shipped a sandbox image without `ffmpeg`, blocked
egress, or never loaded `AGENTS.md`. The evidence is always in the trace. The
problem is that nobody reads every trace.

OpenSmoke does. It asks
[TypeSafe's Jev](https://typesafe.ai/blog/introducing-system-one-models-and-jev),
a model that returns typed decisions with calibrated probabilities rather than
text, three questions about **every step of every run**. Jev costs $0.042 per
million input tokens, so reading all of production traffic is cheaper than
sampling 2% of it with an LLM judge. Runs Jev flags go to a strong LLM for a
second opinion and a root cause. Runs that are missing the same thing are grouped
into one incident, with a fix addressed to whoever owns the environment.

```
every step ──Jev──▶ flagged steps ──Jev──▶ silent · disclosed · recovered · clean
                                                 │
                    silent + disclosed runs ──LLM──▶ root cause + fix ──▶ incidents
```

## What it caught on a real agent

OpenBoffin is a long-running research agent that writes and runs its own
experiments. Pointed at its run database with no configuration, and using only
the offline regex judge, OpenSmoke found two real harness bugs that had each cost
a run its result:

```
● SILENT openboffin-run-1  resource_limit
  step 9 assistant  "The output is truncated: Experiment 1 … is not visible."
● SILENT openboffin-run-3  broken_tool
  step 12 tool_result  File "experiment.py", line 1  ```python  ^ SyntaxError
```

In run 1, the sandbox kept only the tail of the experiment's output, so the
headline table never reached the agent. The agent said so, marked a successful
experiment "inconclusive", and the run finished "done". In run 3, the harness
executed model-written code without stripping its Markdown fence. Both were found
by hand and fixed later; a scan after the first run would have flagged them. Run 1
is the case OpenSmoke is built for: the only evidence was the agent saying so.

## Quickstart

```bash
uv tool install git+https://github.com/aaravriyer193/OpenSmoke

opensmoke scan traces/                          # asks for your keys the first time
                                                # files, folders, JSONL, or an OpenBoffin .db
opensmoke scan traces/ --markdown smoke.md --json smoke.json
opensmoke scan traces/ --webhook $SLACK_WEBHOOK_URL
opensmoke scan traces/ --fail-on silent         # exit 1 in CI if anything silent got through
```

No keys yet? `--judge heuristic` runs a regex baseline offline, which is enough
to try the pipeline:

```bash
git clone https://github.com/aaravriyer193/OpenSmoke && cd OpenSmoke
uv run opensmoke scan examples/traces --judge heuristic --no-diagnose
```

## Bring your own keys

OpenSmoke has no server and no account. You pay TypeSafe and your LLM provider
directly, and your traces go straight from your machine to them.

| Key | Needed for | Get one |
|---|---|---|
| TypeSafe | the Jev judge (required unless `--judge heuristic`) | [console.typesafe.ai](https://console.typesafe.ai) |
| OpenRouter | the LLM second opinion (optional) | [openrouter.ai/keys](https://openrouter.ai/keys) |

The first time you run a scan in a terminal, OpenSmoke asks for any missing key
with a hidden prompt and offers to save it to `~/.config/opensmoke/keys.env`,
which only you can read. `opensmoke keys` shows where each key comes from (masked)
and lets you replace one.

Environment variables always win over the saved file, so CI and secret managers
work as usual: set `TYPESAFE_API_KEY` and `OPENROUTER_API_KEY`. OpenSmoke never
prompts when it isn't attached to a terminal, so a scheduled job fails with a clear
message instead of hanging.

## Using it for real

OpenSmoke reads traces your agents already write. It does not sit in the request
path, so it cannot slow down or break an agent. Three ways to run it:

**On a schedule (production monitoring).** Export the last hour of runs from
wherever you log them (your database, Langfuse, LangSmith, OpenTelemetry) to
JSONL, then scan and alert:

```yaml
# .github/workflows/opensmoke.yml
on:
  schedule: [{cron: "0 * * * *"}]            # hourly
jobs:
  smoke:
    runs-on: ubuntu-latest
    steps:
      - run: ./export-last-hour.sh > runs.jsonl   # your export: one run per line
      - run: uvx --from git+https://github.com/aaravriyer193/OpenSmoke opensmoke scan runs.jsonl --webhook "$SLACK_WEBHOOK_URL"
        env:
          TYPESAFE_API_KEY: ${{ secrets.TYPESAFE_API_KEY }}
          OPENROUTER_API_KEY: ${{ secrets.OPENROUTER_API_KEY }}
          SLACK_WEBHOOK_URL: ${{ secrets.SLACK_WEBHOOK_URL }}
```

The Slack message lists incidents, not runs: *"missing_credential
`stripe_secret_key`: 40 runs (40 silent)"*, with the fix. The full JSON rides
along for anything else listening.

**As a gate before a deploy.** Run your agent's eval suite against the new
sandbox image or config, then `opensmoke scan traces/ --fail-on silent`. A
missing secret or package fails the build before users meet it.

**From Python**, if your agent already runs in Python:

```python
import asyncio
from opensmoke import load, scan
from opensmoke.judges import JevJudge

report = asyncio.run(scan(load(["runs.jsonl"]), JevJudge()))
for incident in report.clusters:
    print(incident.category, incident.fingerprint, len(incident.runs))
```

What it does not do yet: each scan stands alone, so an incident from the last
hour is reported again if it is still happening. There is no memory across scans,
no built-in export from Langfuse or LangSmith, and no `watch` mode that tails a
directory. Those are the next things to build.

## How a run is judged

For each assistant message, tool result, and log line, Jev answers three
questions in one call (see [`questions.py`](src/opensmoke/questions.py)):

| Question | Type | Asks |
|---|---|---|
| `env_broken` | Noul | Is something the agent depends on missing, blocked, truncated, or broken, *as opposed to* the agent's own bug? |
| `category` | Choice | Which of: `missing_dependency`, `missing_credential`, `missing_file`, `permission_denied`, `network`, `missing_context`, `resource_limit`, `broken_tool`, `none` |
| `workaround` | Noul | Is the agent carrying on by assuming, guessing, skipping, or using placeholders instead of reporting? |

A tool result is judged together with the call that caused it and the task, and
the judge never sees more than a few thousand characters of one step at a time.
Jev degrades on long, irrelevant context, so OpenSmoke never hands it a whole
trace.

If any step is flagged, Jev asks three more questions about the final answer: does
it claim success, does it tell the user about the problem, and did the agent
actually recover? That gives each run a status:

| Status | Meaning |
|---|---|
| **silent** | The environment broke and the user was not told. This is the one that matters. |
| **disclosed** | The environment broke and the agent said so. Still an incident for whoever owns the sandbox. |
| **recovered** | The environment broke and the agent fixed it itself (for example, `pip install`). |
| **clean** | Nothing flagged. An agent fixing its own bug is clean. |

Silent and disclosed runs go to the diagnoser (any OpenAI-compatible endpoint;
GLM 5.2 on OpenRouter by default). It can dismiss a false alarm, names the
missing thing (`ffmpeg`, `STRIPE_SECRET_KEY`, `network egress`), and writes the
fix. Forty runs missing the same secret become one incident, not forty alerts.

## Trace format

Anything with steps works. The native shape:

```json
{
  "id": "run-8812",
  "task": "Add Stripe checkout to the /buy page.",
  "steps": [
    {"type": "tool_call",   "name": "bash", "content": "python scripts/check_stripe.py"},
    {"type": "tool_result", "name": "bash", "content": "KeyError: 'STRIPE_SECRET_KEY'", "exit_code": 1},
    {"type": "assistant",   "content": "I'll use a placeholder key for now."}
  ],
  "final_output": "Done! Stripe checkout is integrated. ✅"
}
```

OpenSmoke also reads OpenAI chat messages (`tool_calls` and `role: "tool"`),
Anthropic content blocks (`tool_use` and `tool_result`), JSONL with one run per
line, and OpenBoffin's SQLite database. Writing an adapter for another format means
producing a list of `Step`s; see [`loaders.py`](src/opensmoke/loaders.py).

## Measuring it

[`examples/traces`](examples/traces) holds 12 labeled runs, including ones built
to fool a detector: a `grep` whose *results* contain "No such file or directory",
an agent that hits its own `TypeError` and fixes it, a knowledge base that
silently returns zero documents.

```bash
opensmoke eval examples/traces --judge jev
opensmoke eval examples/traces --judge heuristic
```

| Judge | Precision | Recall | Notes |
|---|---|---|---|
| heuristic (regex) | 0.86 | 0.86 | Flags the `grep` output, misses the empty knowledge base, calls a DNS failure a missing package |
| jev-1.13 | not yet measured | | Run the command above with a key and open a PR with the numbers |

Twelve traces are a smoke test (sorry), not a benchmark. The number that matters
is precision and recall on your own traces, so label a few dozen and run `eval`.

## Limits, honestly

- **Jev can be confidently wrong.** TypeSafe's own eval across four workflows,
  one of them agent-trace observability, reports about 68% agreement with
  reference labels (a GPT model scored the same at 75× the cost). That is why
  flags go to a second-opinion LLM before anyone gets paged.
- **Traces are untrusted input.** A tool result can contain text written to steer
  a classifier. Both stages treat the trace as data, but a determined injection
  can still shift a probability.
- **Multi-hop failures are hard.** "The key was missing at step 3, so the number
  at step 40 is fake" spans too many steps for a per-step judge. The run-level
  questions and the diagnoser help; they do not solve it.
- **Your traces leave your machine.** They go to TypeSafe, and flagged ones go to
  your LLM provider. Scrub secrets first, or run `--judge heuristic --no-diagnose`.
- **It is not the only tool.** [Raindrop](https://raindrop.ai),
  [Galileo](https://galileo.ai), [Patronus Percival](https://www.patronus.ai/percival)
  and Lemma all monitor agents in production.
  OpenSmoke is narrower: open source, focused on *environment* failures and who
  needs to fix them, and cheap enough to read every step.

## Configuration

| Variable | Default | |
|---|---|---|
| `TYPESAFE_API_KEY` | | Required for `--judge jev`; prompted for if missing |
| `TYPESAFE_DEFAULT_MODEL` | `jev-1.13` | Or `--model` |
| `OPENROUTER_API_KEY` | | Enables diagnosis; prompted for if missing |
| `OPENSMOKE_LLM_API_KEY` | | Overrides the above for another provider |
| `OPENSMOKE_LLM_BASE_URL` | `https://openrouter.ai/api/v1` | Any OpenAI-compatible endpoint |
| `OPENSMOKE_LLM_MODEL` | `z-ai/glm-5.2` | Or `--llm-model` |

`--threshold` (default 0.7) sets the `env_broken` probability that flags a step.
Lower catches more and pages more.

## Development

```bash
uv sync
uv run pytest
```

The Jev tests run the real TypeSafe SDK against a fake server, so they check the
request OpenSmoke sends and how it reads the answer, without a key.

## License

MIT
