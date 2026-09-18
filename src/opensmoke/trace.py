"""The one trace shape everything else reads.

Loaders turn whatever an agent framework logs into a `Trace`; judges, the
pipeline and the report never see a framework-specific format.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

StepKind = Literal["user", "assistant", "tool_call", "tool_result", "log"]

# Steps where the environment can show through. User messages and the agent's
# own tool calls are context for these, not evidence on their own.
JUDGED_KINDS: frozenset[str] = frozenset({"assistant", "tool_result", "log"})


@dataclass
class Step:
    index: int
    kind: StepKind
    content: str
    name: str = ""  # tool name, for tool_call / tool_result
    exit_code: int | None = None
    meta: dict[str, Any] = field(default_factory=dict)

    @property
    def label(self) -> str:
        return f"{self.kind}:{self.name}" if self.name else self.kind


@dataclass
class Trace:
    id: str
    task: str
    steps: list[Step]
    final_output: str = ""
    meta: dict[str, Any] = field(default_factory=dict)
    # Ground truth for `opensmoke eval`: {"smoke": bool, "silent": bool, "category": str}
    expect: dict[str, Any] | None = None

    def previous(self, step: Step, kind: str | None = None) -> Step | None:
        """The nearest earlier step, optionally of one kind."""
        for s in reversed(self.steps[: step.index]):
            if kind is None or s.kind == kind:
                return s
        return None


# Our own elision must never read as evidence: a judge that sees "characters
# omitted" would blame the agent's environment for a cut OpenSmoke made.
ELIDED = "⟪opensmoke elided {n} chars for length; not part of the trace⟫"


def head_tail(text: str, limit: int) -> str:
    """Keep both ends of long text.

    Errors live at the end of output and the command that caused them at the
    start; cutting either end hides exactly what a judge needs to see.
    """
    if len(text) <= limit:
        return text
    head = int(limit * 0.4)
    tail = limit - head
    return f"{text[:head]}\n{ELIDED.format(n=f'{len(text) - limit:,}')}\n{text[-tail:]}"
