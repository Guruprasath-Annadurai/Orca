"""
Orca Agent Loop — god-mode reasoning.

Pipeline per response:
  1. THINK    — reason about what's needed
  2. PLAN     — decide if tools are needed and which
  3. ACT      — call tools, observe outputs
  4. RESPOND  — draft the answer
  5. REFLECT  — critique and improve (optional, triggers on complex tasks)

This is what separates Orca from a simple chat wrapper.
Every response goes through this loop. Tools are called locally.
Self-reflection catches mistakes before the user sees them.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Callable, Iterator

from orca.brain.providers import OrcaBrain
from orca.tools import ToolRegistry
from orca.character import CORE_SYSTEM_WITH_TOOLS, REFLECTION_PROMPT

MAX_TOOL_ROUNDS = 6
REFLECTION_THRESHOLD = 150  # reflect if response > N words


PLANNER_SYSTEM = """\
You are Orca's planning module. Given a user message, decide:
1. Can this be answered directly from knowledge? → {"action": "direct"}
2. Does it need tools? → {"action": "tools", "calls": [{"tool": "tool_name", "args": {...}}, ...]}

Available tools: {tools}

Rules:
- Use web_search for: news, docs, current prices/versions, anything time-sensitive
- Use run_code for: math, data processing, verifying code logic
- Use shell for: file listings, git info, system status
- Use read_file/write_file for: local files the user references
- Use memory_recall for: "earlier you said", "last time we", "remember when"
- Combine tools when needed: web_search + run_code to fetch data then process it

Output ONLY valid JSON. No explanation.
"""

TOOL_OBSERVER_SYSTEM = """\
You are Orca. You called tools and got results.
Now answer the user's original question using those results.
Be direct. Cite tool output when relevant but don't dump raw data.
"""


@dataclass
class ToolCall:
    tool: str
    args: dict
    result: str = ""


@dataclass
class AgentTrace:
    """Full trace of an agent reasoning cycle — useful for debugging."""
    user_input: str
    plan_action: str = "direct"
    tool_calls: list[ToolCall] = field(default_factory=list)
    draft: str = ""
    reflected: bool = False
    final: str = ""


class AgentLoop:
    """
    God-mode reasoning loop.
    Think → Plan → Act → Respond → Reflect
    """

    def __init__(
        self,
        brain: OrcaBrain,
        tools: ToolRegistry,
        session_id: str = "default",
        on_thought: Callable[[str], None] | None = None,
        reflect: bool = True,
    ):
        self.brain = brain
        self.tools = tools
        self.session_id = session_id
        self.on_thought = on_thought or (lambda _: None)
        self.reflect = reflect
        self._history: list[dict] = []

    def run(self, user_input: str, system: str | None = None) -> tuple[str, AgentTrace]:
        """Full agent cycle — returns (final_response, trace)."""
        trace = AgentTrace(user_input=user_input)
        sys_prompt = system or CORE_SYSTEM_WITH_TOOLS.format(
            tools=", ".join(self.tools.all_names())
        )

        # ── Step 1: Plan ─────────────────────────────────────────────────────
        plan = self._plan(user_input, sys_prompt)
        trace.plan_action = plan.get("action", "direct")
        self.on_thought(f"[plan] {trace.plan_action}")

        # ── Step 2: Act (tool calls) ─────────────────────────────────────────
        tool_context = ""
        if trace.plan_action == "tools":
            calls = plan.get("calls", [])
            tool_context = self._execute_tools(calls, trace)

        # ── Step 3: Respond ──────────────────────────────────────────────────
        messages = list(self._history)
        user_msg = user_input
        if tool_context:
            user_msg = f"{user_input}\n\n[Tool results]\n{tool_context}"

        messages.append({"role": "user", "content": user_msg})
        draft = self.brain.complete(messages, system=sys_prompt)
        trace.draft = draft
        self.on_thought("[drafted]")

        # ── Step 4: Reflect ──────────────────────────────────────────────────
        word_count = len(draft.split())
        if self.reflect and word_count > REFLECTION_THRESHOLD:
            final = self._reflect(user_input, draft, sys_prompt)
            trace.reflected = True
            self.on_thought("[reflected]")
        else:
            final = draft

        trace.final = final

        # Update history
        self._history.append({"role": "user", "content": user_input})
        self._history.append({"role": "assistant", "content": final})

        return final, trace

    def stream(
        self,
        user_input: str,
        system: str | None = None,
    ) -> tuple[Iterator[str], AgentTrace]:
        """
        Streaming version. Tools and planning happen before streaming starts.
        Returns (stream_iterator, trace) — iterate the stream for chunks.
        """
        trace = AgentTrace(user_input=user_input)
        sys_prompt = system or CORE_SYSTEM_WITH_TOOLS.format(
            tools=", ".join(self.tools.all_names())
        )

        plan = self._plan(user_input, sys_prompt)
        trace.plan_action = plan.get("action", "direct")
        self.on_thought(f"[plan] {trace.plan_action}")

        tool_context = ""
        if trace.plan_action == "tools":
            tool_context = self._execute_tools(plan.get("calls", []), trace)

        messages = list(self._history)
        user_msg = user_input
        if tool_context:
            user_msg = f"{user_input}\n\n[Tool results]\n{tool_context}"
        messages.append({"role": "user", "content": user_msg})

        def _gen():
            full = ""
            for chunk in self.brain.stream(messages, system=sys_prompt):
                full += chunk
                yield chunk
            trace.final = full
            trace.draft = full
            self._history.append({"role": "user", "content": user_input})
            self._history.append({"role": "assistant", "content": full})

        return _gen(), trace

    def reset(self) -> None:
        self._history.clear()

    def load_history(self, messages: list[dict]) -> None:
        self._history = list(messages)

    def get_history(self) -> list[dict]:
        return list(self._history)

    # ── Internal ──────────────────────────────────────────────────────────────

    def _plan(self, user_input: str, sys_prompt: str) -> dict:
        """Ask Orca whether tools are needed."""
        tool_list = ", ".join(self.tools.all_names())
        planner_sys = PLANNER_SYSTEM.format(tools=tool_list)
        history_summary = ""
        if self._history:
            last = self._history[-2:]
            history_summary = "\n".join(
                f"{m['role']}: {m['content'][:200]}" for m in last
            )

        prompt = f"Recent context:\n{history_summary}\n\nUser: {user_input}" if history_summary else f"User: {user_input}"

        response = self.brain.complete(
            [{"role": "user", "content": prompt}],
            system=planner_sys,
            temperature=0.1,
            max_tokens=512,
        )
        try:
            start, end = response.find("{"), response.rfind("}") + 1
            return json.loads(response[start:end])
        except (json.JSONDecodeError, ValueError):
            return {"action": "direct"}

    def _execute_tools(self, calls: list[dict], trace: AgentTrace) -> str:
        results = []
        for call in calls[:MAX_TOOL_ROUNDS]:
            tool_name = call.get("tool", "")
            args = call.get("args", {})
            self.on_thought(f"[tool] {tool_name}({_fmt_args(args)})")
            result = self.tools.call(tool_name, args)
            tc = ToolCall(tool=tool_name, args=args, result=result[:3000])
            trace.tool_calls.append(tc)
            results.append(f"[{tool_name}]\n{result[:3000]}")
        return "\n\n".join(results)

    def _reflect(self, original_input: str, draft: str, sys_prompt: str) -> str:
        """Self-critique pass — improves the draft if there are real issues."""
        messages = [
            {"role": "user", "content": original_input},
            {"role": "assistant", "content": draft},
            {"role": "user", "content": REFLECTION_PROMPT},
        ]
        return self.brain.complete(messages, system=sys_prompt, temperature=0.3)


def _fmt_args(args: dict) -> str:
    if not args:
        return ""
    items = []
    for k, v in args.items():
        v_str = str(v)[:40] + "..." if len(str(v)) > 40 else str(v)
        items.append(f"{k}={v_str!r}")
    return ", ".join(items)
