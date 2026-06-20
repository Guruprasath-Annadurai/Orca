"""
Orca Ultra — apex multi-agent orchestrator. God mode.

Pod structure: Orca coordinates specialized sub-agents like a killer whale pod.
Each agent has a role, runs in parallel, reports back.
Ultra synthesizes, grades, and self-heals if quality is low.
"""
from __future__ import annotations

import asyncio
import json
import time
import uuid
from dataclasses import dataclass, field
from typing import Callable

from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn

from orca.brain.providers import get_brain
from orca.brain.agent import AgentLoop
from orca.tools import build_registry
from orca.character import ULTRA_SYSTEM, AGENT_SYSTEMS, banner
from orca.config import CONFIG

console = Console()

DECOMPOSE_SYSTEM = """\
You are ORCA-ULTRA's planning module. Decompose the user's task.

Output ONLY this JSON:
{
  "goal": "restated goal in one sentence",
  "subtasks": [
    {"id": "t1", "agent": "researcher|coder|analyst|writer|critic|architect", "task": "specific task description", "depends_on": []}
  ],
  "synthesis_prompt": "How to combine all results into a final answer"
}

Rules:
- Parallelize everything that can be parallel (depends_on: [])
- Use critic agent to validate output from coder/architect
- Use researcher before coder if external info is needed
- Maximum 6 subtasks unless the task genuinely requires more
"""

GRADE_SYSTEM = """\
You are ORCA's quality grader. Score the output against the goal.
Output ONLY JSON: {"score": int 0-100, "feedback": "specific improvement if score < 80"}
Score 100 = perfect. Score < 60 = retry. Be a harsh judge.
"""


@dataclass
class SubTask:
    id: str
    agent: str
    task: str
    depends_on: list[str] = field(default_factory=list)
    result: str = ""
    status: str = "pending"
    duration: float = 0.0


@dataclass
class Pipeline:
    goal: str
    subtasks: list[SubTask]
    synthesis_prompt: str = ""
    final_output: str = ""
    grade: dict = field(default_factory=dict)
    run_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    iterations: int = 1


class OrcaUltra:
    """Multi-agent orchestrator — apex intelligence coordinating a pod."""

    def __init__(
        self,
        on_progress: Callable[[str], None] | None = None,
        model: str | None = None,
        use_tools: bool = True,
    ):
        self.brain = get_brain(model or CONFIG.ollama.model_ultra)
        self.tools = build_registry() if use_tools else None
        self.on_progress = on_progress or print

    def run(self, goal: str, max_retries: int = 2) -> Pipeline:
        return asyncio.run(self._run_async(goal, max_retries, iteration=1))

    def chat(self) -> None:
        """Interactive Ultra session."""
        from rich.markdown import Markdown
        from orca.tui import chat_help_panel, grade_display, agent_pod_display
        banner("ultra", self.brain.name)

        while True:
            try:
                task = console.input("\n[bold magenta]task ▸[/bold magenta] ").strip()
            except (KeyboardInterrupt, EOFError):
                console.print("\n[dim]Goodbye.[/dim]")
                break
            if not task:
                continue
            if task.lower() in ("exit", "quit"):
                console.print("[dim]Goodbye.[/dim]")
                break
            if task.lower() in ("/help", "help"):
                chat_help_panel("ultra")
                continue

            pipeline = self.run(task)
            console.print("\n")
            console.print(Markdown(pipeline.final_output))
            grade_display(pipeline.grade.get("score", 0), pipeline.iterations)

    async def _run_async(self, goal: str, max_retries: int, iteration: int = 1) -> Pipeline:
        self.on_progress(f"\n[ORCA ULTRA] Deploying pod for: {goal[:70]}...")

        # Decompose
        plan = await self._decompose(goal)
        subtasks = [
            SubTask(
                id=t.get("id", f"t{i}"),
                agent=t.get("agent", "researcher"),
                task=t.get("task", ""),
                depends_on=t.get("depends_on", []),
            )
            for i, t in enumerate(plan.get("subtasks", []))
            if t.get("task")
        ]

        if not subtasks:
            subtasks = [SubTask(id="t1", agent="researcher", task=goal)]

        pipeline = Pipeline(
            goal=goal,
            subtasks=subtasks,
            synthesis_prompt=plan.get("synthesis_prompt", "Combine all agent outputs."),
            iterations=iteration,
        )

        self.on_progress(f"[pod] {len(subtasks)} agents: {', '.join(t.agent for t in subtasks)}")

        # Execute
        await self._execute(pipeline)

        # Synthesize
        self.on_progress("[synthesize] Merging results...")
        pipeline.final_output = await self._synthesize(pipeline)

        # Grade
        pipeline.grade = await self._grade(goal, pipeline.final_output)
        score = pipeline.grade.get("score", 0)
        self.on_progress(f"[grade] {score}/100")

        # Self-heal if score too low
        if score < 65 and max_retries > 0:
            feedback = pipeline.grade.get("feedback", "")
            self.on_progress(f"[self-heal] Score {score} — retrying (attempt {iteration + 1})...")
            improved_goal = f"{goal}\n\n[Previous attempt was insufficient. Feedback: {feedback}. Improve specifically on this.]"
            return await self._run_async(improved_goal, max_retries - 1, iteration + 1)

        return pipeline

    async def _decompose(self, goal: str) -> dict:
        result = await asyncio.to_thread(
            self.brain.complete,
            [{"role": "user", "content": f"Decompose this task:\n{goal}"}],
            DECOMPOSE_SYSTEM,
            0.1,
        )
        try:
            s, e = result.find("{"), result.rfind("}") + 1
            return json.loads(result[s:e])
        except (json.JSONDecodeError, ValueError):
            return {
                "subtasks": [{"id": "t1", "agent": "researcher", "task": goal, "depends_on": []}],
                "synthesis_prompt": "Present the findings.",
            }

    async def _execute(self, pipeline: Pipeline) -> None:
        completed: set[str] = set()
        remaining = list(pipeline.subtasks)

        for _ in range(len(remaining) + 2):
            if not remaining:
                break
            ready = [t for t in remaining if all(d in completed for d in t.depends_on)]
            if not ready:
                break

            self.on_progress(f"[parallel] {len(ready)} agent(s) running: {[t.agent for t in ready]}")
            results = await asyncio.gather(*[self._run_agent(t) for t in ready])

            for task, result in zip(ready, results):
                task.result = result
                task.status = "done"
                completed.add(task.id)
                remaining.remove(task)

    async def _run_agent(self, task: SubTask) -> str:
        start = time.time()
        system = AGENT_SYSTEMS.get(task.agent, AGENT_SYSTEMS["researcher"])
        self.on_progress(f"  [{task.agent}] {task.task[:65]}...")

        if self.tools:
            # Agents with tool access
            loop = AgentLoop(brain=self.brain, tools=self.tools, reflect=False)
            result, _ = await asyncio.to_thread(loop.run, task.task, system)
        else:
            result = await asyncio.to_thread(
                self.brain.complete,
                [{"role": "user", "content": task.task}],
                system,
            )

        task.duration = time.time() - start
        self.on_progress(f"  [{task.agent}] done ({task.duration:.1f}s)")
        return result

    async def _synthesize(self, pipeline: Pipeline) -> str:
        agent_outputs = "\n\n".join(
            f"━━ {t.agent.upper()} (task: {t.task[:60]}) ━━\n{t.result}"
            for t in pipeline.subtasks if t.result
        )
        prompt = (
            f"Original goal: {pipeline.goal}\n\n"
            f"Synthesis instruction: {pipeline.synthesis_prompt}\n\n"
            f"Agent outputs:\n{agent_outputs}"
        )
        return await asyncio.to_thread(
            self.brain.complete,
            [{"role": "user", "content": prompt}],
            ULTRA_SYSTEM,
        )

    async def _grade(self, goal: str, output: str) -> dict:
        resp = await asyncio.to_thread(
            self.brain.complete,
            [{"role": "user", "content": f"Goal:\n{goal}\n\nOutput:\n{output[:3000]}"}],
            GRADE_SYSTEM,
            0.1,
        )
        try:
            s, e = resp.find("{"), resp.rfind("}") + 1
            return json.loads(resp[s:e])
        except (json.JSONDecodeError, ValueError):
            return {"score": 75, "feedback": resp[:100]}
