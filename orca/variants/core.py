"""
Orca Core — god-mode variant.

Every response runs through:
  Think → Plan → Act (tools) → Respond → Reflect

Features:
  - Persistent memory (4 layers)
  - Web search (no API key)
  - Code execution (local sandbox)
  - Shell access (safe commands only)
  - File read/write
  - Self-reflection on complex answers
  - Auto-collects training data from every conversation
"""
from __future__ import annotations

import uuid
from typing import Iterator

from rich.console import Console
from rich.markdown import Markdown

from orca.config import CONFIG
from orca.character import CORE_SYSTEM_WITH_TOOLS, banner
from orca.brain.providers import get_brain
from orca.brain.memory import MemoryEngine
from orca.brain.context import ContextManager
from orca.brain.agent import AgentLoop
from orca.tools import build_registry

console = Console()


class OrcaCore:
    """Full-featured Orca with agent loop, tools, and persistent memory."""

    def __init__(
        self,
        session_id: str | None = None,
        load_session: str | None = None,
        collect_data: bool = True,
        model: str | None = None,
        reflect: bool = True,
        show_thoughts: bool = False,
    ):
        self.brain = get_brain(model or CONFIG.ollama.model_core)
        self.memory = MemoryEngine(session_id=session_id or str(uuid.uuid4()))
        self.ctx = ContextManager(self.brain)
        self.session_id = self.memory.session_id
        self.show_thoughts = show_thoughts

        self.tools = build_registry(memory_engine=self.memory)

        def _thought(msg: str):
            if self.show_thoughts:
                console.print(f"  [dim]{msg}[/dim]")

        self.agent = AgentLoop(
            brain=self.brain,
            tools=self.tools,
            session_id=self.session_id,
            on_thought=_thought,
            reflect=reflect,
        )

        # Inject distilled facts from prior sessions into the agent's context
        prior = self.memory.load_prior_context()
        if prior:
            self.agent.load_history([
                {"role": "user", "content": f"[Prior session context]\n{prior}"},
                {"role": "assistant", "content": "Context loaded. I remember."},
            ])

        self._collector = None
        if collect_data:
            try:
                from orca.data.collector import DataCollector
                self._collector = DataCollector(variant="core")
                self._collector.start_conversation()
            except Exception:
                pass

        if load_session:
            loaded = self.memory.load_session(load_session)
            if loaded:
                self.agent.load_history(self.memory.messages())
                console.print(f"[dim]Resumed session {load_session[:8]}[/dim]")

    def think(self, prompt: str, piped_input: str | None = None) -> str:
        """Single-shot deep response with full agent loop."""
        content = f"{piped_input}\n\n{prompt}" if piped_input else prompt
        mem_ctx = self.memory.recall_context(content)
        if mem_ctx:
            content = f"[Memory context]\n{mem_ctx}\n\n{content}"

        final, trace = self.agent.run(content)

        self.memory.add_turn("user", prompt)
        self.memory.add_turn("assistant", final)
        self.memory.commit_to_long_term(f"Q: {prompt[:200]}\nA: {final[:500]}")
        self._collect(prompt, final)
        return final

    def stream(self, prompt: str, piped_input: str | None = None) -> Iterator[str]:
        """Streaming response — tools run first, then streams the answer."""
        content = f"{piped_input}\n\n{prompt}" if piped_input else prompt
        mem_ctx = self.memory.recall_context(content)
        if mem_ctx:
            content = f"[Memory context]\n{mem_ctx}\n\n{content}"

        gen, trace = self.agent.stream(content)
        full = ""
        for chunk in gen:
            full += chunk
            yield chunk

        self.memory.add_turn("user", prompt)
        self.memory.add_turn("assistant", full)
        self.memory.commit_to_long_term(f"Q: {prompt[:200]}\nA: {full[:500]}")
        self._collect(prompt, full)

    def chat(self) -> None:
        """Interactive god-mode chat session."""
        from orca.tui import chat_help_panel, tool_spinner
        banner("core", self.brain.name)

        while True:
            try:
                prompt = console.input("\n[bold green]you ▸[/bold green] ").strip()
            except (KeyboardInterrupt, EOFError):
                self._save()
                console.print("\n[dim]Saved. Goodbye.[/dim]")
                break

            if not prompt:
                continue

            # Built-in commands
            if prompt.lower() in ("exit", "quit", "q"):
                self._save()
                console.print("[dim]Saved. Goodbye.[/dim]")
                break
            if prompt.lower() in ("/help", "help"):
                chat_help_panel("core")
                continue
            if prompt.lower() == "save" or prompt.lower() == "/save":
                self._save()
                continue
            if prompt.lower() == "tools" or prompt.lower() == "/tools":
                console.print(f"[dim]Available: {', '.join(self.tools.all_names())}[/dim]")
                continue
            if prompt.lower() in ("thoughts on", "/thoughts on"):
                self.show_thoughts = True
                self.agent.on_thought = lambda m: console.print(f"  [dim]◈ {m}[/dim]")
                console.print("[dim]◈ thought trace enabled[/dim]")
                continue
            if prompt.lower() in ("thoughts off", "/thoughts off"):
                self.show_thoughts = False
                self.agent.on_thought = lambda _: None
                console.print("[dim]◈ thought trace disabled[/dim]")
                continue
            if prompt.lower() in ("/clear", "clear"):
                self.memory.short.clear()
                self.agent.load_history([])
                console.print("[dim]◈ context cleared[/dim]")
                continue
            if prompt.lower() in ("/session", "session"):
                console.print(f"[dim]Session: {self.session_id}[/dim]")
                continue

            # Direct tool shortcuts — results feed back into Orca as context
            if prompt.startswith("/web "):
                query = prompt[5:].strip()
                from orca.tools.web import search_and_fetch
                with tool_spinner(f"searching: {query[:50]}"):
                    result = search_and_fetch(query)
                console.print(f"[dim]{result[:800]}[/dim]\n")
                # Feed result back into agent as context
                prompt = f"I searched for '{query}'. Here are the results:\n{result[:2000]}\n\nSummarize the key points and what they mean for me."
            elif prompt.startswith("/run "):
                code = prompt[5:].strip()
                from orca.tools.code import run_python
                with tool_spinner("running code"):
                    result = run_python(code)
                console.print(Panel(result.format(), title="[dim]output[/dim]", border_style="dim"))
                prompt = f"I ran this code:\n```python\n{code}\n```\nOutput:\n{result.format()}\n\nExplain what happened and if there are issues."
            elif prompt.startswith("/shell "):
                cmd = prompt[7:].strip()
                from orca.tools.code import run_shell
                with tool_spinner(f"$ {cmd[:50]}"):
                    result = run_shell(cmd)
                console.print(Panel(result.format(), title="[dim]shell[/dim]", border_style="dim"))
                prompt = f"Shell command `{cmd}` output:\n{result.format()}\n\nWhat does this tell us?"
            elif prompt.startswith("/recall "):
                query = prompt[8:].strip()
                with tool_spinner("searching memory"):
                    hits = self.memory.long.recall(query, n=5)
                prior = self.memory.load_prior_context()
                if not hits and not prior:
                    console.print("[dim]Nothing found in memory.[/dim]")
                else:
                    for h in hits:
                        console.print(f"  [dim]▸[/dim] {h['text'][:200]}")
                    if prior and not hits:
                        console.print(f"[dim]{prior[:600]}[/dim]")
                continue
            elif prompt.startswith("/think "):
                inner = prompt[7:].strip()
                with console.status("[dim]◈ deep thinking...[/dim]", spinner="dots"):
                    result = self.think(inner)
                console.print(Markdown(result))
                continue
            elif prompt.startswith("/remember "):
                fact = prompt[10:].strip()
                self.memory.commit_to_long_term(fact, {"type": "user_fact", "manual": True})
                self.memory.semantic.store_fact(f"user_note_{int(__import__('time').time())}", fact)
                # also persist to all_sessions_summary
                existing = self.memory.semantic.recall_fact("all_sessions_summary") or ""
                self.memory.semantic.store_fact("all_sessions_summary", f"{existing}\n[User note] {fact}".strip())
                console.print(f"[dim]◈ remembered: {fact[:80]}[/dim]")
                continue

            # Memory injection — recall relevant past context before responding
            mem_ctx = self.memory.recall_context(prompt, n=3)

            # Context compaction
            msgs = self.memory.messages()
            if msgs and self.ctx.needs_compaction(msgs):
                console.print("[dim]◈ compacting context...[/dim]")
                compacted = self.ctx.compact(msgs)
                self.memory.short._messages.clear()
                for m in compacted:
                    self.memory.short.add(m["role"], m["content"])
                self.agent.load_history(self.memory.messages())

            # Build enriched input with memory context
            enriched = f"[Relevant memory]\n{mem_ctx}\n\n{prompt}" if mem_ctx else prompt

            # Agent loop — stream response
            console.print("[bold blue]orca ▸[/bold blue] ", end="")
            full = ""
            try:
                gen, trace = self.agent.stream(enriched)
                for chunk in gen:
                    print(chunk, end="", flush=True)
                    full += chunk
            except Exception as e:
                console.print(f"\n[red]{e}[/red]")
                continue
            print()

            self.memory.add_turn("user", prompt)
            self.memory.add_turn("assistant", full)
            self.memory.commit_to_long_term(f"Q: {prompt[:200]}\nA: {full[:500]}")
            self._collect(prompt, full)

    def _collect(self, prompt: str, response: str) -> None:
        if self._collector:
            self._collector.log_turn("human", prompt)
            self._collector.log_turn("gpt", response)
            self._collector.save()
            self._collector.start_conversation()

    def _save(self) -> None:
        if len(self.memory.short) >= 2:
            with console.status("[dim]◈ distilling session to memory...[/dim]", spinner="dots"):
                summary = self.memory.distill_and_save(self.brain)
            if summary:
                console.print(f"[dim]◈ {len(summary.splitlines())} facts committed to long-term memory[/dim]")
        else:
            self.memory.save_session()
        console.print(f"[dim]Session saved: {self.session_id[:8]}[/dim]")
