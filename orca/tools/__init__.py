"""
Orca Tool Registry — all tools available to the agent loop.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Callable


@dataclass
class Tool:
    name: str
    description: str
    fn: Callable
    params: dict  # JSON schema of parameters


class ToolRegistry:
    def __init__(self):
        self._tools: dict[str, Tool] = {}

    def register(self, tool: Tool) -> None:
        self._tools[tool.name] = tool

    def get(self, name: str) -> Tool | None:
        return self._tools.get(name)

    def all_names(self) -> list[str]:
        return list(self._tools.keys())

    def schema_list(self) -> list[dict]:
        return [
            {"name": t.name, "description": t.description, "parameters": t.params}
            for t in self._tools.values()
        ]

    def call(self, name: str, args: dict) -> str:
        tool = self._tools.get(name)
        if not tool:
            return f"Unknown tool: {name}. Available: {', '.join(self.all_names())}"
        try:
            result = tool.fn(**args)
            return str(result) if not isinstance(result, str) else result
        except Exception as e:
            return f"Tool '{name}' error: {e}"


def build_registry(memory_engine=None) -> ToolRegistry:
    from orca.tools.web import search_and_fetch
    from orca.tools.code import run_code

    registry = ToolRegistry()

    registry.register(Tool(
        name="web_search",
        description="Search the web for current information, documentation, or facts",
        fn=lambda query, n=5: search_and_fetch(query, n=n),
        params={
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Search query"},
                "n": {"type": "integer", "description": "Number of results", "default": 5},
            },
            "required": ["query"],
        },
    ))

    registry.register(Tool(
        name="run_code",
        description="Execute Python or shell code locally and return output",
        fn=lambda code, language="python": run_code(code, language).format(),
        params={
            "type": "object",
            "properties": {
                "code": {"type": "string", "description": "Code to execute"},
                "language": {"type": "string", "enum": ["python", "shell"], "default": "python"},
            },
            "required": ["code"],
        },
    ))

    registry.register(Tool(
        name="read_file",
        description="Read a local file and return its contents",
        fn=_read_file,
        params={
            "type": "object",
            "properties": {
                "path": {"type": "string"},
                "lines": {"type": "integer", "description": "Max lines to read"},
            },
            "required": ["path"],
        },
    ))

    registry.register(Tool(
        name="write_file",
        description="Write content to a local file",
        fn=_write_file,
        params={
            "type": "object",
            "properties": {
                "path": {"type": "string"},
                "content": {"type": "string"},
            },
            "required": ["path", "content"],
        },
    ))

    registry.register(Tool(
        name="shell",
        description="Run a non-destructive shell command",
        fn=lambda command: __import__("orca.tools.code", fromlist=["run_shell"]).run_shell(command).format(),
        params={
            "type": "object",
            "properties": {"command": {"type": "string"}},
            "required": ["command"],
        },
    ))

    registry.register(Tool(
        name="memory_recall",
        description="Search your long-term memory for relevant past context",
        fn=(lambda q: _recall_memory_engine(q, memory_engine))
           if memory_engine else
           (lambda query, session_id="default": _recall_memory(query, session_id)),
        params={
            "type": "object",
            "properties": {
                "query": {"type": "string"},
            },
            "required": ["query"],
        },
    ))

    registry.register(Tool(
        name="investor_research",
        description="Research investors, VCs, funding rounds, and market data for business development",
        fn=_investor_research,
        params={
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "e.g. 'AI developer tools pre-seed VCs', 'market size private AI assistants'"},
            },
            "required": ["query"],
        },
    ))

    return registry


def _read_file(path: str, lines: int | None = None) -> str:
    from pathlib import Path
    p = Path(path).expanduser()
    if not p.exists():
        return f"File not found: {path}"
    text = p.read_text(errors="replace")
    if lines:
        text = "\n".join(text.splitlines()[:lines])
    return text


def _write_file(path: str, content: str) -> str:
    from pathlib import Path
    p = Path(path).expanduser()
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(content)
    return f"Written: {path} ({len(content)} chars)"


def _investor_research(query: str) -> str:
    """Targeted web search for investor and market intelligence."""
    from orca.tools.web import search_and_fetch
    investor_query = f"{query} site:crunchbase.com OR site:techcrunch.com OR site:ycombinator.com OR investors funding"
    results = search_and_fetch(investor_query, n=5)
    if not results or results.startswith("Search failed"):
        # Fallback: plain search
        results = search_and_fetch(query, n=5)
    return results


def _recall_memory_engine(query: str, engine) -> str:
    try:
        hits = engine.long.recall(query, n=5)
        if not hits:
            prior = engine.load_prior_context()
            return prior[:1000] if prior else "No relevant memories found."
        return "\n".join(f"- {h['text']}" for h in hits)
    except Exception:
        return "Memory not available."


def _recall_memory(query: str, session_id: str) -> str:
    try:
        from orca.brain.memory import LongTermMemory
        mem = LongTermMemory(session_id)
        hits = mem.recall(query, n=5)
        if not hits:
            return "No relevant memories found."
        return "\n".join(f"- {h['text']}" for h in hits)
    except Exception:
        return "Memory not available."
