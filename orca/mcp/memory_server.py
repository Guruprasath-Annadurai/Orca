"""
orca-memory MCP server — exposes Orca's memory engine as MCP tools.
Run: python -m orca.mcp.memory_server
"""
from __future__ import annotations

import json
from typing import Any

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import Tool, TextContent

from orca.brain.memory import MemoryEngine, EpisodicMemory, SemanticMemory

app = Server("orca-memory")
_memory: dict[str, MemoryEngine] = {}
_semantic = SemanticMemory()


def _get_mem(session_id: str) -> MemoryEngine:
    if session_id not in _memory:
        _memory[session_id] = MemoryEngine(session_id=session_id)
    return _memory[session_id]


@app.list_tools()
async def list_tools() -> list[Tool]:
    return [
        Tool(
            name="memory_store",
            description="Store a piece of information in long-term memory for a session",
            inputSchema={
                "type": "object",
                "properties": {
                    "session_id": {"type": "string"},
                    "text": {"type": "string", "description": "Text to remember"},
                    "metadata": {"type": "object", "default": {}},
                },
                "required": ["session_id", "text"],
            },
        ),
        Tool(
            name="memory_recall",
            description="Recall relevant memories for a given query",
            inputSchema={
                "type": "object",
                "properties": {
                    "session_id": {"type": "string"},
                    "query": {"type": "string"},
                    "n": {"type": "integer", "default": 5},
                },
                "required": ["session_id", "query"],
            },
        ),
        Tool(
            name="memory_sessions",
            description="List all saved sessions",
            inputSchema={"type": "object", "properties": {}},
        ),
        Tool(
            name="fact_store",
            description="Store a persistent fact in semantic memory",
            inputSchema={
                "type": "object",
                "properties": {
                    "key": {"type": "string"},
                    "value": {"type": "string"},
                },
                "required": ["key", "value"],
            },
        ),
        Tool(
            name="fact_recall",
            description="Retrieve a stored fact",
            inputSchema={
                "type": "object",
                "properties": {"key": {"type": "string"}},
                "required": ["key"],
            },
        ),
    ]


@app.call_tool()
async def call_tool(name: str, arguments: dict[str, Any]) -> list[TextContent]:
    if name == "memory_store":
        mem = _get_mem(arguments["session_id"])
        doc_id = mem.commit_to_long_term(arguments["text"], arguments.get("metadata", {}))
        return [TextContent(type="text", text=f"Stored with id: {doc_id}")]

    if name == "memory_recall":
        mem = _get_mem(arguments["session_id"])
        hits = mem.long.recall(arguments["query"], n=arguments.get("n", 5))
        return [TextContent(type="text", text=json.dumps(hits, indent=2))]

    if name == "memory_sessions":
        sessions = EpisodicMemory.list_sessions()
        return [TextContent(type="text", text=json.dumps(sessions))]

    if name == "fact_store":
        _semantic.store_fact(arguments["key"], arguments["value"])
        return [TextContent(type="text", text="Fact stored.")]

    if name == "fact_recall":
        val = _semantic.recall_fact(arguments["key"])
        return [TextContent(type="text", text=str(val) if val is not None else "Not found.")]

    return [TextContent(type="text", text=f"Unknown tool: {name}")]


async def main():
    async with stdio_server() as (r, w):
        await app.run(r, w, app.create_initialization_options())


if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
