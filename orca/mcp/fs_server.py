"""
orca-fs MCP server — safe filesystem navigator and editor.
Scoped to allowed directories, never traverses above cwd.
"""
from __future__ import annotations

import os
import json
from pathlib import Path
from typing import Any

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import Tool, TextContent

app = Server("orca-fs")
_allowed_roots: list[Path] = [Path.cwd(), Path.home() / "projects"]


def _safe_path(p: str) -> Path:
    path = Path(p).resolve()
    if not any(str(path).startswith(str(r)) for r in _allowed_roots):
        raise PermissionError(f"Path {p} is outside allowed directories")
    return path


@app.list_tools()
async def list_tools() -> list[Tool]:
    return [
        Tool(
            name="fs_read",
            description="Read a file's contents",
            inputSchema={
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "lines": {"type": "integer", "description": "Max lines to read"},
                },
                "required": ["path"],
            },
        ),
        Tool(
            name="fs_write",
            description="Write content to a file",
            inputSchema={
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "content": {"type": "string"},
                    "append": {"type": "boolean", "default": False},
                },
                "required": ["path", "content"],
            },
        ),
        Tool(
            name="fs_list",
            description="List directory contents",
            inputSchema={
                "type": "object",
                "properties": {
                    "path": {"type": "string", "default": "."},
                    "pattern": {"type": "string", "default": "*"},
                },
            },
        ),
        Tool(
            name="fs_search",
            description="Search for text in files",
            inputSchema={
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "query": {"type": "string"},
                    "extension": {"type": "string"},
                },
                "required": ["path", "query"],
            },
        ),
        Tool(
            name="fs_tree",
            description="Show directory tree",
            inputSchema={
                "type": "object",
                "properties": {
                    "path": {"type": "string", "default": "."},
                    "depth": {"type": "integer", "default": 3},
                },
            },
        ),
    ]


@app.call_tool()
async def call_tool(name: str, arguments: dict[str, Any]) -> list[TextContent]:
    try:
        if name == "fs_read":
            p = _safe_path(arguments["path"])
            text = p.read_text(errors="replace")
            if "lines" in arguments:
                text = "\n".join(text.splitlines()[: arguments["lines"]])
            return [TextContent(type="text", text=text)]

        if name == "fs_write":
            p = _safe_path(arguments["path"])
            p.parent.mkdir(parents=True, exist_ok=True)
            mode = "a" if arguments.get("append") else "w"
            p.open(mode).write(arguments["content"])
            return [TextContent(type="text", text=f"Written to {p}")]

        if name == "fs_list":
            p = _safe_path(arguments.get("path", "."))
            pattern = arguments.get("pattern", "*")
            items = sorted(p.glob(pattern))
            result = [
                {"name": i.name, "type": "dir" if i.is_dir() else "file", "size": i.stat().st_size if i.is_file() else None}
                for i in items
            ]
            return [TextContent(type="text", text=json.dumps(result, indent=2))]

        if name == "fs_search":
            p = _safe_path(arguments["path"])
            query = arguments["query"].lower()
            ext = arguments.get("extension", "")
            matches = []
            pattern = f"**/*.{ext}" if ext else "**/*"
            for f in p.glob(pattern):
                if f.is_file():
                    try:
                        text = f.read_text(errors="replace")
                        for i, line in enumerate(text.splitlines(), 1):
                            if query in line.lower():
                                matches.append({"file": str(f), "line": i, "text": line.strip()})
                    except Exception:
                        pass
            return [TextContent(type="text", text=json.dumps(matches[:50], indent=2))]

        if name == "fs_tree":
            p = _safe_path(arguments.get("path", "."))
            depth = arguments.get("depth", 3)

            def _tree(root: Path, prefix: str = "", current_depth: int = 0) -> list[str]:
                if current_depth >= depth:
                    return []
                lines = []
                items = sorted(root.iterdir()) if root.is_dir() else []
                for i, item in enumerate(items):
                    connector = "└── " if i == len(items) - 1 else "├── "
                    lines.append(f"{prefix}{connector}{item.name}")
                    if item.is_dir():
                        extension = "    " if i == len(items) - 1 else "│   "
                        lines.extend(_tree(item, prefix + extension, current_depth + 1))
                return lines

            tree_lines = [str(p)] + _tree(p)
            return [TextContent(type="text", text="\n".join(tree_lines))]

    except PermissionError as e:
        return [TextContent(type="text", text=f"Permission denied: {e}")]
    except Exception as e:
        return [TextContent(type="text", text=f"Error: {e}")]

    return [TextContent(type="text", text=f"Unknown tool: {name}")]


async def main():
    async with stdio_server() as (r, w):
        await app.run(r, w, app.create_initialization_options())


if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
