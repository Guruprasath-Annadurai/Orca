"""
Orca TUI — visual shell components.

Boot screen · Live dashboard · Session history · Streaming indicators
"""
from __future__ import annotations

import sys
import time
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Callable, Iterator

from rich.align import Align
from rich.columns import Columns
from rich.console import Console
from rich.layout import Layout
from rich.live import Live
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.rule import Rule
from rich.table import Table
from rich.text import Text
from rich import box

console = Console()

# ─────────────────────────────────────────────────────────────────────────────
#  ASCII art
# ─────────────────────────────────────────────────────────────────────────────

ORCA_ASCII = """\
[white]                ▄█▄                [/white]
[white]              ▄█████               [/white]
[white]  ▄▄▄███████████████████████▄▄     [/white]
[white] ████[/white][dim]░░[/dim][white]████[/white][cyan]◉[/cyan][white]████████████████████▄ [/white]
[white] █████████████████████████████████[/white]
[white]  ▀█████████████████████████████▀  [/white]
[white]      ▀▀████▀▀       ▀▀▀████▀▀     [/white]"""

ORCA_ASCII_COMPACT = """\
[white]       ▄█▄        [/white]
[white]     ▄█████       [/white]
[white] ▄███[cyan]◉[/cyan][white]████████▄  [/white]
[white] █████████████▌  [/white]
[white]  ▀███████████▀  [/white]
[white]     ▀▀▀▀▀▀▀     [/white]"""

LOGO_TEXT = """\
[bold white]  ██████╗ ██████╗  ██████╗ █████╗ [/bold white]
[bold white]  ██╔═══██╗██╔══██╗██╔════╝██╔══██╗[/bold white]
[bold white]  ██║   ██║██████╔╝██║     ███████║[/bold white]
[bold white]  ██║   ██║██╔══██╗██║     ██╔══██║[/bold white]
[bold white]  ╚██████╔╝██║  ██║╚██████╗██║  ██║[/bold white]
[bold white]   ╚═════╝ ╚═╝  ╚═╝ ╚═════╝╚═╝  ╚═╝[/bold white]"""

# ─────────────────────────────────────────────────────────────────────────────
#  Boot screen
# ─────────────────────────────────────────────────────────────────────────────

def boot_screen(variant: str = "core", model: str = "unknown", animate: bool = True) -> None:
    """Full-screen boot sequence with ASCII orca art."""
    variant_colors = {
        "nano":  ("cyan",    "NANO",  "fast · precise · terminal-native"),
        "core":  ("blue",    "CORE",  "full intelligence · memory · tools"),
        "ultra": ("magenta", "ULTRA", "multi-agent · apex orchestration"),
    }
    color, label, desc = variant_colors.get(variant, ("cyan", "CORE", ""))

    lines = [
        "",
        ORCA_ASCII,
        "",
        LOGO_TEXT,
        "",
        f"[dim]  Your AI. Your hardware. Your data.[/dim]",
        f"[dim]  No Anthropic. No OpenAI. No cloud.[/dim]",
        "",
        f"  [{color}]▓ {label} ▓[/{color}]  [dim]{desc}[/dim]",
        f"  [dim]model:[/dim] [bold]{model}[/bold]",
        "",
        f"  [dim]Type[/dim] [bold]/help[/bold] [dim]for commands  ·  Ctrl+C to exit[/dim]",
        "",
    ]

    if animate:
        with Live(console=console, refresh_per_second=20) as live:
            for i, line in enumerate(lines):
                time.sleep(0.04)
                live.update(Text.from_markup("\n".join(lines[:i+1])))
        # Hold the final frame
        console.print(Text.from_markup("\n".join(lines)))
    else:
        console.print(Text.from_markup("\n".join(lines)))


# ─────────────────────────────────────────────────────────────────────────────
#  Live dashboard  (orca status)
# ─────────────────────────────────────────────────────────────────────────────

def _model_panel(brain_info: dict) -> Panel:
    t = Table.grid(padding=(0, 2))
    t.add_column(style="dim", min_width=14)
    t.add_column(style="bold white")

    active = brain_info.get("active", "not connected")
    available = brain_info.get("available", False)
    models = brain_info.get("models", [])
    orca_models = [m for m in models if "orca" in m.lower()]

    t.add_row("status", f"[green]connected[/green]" if available else "[red]offline[/red]")
    t.add_row("active model", f"[cyan]{active}[/cyan]" if available else "[dim]—[/dim]")
    t.add_row("your orca", f"[green]{', '.join(orca_models)}[/green]" if orca_models else "[dim]not trained yet[/dim]")
    t.add_row("all models", f"[dim]{len(models)} available[/dim]" if models else "[dim]none[/dim]")
    t.add_row("endpoint", brain_info.get("host", "http://localhost:11434"))

    return Panel(t, title="[bold cyan]◈ Brain[/bold cyan]", border_style="cyan", box=box.ROUNDED)


def _memory_panel(mem_info: dict) -> Panel:
    t = Table.grid(padding=(0, 2))
    t.add_column(style="dim", min_width=14)
    t.add_column(style="bold white")

    t.add_row("short-term", f"{mem_info.get('short_term', 0)} messages in context")
    t.add_row("long-term", "[green]chromadb[/green]" if mem_info.get("chromadb") else "[dim]jsonl (fallback)[/dim]")
    t.add_row("episodes", f"{mem_info.get('episodes', 0)} saved sessions")
    t.add_row("semantic", f"{mem_info.get('facts', 0)} facts · {mem_info.get('concepts', 0)} concepts")
    t.add_row("data dir", str(mem_info.get("path", "~/.orca/memory")))

    return Panel(t, title="[bold blue]◈ Memory[/bold blue]", border_style="blue", box=box.ROUNDED)


def _data_panel(data_info: dict) -> Panel:
    t = Table.grid(padding=(0, 2))
    t.add_column(style="dim", min_width=14)
    t.add_column(style="bold white")

    total = data_info.get("total_raw", 0)
    curated = data_info.get("curated", 0)

    t.add_row("raw examples", f"[yellow]{total}[/yellow]")
    t.add_row("curated", f"[green]{curated}[/green]")
    t.add_row("synthetic", str(data_info.get("synthetic", 0)))
    t.add_row("core chat", str(data_info.get("core", 0)))
    t.add_row("data dir", str(data_info.get("path", "~/.orca/training")))

    goal = 1000
    bar_len = 20
    filled = min(int((total / goal) * bar_len), bar_len)
    bar = "[green]" + "█" * filled + "[/green][dim]" + "░" * (bar_len - filled) + "[/dim]"
    t.add_row("→ first train", f"{bar} [dim]{total}/{goal}[/dim]")

    return Panel(t, title="[bold yellow]◈ Training Data[/bold yellow]", border_style="yellow", box=box.ROUNDED)


def _tools_panel() -> Panel:
    t = Table.grid(padding=(0, 2))
    t.add_column(style="cyan", min_width=16)
    t.add_column(style="dim")

    tools = [
        ("web_search", "DuckDuckGo · no API key"),
        ("run_code", "Python · sandboxed subprocess"),
        ("shell", "non-destructive commands only"),
        ("read_file", "local filesystem access"),
        ("write_file", "local filesystem write"),
        ("memory_recall", "semantic long-term search"),
    ]
    for name, desc in tools:
        t.add_row(f"[green]✓[/green] {name}", desc)

    return Panel(t, title="[bold green]◈ Tools[/bold green]", border_style="green", box=box.ROUNDED)


def _sessions_panel(sessions: list[str]) -> Panel:
    t = Table.grid(padding=(0, 2))
    t.add_column(style="cyan", min_width=10)
    t.add_column(style="dim")

    if not sessions:
        t.add_row("[dim]no sessions yet[/dim]", "")
        t.add_row("", "[dim]start: orca core chat[/dim]")
    else:
        for sid in sessions[:5]:
            t.add_row(sid[:8], f"[dim]{sid}[/dim]")
        if len(sessions) > 5:
            t.add_row(f"[dim]+{len(sessions) - 5} more[/dim]", "")

    return Panel(t, title=f"[bold]◈ Sessions[/bold] [dim]({len(sessions)} total)[/dim]", border_style="dim", box=box.ROUNDED)


def status_dashboard() -> None:
    """Live status dashboard — `orca status`."""
    console.print()
    console.print(Align.center(Text.from_markup(LOGO_TEXT)))
    console.print(Align.center(Text.from_markup(
        f"\n  [dim]Apex intelligence. Zero noise.[/dim]\n"
        f"  [dim]{datetime.now().strftime('%Y-%m-%d  %H:%M:%S')}[/dim]\n"
    )))
    console.print(Rule(style="dim"))
    console.print()

    # Collect info
    brain_info: dict = {}
    try:
        from orca.brain.providers import OrcaBrain
        from orca.config import CONFIG
        b = OrcaBrain()
        brain_info["host"] = CONFIG.ollama.host
        brain_info["available"] = b.is_available()
        if brain_info["available"]:
            brain_info["active"] = b.name
            brain_info["models"] = b.list_models()
    except Exception:
        pass

    mem_info: dict = {"path": Path.home() / ".orca" / "memory"}
    try:
        from orca.brain.memory import EpisodicMemory, SemanticMemory
        sessions = EpisodicMemory.list_sessions()
        mem_info["episodes"] = len(sessions)
        sem = SemanticMemory()
        mem_info["facts"] = len([k for k in sem._cache if k.startswith("fact:")])
        mem_info["concepts"] = len(sem.all_concepts())
        try:
            import chromadb
            mem_info["chromadb"] = True
        except ImportError:
            mem_info["chromadb"] = False
    except Exception:
        sessions = []

    data_info: dict = {"path": Path.home() / ".orca" / "training"}
    try:
        from orca.data.collector import DataCollector
        counts = DataCollector.count_examples()
        data_info["total_raw"] = sum(counts.values())
        data_info["synthetic"] = counts.get("synthetic", 0)
        data_info["core"] = counts.get("core", 0)
        from orca.data.curator import DataCurator
        stats = DataCurator().stats()
        data_info["curated"] = stats.get("examples", 0)
    except Exception:
        pass

    # Layout
    console.print(Columns([
        _model_panel(brain_info),
        _memory_panel(mem_info),
    ], equal=True, expand=True))
    console.print()
    console.print(Columns([
        _data_panel(data_info),
        _tools_panel(),
    ], equal=True, expand=True))
    console.print()
    console.print(_sessions_panel(sessions))
    console.print()

    # Quick commands
    cmds = Table.grid(padding=(0, 3))
    cmds.add_column(style="cyan bold")
    cmds.add_column(style="dim")
    quick = [
        ("orca core chat",        "start a god-mode session"),
        ("orca data seed --n 500","generate 500 synthetic training examples"),
        ("orca data curate",      "clean and deduplicate data"),
        ("orca train run",        "start QLoRA fine-tune (needs GPU)"),
        ("orca backend",          "show active model details"),
    ]
    for cmd, desc in quick:
        cmds.add_row(cmd, desc)
    console.print(Panel(cmds, title="[dim]Quick Commands[/dim]", border_style="dim", box=box.SIMPLE))
    console.print()


# ─────────────────────────────────────────────────────────────────────────────
#  Session history browser
# ─────────────────────────────────────────────────────────────────────────────

def history_browser(n: int = 20) -> None:
    """Interactive session history view — `orca history`."""
    from orca.brain.memory import EpisodicMemory

    sessions = EpisodicMemory.list_sessions()
    if not sessions:
        console.print("[dim]No sessions saved yet.[/dim]")
        console.print("Start one: [bold cyan]orca core chat[/bold cyan]")
        return

    console.print(f"\n[bold]Saved Sessions[/bold] [dim]({len(sessions)} total)[/dim]\n")

    table = Table(box=box.SIMPLE_HEAD, show_header=True, header_style="bold dim")
    table.add_column("#", style="dim", width=4)
    table.add_column("Session ID", style="cyan", width=36)
    table.add_column("ID Short", style="bold", width=10)
    table.add_column("Messages", justify="right", width=10)
    table.add_column("Summary", style="dim")

    for i, sid in enumerate(sessions[:n]):
        ep = EpisodicMemory(sid).load()
        msgs = len(ep.messages) if ep else "?"
        summary = (ep.summary[:60] + "…") if ep and ep.summary else "[dim]—[/dim]"
        tags = f"  [blue]{' '.join('#'+t for t in ep.tags[:3])}[/blue]" if ep and ep.tags else ""
        table.add_row(str(i + 1), sid, sid[:8], str(msgs), summary + tags)

    console.print(table)

    if len(sessions) > n:
        console.print(f"[dim]  … and {len(sessions) - n} more. Use --n to see more.[/dim]")

    console.print()
    console.print("[dim]Resume a session: [bold]orca core chat --session <ID>[/bold][/dim]")
    console.print()


# ─────────────────────────────────────────────────────────────────────────────
#  Streaming display helpers
# ─────────────────────────────────────────────────────────────────────────────

@contextmanager
def tool_spinner(label: str):
    """Show a spinner while a tool is running."""
    with Progress(
        SpinnerColumn("dots"),
        TextColumn(f"[dim]{label}[/dim]"),
        console=console,
        transient=True,
    ) as p:
        p.add_task("", total=None)
        yield


def stream_with_cursor(chunks: Iterator[str], prefix: str = "") -> str:
    """Print streaming chunks with a blinking cursor effect."""
    full = []
    if prefix:
        console.print(prefix, end="")
    try:
        for chunk in chunks:
            print(chunk, end="", flush=True)
            full.append(chunk)
    finally:
        print()
    return "".join(full)


def thinking_block(text: str, visible: bool = True) -> None:
    """Render a collapsible thinking/trace block."""
    if not visible:
        return
    panel = Panel(
        Text(text, style="dim italic"),
        title="[dim]◈ thinking[/dim]",
        border_style="dim",
        box=box.SIMPLE,
    )
    console.print(panel)


def tool_call_display(tool: str, input_summary: str, result_summary: str) -> None:
    """Render a tool call inline."""
    console.print(
        f"  [dim]▸[/dim] [cyan]{tool}[/cyan][dim]({input_summary})[/dim]"
        f"  [dim]→[/dim] [dim]{result_summary[:120]}[/dim]"
    )


def agent_pod_display(agents: list[str]) -> None:
    """Show the Ultra agent pod launching."""
    console.print()
    console.print(Panel(
        " ".join(f"[cyan]{a.upper()}[/cyan]" for a in agents),
        title="[magenta]◈ Pod Launching[/magenta]",
        border_style="magenta",
        box=box.ROUNDED,
    ))
    console.print()


def grade_display(score: int, iterations: int) -> None:
    """Show quality grade bar for Ultra output."""
    bar_len = 30
    filled = int((score / 100) * bar_len)
    color = "green" if score >= 80 else "yellow" if score >= 65 else "red"
    bar = f"[{color}]" + "█" * filled + f"[/{color}][dim]" + "░" * (bar_len - filled) + "[/dim]"
    console.print(
        f"\n  [dim]quality[/dim] {bar} [{color}]{score}/100[/{color}]"
        f"  [dim]·  {iterations} iteration{'s' if iterations != 1 else ''}[/dim]\n"
    )


# ─────────────────────────────────────────────────────────────────────────────
#  Inline help panel
# ─────────────────────────────────────────────────────────────────────────────

def chat_help_panel(variant: str = "core") -> None:
    """Print the in-chat help panel."""
    t = Table.grid(padding=(0, 3))
    t.add_column(style="cyan bold", min_width=18)
    t.add_column(style="dim")

    shared = [
        ("/help",           "show this panel"),
        ("exit / quit",     "end session"),
    ]
    core_cmds = [
        ("/web <query>",    "search the web"),
        ("/run <code>",     "execute Python"),
        ("/shell <cmd>",    "run shell command"),
        ("/think <prompt>", "deep reasoning pass"),
        ("/save",           "save session to disk"),
        ("/recall <query>", "search long-term memory"),
        ("/thoughts on|off","toggle reasoning trace"),
        ("/clear",          "clear conversation history"),
        ("/session",        "show current session ID"),
    ]
    ultra_cmds = [
        ("/agents",         "list available agent types"),
        ("/grade",          "show last quality score"),
    ]

    cmds = shared + (core_cmds if variant == "core" else ultra_cmds if variant == "ultra" else [])
    for cmd, desc in cmds:
        t.add_row(cmd, desc)

    console.print(Panel(t, title="[dim]Commands[/dim]", border_style="dim", box=box.SIMPLE))


# ─────────────────────────────────────────────────────────────────────────────
#  Error / warning displays
# ─────────────────────────────────────────────────────────────────────────────

def ollama_offline_panel() -> None:
    console.print(Panel(
        "[red bold]Ollama is offline.[/red bold]\n\n"
        "[dim]Start Ollama:[/dim]  [bold]ollama serve[/bold]\n"
        "[dim]Install:[/dim]       [bold]brew install ollama[/bold]\n"
        "[dim]Pull a model:[/dim]  [bold]ollama pull llama3.1:8b[/bold]\n\n"
        "[dim]Then run:[/dim]      [bold cyan]orca core chat[/bold cyan]",
        title="[red]◈ Brain Offline[/red]",
        border_style="red",
        box=box.ROUNDED,
    ))


def model_not_found_panel(requested: str, available: list[str]) -> None:
    avail_str = "  " + "\n  ".join(available[:8]) if available else "  [dim]none[/dim]"
    console.print(Panel(
        f"[yellow]Model not found:[/yellow] [bold]{requested}[/bold]\n\n"
        f"[dim]Available models:[/dim]\n{avail_str}\n\n"
        f"[dim]Pull a model:[/dim]  [bold]ollama pull {requested}[/bold]",
        title="[yellow]◈ Model Not Found[/yellow]",
        border_style="yellow",
        box=box.ROUNDED,
    ))
