"""
Orca Parallel Seed Pipeline — async multi-worker data generator.

Spawns N async workers, each pulling from domain queues.
Rich Live display shows per-worker progress + live total.
100% local — no external API calls.
"""
from __future__ import annotations

import asyncio
import json
import re
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from rich.console import Console
from rich.live import Live
from rich.table import Table
from rich.text import Text
from rich import box

from orca.config import ORCA_HOME
from orca.data.collector import Conversation, RAW_DATA_DIR
from orca.data.seeds import Domain, build_prompt, sample_domains, ALL_DOMAINS

console = Console()


# ─────────────────────────────────────────────────────────────────────────────
#  Worker state
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class WorkerState:
    id: int
    domain: str = "—"
    done: int = 0
    target: int = 0
    skipped: int = 0
    status: str = "idle"
    last: str = ""
    started_at: float = field(default_factory=time.time)

    @property
    def elapsed(self) -> str:
        s = int(time.time() - self.started_at)
        return f"{s//60:02d}:{s%60:02d}"

    @property
    def rate(self) -> str:
        elapsed = time.time() - self.started_at
        if elapsed < 1 or self.done == 0:
            return "—"
        r = self.done / elapsed * 60
        return f"{r:.0f}/min"


@dataclass
class PipelineResult:
    total_generated: int = 0
    total_skipped: int = 0
    by_domain: dict[str, int] = field(default_factory=dict)
    output_files: list[str] = field(default_factory=list)
    duration_sec: float = 0.0


# ─────────────────────────────────────────────────────────────────────────────
#  Output writer (thread-safe via asyncio lock)
# ─────────────────────────────────────────────────────────────────────────────

class _Writer:
    def __init__(self):
        self._lock = asyncio.Lock()
        self._files: dict[str, Path] = {}
        self._counts: dict[str, int] = {}

    def _path(self, domain: str) -> Path:
        if domain not in self._files:
            stamp = time.strftime("%Y%m%d_%H%M")
            self._files[domain] = RAW_DATA_DIR / f"seed_{domain}_{stamp}.jsonl"
        return self._files[domain]

    async def write(self, domain: str, conv: dict) -> None:
        async with self._lock:
            path = self._path(domain)
            with open(path, "a") as f:
                f.write(json.dumps(conv) + "\n")
            self._counts[domain] = self._counts.get(domain, 0) + 1

    def files(self) -> list[str]:
        return [str(p) for p in self._files.values()]

    def counts(self) -> dict[str, int]:
        return dict(self._counts)


# ─────────────────────────────────────────────────────────────────────────────
#  Robust JSON extractor — handles all LLM quirks
# ─────────────────────────────────────────────────────────────────────────────

def _fix_string_literals(text: str) -> str:
    """
    Escape literal control chars that appear INSIDE JSON string values.
    The model sometimes outputs bare newlines inside strings instead of \\n.
    """
    result = []
    in_string = False
    escape_next = False
    for ch in text:
        if escape_next:
            result.append(ch)
            escape_next = False
        elif ch == "\\" and in_string:
            result.append(ch)
            escape_next = True
        elif ch == '"':
            in_string = not in_string
            result.append(ch)
        elif in_string and ord(ch) < 0x20:
            # Replace literal control chars with their JSON escape equivalents
            escapes = {"\n": "\\n", "\r": "\\r", "\t": "\\t"}
            result.append(escapes.get(ch, f"\\u{ord(ch):04x}"))
        else:
            result.append(ch)
    return "".join(result)


def _extract_json(text: str) -> dict | None:
    """
    Multi-strategy JSON extraction for LLM outputs.
    Handles: markdown fences, literal control chars, extra trailing text.
    """
    # Strip markdown code fences
    text = re.sub(r"```[a-z]*\n?", "", text).strip()

    # Find first {
    start = text.find("{")
    if start == -1:
        return None
    text = text[start:]

    # Strategy 1: direct parse — works for well-formed output
    try:
        obj, _ = json.JSONDecoder().raw_decode(text)
        return obj
    except json.JSONDecodeError:
        pass

    # Strategy 2: fix literal control chars inside strings, then retry
    fixed = _fix_string_literals(text)
    try:
        obj, _ = json.JSONDecoder().raw_decode(fixed)
        return obj
    except json.JSONDecodeError:
        pass

    # Strategy 3: also strip any remaining non-printable chars
    clean = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", fixed)
    end = clean.rfind("}")
    if end == -1:
        return None
    try:
        return json.loads(clean[: end + 1])
    except json.JSONDecodeError:
        return None


# ─────────────────────────────────────────────────────────────────────────────
#  Conversation parser — handles QA and multi-turn JSON
# ─────────────────────────────────────────────────────────────────────────────

def _parse_response(text: str, domain: Domain) -> Conversation | None:
    data = _extract_json(text)
    if data is None:
        return None

    conv = Conversation(
        source="seed",
        variant=domain.name,
        metadata={"domain": domain.name},
    )

    if domain.multi_turn:
        # Multi-turn: {"conversation": [{"role": ..., "content": ...}, ...]}
        turns = data.get("conversation") or data.get("conversations", [])
        if not turns or len(turns) < 2:
            return None
        has_human = any(t.get("role") in ("human", "user") for t in turns)
        has_gpt = any(t.get("role") in ("gpt", "assistant") for t in turns)
        if not has_human or not has_gpt:
            return None

        from orca.data.collector import ORCA_SYSTEM_PROMPT
        conv.add_system(ORCA_SYSTEM_PROMPT)
        for turn in turns:
            role = turn.get("role", "")
            content = turn.get("content", "").strip()
            if not content:
                continue
            if role in ("human", "user"):
                conv.add_human(content)
            elif role in ("gpt", "assistant"):
                conv.add_gpt(content)
    else:
        # Single-turn QA: {"question": ..., "answer": ...}
        q = data.get("question", "")
        a = data.get("answer", "")
        # Model sometimes returns lists — join them
        if isinstance(q, list):
            q = " ".join(str(x) for x in q)
        if isinstance(a, list):
            a = "\n".join(str(x) for x in a)
        q, a = str(q).strip(), str(a).strip()
        # Minimum: question > 5 chars, answer > 5 chars (bash one-liners are valid!)
        if not q or not a or len(q) < 5 or len(a) < 5:
            return None

        from orca.data.collector import ORCA_SYSTEM_PROMPT
        conv.add_system(ORCA_SYSTEM_PROMPT)
        conv.add_human(q)
        conv.add_gpt(a)

    return conv if conv.is_valid() else None


# ─────────────────────────────────────────────────────────────────────────────
#  Single async worker
# ─────────────────────────────────────────────────────────────────────────────

async def _worker(
    worker_id: int,
    tasks: asyncio.Queue,
    state: WorkerState,
    writer: _Writer,
    brain,
    temperature: float = 0.85,
    max_tokens: int = 800,
) -> None:
    state.status = "running"

    while True:
        try:
            domain, remaining = tasks.get_nowait()
        except asyncio.QueueEmpty:
            break

        state.domain = domain.name
        state.target = remaining + state.done

        system_prompt, user_prompt = build_prompt(domain)
        state.last = user_prompt[:50]
        state.status = "generating"

        try:
            response = await asyncio.to_thread(
                brain.complete,
                [{"role": "user", "content": user_prompt}],
                system_prompt,
                temperature,
                max_tokens,
            )

            conv = _parse_response(response, domain)
            if conv:
                await writer.write(domain.name, conv.to_dict())
                state.done += 1
                state.status = "running"
            else:
                state.skipped += 1
                state.last = "bad json"
                state.status = "running"

        except RuntimeError as exc:
            msg = str(exc)
            # Connectivity error — abort immediately, don't waste time
            if "not running" in msg or "disconnected" in msg or "Cannot reach" in msg:
                state.status = "offline"
                state.last = "Ollama offline — run: ollama serve"
                # Drain the queue so other workers stop too
                while not tasks.empty():
                    try:
                        tasks.get_nowait()
                        tasks.task_done()
                    except asyncio.QueueEmpty:
                        break
                tasks.task_done()
                return
            state.skipped += 1
            state.last = f"err: {msg[:50]}"
            state.status = "running"
            await asyncio.sleep(0.5)

        except Exception as exc:
            state.skipped += 1
            state.last = f"{type(exc).__name__}: {str(exc)[:40]}"
            state.status = "running"
            await asyncio.sleep(0.5)

        tasks.task_done()

    state.status = "done"


# ─────────────────────────────────────────────────────────────────────────────
#  Progress display
# ─────────────────────────────────────────────────────────────────────────────

def _make_table(workers: list[WorkerState], total_target: int) -> Table:
    t = Table(box=box.SIMPLE_HEAD, show_header=True, header_style="bold dim", expand=True)
    t.add_column("#", width=4, style="dim")
    t.add_column("Domain", width=14, style="cyan")
    t.add_column("Progress", width=18)
    t.add_column("Rate", width=8, style="dim")
    t.add_column("Skip", width=6, style="dim")
    t.add_column("Time", width=8, style="dim")
    t.add_column("Status", width=12)
    t.add_column("Last", style="dim")

    total_done = sum(w.done for w in workers)
    total_skip = sum(w.skipped for w in workers)

    for w in workers:
        bar_len = 12
        pct = w.done / max(w.target, 1)
        filled = int(pct * bar_len)
        bar = "[green]" + "█" * filled + "[/green][dim]" + "░" * (bar_len - filled) + "[/dim]"
        prog = f"{bar} [bold]{w.done}[/bold][dim]/{w.target}[/dim]"

        color = {"running": "green", "generating": "yellow", "done": "dim", "idle": "dim"}.get(w.status, "dim")
        status_text = f"[{color}]{w.status}[/{color}]"

        t.add_row(
            str(w.id + 1),
            w.domain,
            Text.from_markup(prog),
            w.rate,
            str(w.skipped),
            w.elapsed,
            Text.from_markup(status_text),
            w.last[:35],
        )

    # Total row
    bar_len = 12
    pct = total_done / max(total_target, 1)
    filled = int(pct * bar_len)
    bar = "[bold green]" + "█" * filled + "[/bold green][dim]" + "░" * (bar_len - filled) + "[/dim]"
    prog = f"{bar} [bold]{total_done}/{total_target}[/bold]"
    t.add_row(
        "∑",
        "[bold]total[/bold]",
        Text.from_markup(prog),
        "—",
        str(total_skip),
        workers[0].elapsed if workers else "—",
        "",
        "",
    )

    return t


# ─────────────────────────────────────────────────────────────────────────────
#  Main pipeline
# ─────────────────────────────────────────────────────────────────────────────

class SeedPipeline:
    """
    Parallel async seed generator.

    Usage:
        pipeline = SeedPipeline(brain, n=1000, workers=4)
        result = pipeline.run()

    Note: Ollama processes one request at a time on a single machine.
    Multiple workers help with I/O overlap but not generation speed.
    For faster generation, reduce max_tokens (--fast flag in CLI).
    """

    def __init__(
        self,
        brain,
        n: int = 500,
        workers: int = 2,
        domains: list[str] | None = None,
        temperature: float = 0.85,
        max_tokens: int = 800,
    ):
        self.brain = brain
        self.n = n
        self.workers = min(workers, n)
        self.domains = domains
        self.temperature = temperature
        self.max_tokens = max_tokens

    def run(self) -> PipelineResult:
        return asyncio.run(self._run())

    async def _run(self) -> PipelineResult:
        start = time.time()

        # Fast connectivity check before launching workers
        if not self.brain.is_available():
            console.print("[red bold]✗ Ollama is not running.[/red bold]")
            console.print("  Start it:  [bold]ollama serve[/bold]")
            console.print("  Then retry: [bold]orca data seed --n ...[/bold]")
            raise RuntimeError("Ollama offline")

        # Verify the resolved model exists
        model_name = self.brain.name
        console.print(f"  [dim]brain:[/dim]   [bold]{model_name}[/bold] ✓\n")

        # Distribute work
        domain_counts = sample_domains(self.n, self.domains)

        # Build task queue — one item per example
        queue: asyncio.Queue = asyncio.Queue()
        for domain, count in domain_counts:
            for i in range(count):
                await queue.put((domain, count - i))

        total = queue.qsize()

        # Worker states
        states = [WorkerState(id=i, target=total // self.workers) for i in range(self.workers)]
        writer = _Writer()

        # Spin up workers + live display together
        with Live(console=console, refresh_per_second=4, transient=False) as live:
            worker_tasks = [
                asyncio.create_task(
                    _worker(i, queue, states[i], writer, self.brain, self.temperature, self.max_tokens)
                )
                for i in range(self.workers)
            ]

            while not all(s.status == "done" for s in states):
                live.update(_make_table(states, total))
                await asyncio.sleep(0.25)

            await asyncio.gather(*worker_tasks)
            live.update(_make_table(states, total))

        duration = time.time() - start
        counts = writer.counts()

        return PipelineResult(
            total_generated=sum(s.done for s in states),
            total_skipped=sum(s.skipped for s in states),
            by_domain=counts,
            output_files=writer.files(),
            duration_sec=duration,
        )


# ─────────────────────────────────────────────────────────────────────────────
#  Single-shot domain preview (for testing)
# ─────────────────────────────────────────────────────────────────────────────

def preview_domain(domain_name: str, brain) -> str:
    """Generate one example from a domain and print it — for testing."""
    from orca.data.seeds import get_domain
    domain = get_domain(domain_name)
    system, user = build_prompt(domain)
    return brain.complete(
        [{"role": "user", "content": user}],
        system,
        temperature=0.85,
    )
