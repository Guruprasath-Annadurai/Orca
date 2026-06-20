"""
Orca Doctor — system health check, auto-fixer, and first-run setup wizard.

Usage (CLI):
  orca doctor            → full health report
  orca doctor --fix      → auto-repair failing checks
  orca doctor --wizard   → interactive first-run setup

Designed to be the single source of truth for "is Orca healthy?"
"""
from __future__ import annotations

import importlib
import os
import platform
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

from rich.console import Console
from rich.panel import Panel
from rich.rule import Rule
from rich import box

console = Console()

# ─────────────────────────────────────────────────────────────────────────────
#  Models the wizard can pull
# ─────────────────────────────────────────────────────────────────────────────

RECOMMENDED_MODELS = [
    ("llama3.1:8b",  "4.7 GB", "Best overall — recommended for most hardware"),
    ("mistral:7b",   "4.1 GB", "Fast and efficient, great for chat"),
    ("codellama:7b", "3.8 GB", "Optimized for code generation"),
    ("phi3:mini",    "2.3 GB", "Lightweight — 8 GB RAM machines"),
]

REQUIRED_PACKAGES = [
    ("httpx",              "httpx"),
    ("rich",               "rich"),
    ("typer",              "typer"),
    ("pydantic",           "pydantic"),
    ("python-dotenv",      "dotenv"),
    ("aiofiles",           "aiofiles"),
    ("diskcache",          "diskcache"),
    ("fastapi",            "fastapi"),
    ("uvicorn",            "uvicorn"),
    ("python-multipart",   "multipart"),
]

OPTIONAL_PACKAGES = [
    ("chromadb", "chromadb",  "vector long-term memory"),
    ("mcp",      "mcp",       "MCP tool protocol"),
]

SETUP_SENTINEL = Path.home() / ".orca" / ".setup_complete"


# ─────────────────────────────────────────────────────────────────────────────
#  Data structures
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class Check:
    name: str
    status: str       # "ok" | "warn" | "fail" | "info"
    message: str
    detail: str = ""
    fix: str = ""     # shell command to run (or special keyword)


@dataclass
class DoctorReport:
    checks: list[Check] = field(default_factory=list)

    def add(self, check: Check) -> Check:
        self.checks.append(check)
        return check

    @property
    def ok_count(self) -> int:
        return sum(1 for c in self.checks if c.status == "ok")

    @property
    def warn_count(self) -> int:
        return sum(1 for c in self.checks if c.status == "warn")

    @property
    def fail_count(self) -> int:
        return sum(1 for c in self.checks if c.status == "fail")

    @property
    def healthy(self) -> bool:
        return self.fail_count == 0


# ─────────────────────────────────────────────────────────────────────────────
#  Doctor — runs all checks
# ─────────────────────────────────────────────────────────────────────────────

class OrcaDoctor:
    """Runs all system health checks and returns a DoctorReport."""

    def __init__(self):
        self.report = DoctorReport()
        self._ollama_online = False
        self._available_models: list[str] = []

    def run_all(self) -> DoctorReport:
        self._check_python()
        self._check_data_dir()
        self._check_config()
        self._check_deps()
        self._check_optional_deps()
        self._check_ollama_installed()
        self._check_ollama_running()
        self._check_models()
        self._check_training_data()
        self._check_hardware()
        self._check_web_server()
        return self.report

    # ── Individual checks ────────────────────────────────────────────────────

    def _check_python(self):
        v = sys.version_info
        ver = f"{v.major}.{v.minor}.{v.micro}"
        if v.major == 3 and v.minor >= 11:
            self.report.add(Check("python", "ok", ver))
        elif v.major == 3 and v.minor >= 9:
            self.report.add(Check("python", "warn", ver,
                detail="Python 3.11+ recommended for best compatibility."))
        else:
            self.report.add(Check("python", "fail", ver,
                detail="Python 3.11+ required.",
                fix="Install Python 3.11 from https://python.org"))

    def _check_data_dir(self):
        try:
            from orca.config import ORCA_HOME
            if not ORCA_HOME.exists():
                _ensure_dirs(ORCA_HOME)
                self.report.add(Check("data directory", "ok", f"{ORCA_HOME}  (created)"))
            else:
                # Ensure all required subdirectories exist
                _ensure_dirs(ORCA_HOME)
                self.report.add(Check("data directory", "ok", str(ORCA_HOME)))
        except Exception as e:
            self.report.add(Check("data directory", "fail", str(e),
                fix="mkdir -p ~/.orca/training/raw ~/.orca/memory/episodic ~/.orca/models"))

    def _check_config(self):
        try:
            from orca.config import CONFIG
            self.report.add(Check("config", "ok",
                f"core={CONFIG.ollama.model_core}  nano={CONFIG.ollama.model_nano}"))
        except Exception as e:
            self.report.add(Check("config", "warn", f"config load error: {e}"))

    def _check_deps(self):
        missing = []
        for pip_name, import_name in REQUIRED_PACKAGES:
            try:
                importlib.import_module(import_name)
            except ImportError:
                missing.append(pip_name)
        if missing:
            self.report.add(Check("core packages", "fail",
                f"{len(missing)} missing: {', '.join(missing)}",
                fix="uv pip install " + " ".join(missing)))
        else:
            self.report.add(Check("core packages", "ok",
                f"all {len(REQUIRED_PACKAGES)} installed"))

    def _check_optional_deps(self):
        for pip_name, import_name, purpose in OPTIONAL_PACKAGES:
            try:
                importlib.import_module(import_name)
                self.report.add(Check(f"optional: {pip_name}", "ok", purpose))
            except ImportError:
                self.report.add(Check(f"optional: {pip_name}", "warn",
                    f"not installed — {purpose} unavailable",
                    fix=f"uv pip install {pip_name}"))

    def _check_ollama_installed(self):
        path = shutil.which("ollama")
        if path:
            self.report.add(Check("ollama binary", "ok", path))
        else:
            self.report.add(Check("ollama binary", "fail",
                "not found in PATH",
                detail="Ollama is required for all AI inference.",
                fix="brew install ollama"))

    def _check_ollama_running(self):
        try:
            import httpx
            r = httpx.get("http://localhost:11434/api/tags", timeout=3.0)
            if r.status_code == 200:
                self._available_models = [m["name"] for m in r.json().get("models", [])]
                self._ollama_online = True
                self.report.add(Check("ollama server", "ok", "http://localhost:11434"))
                return
        except Exception:
            pass
        self._ollama_online = False
        self.report.add(Check("ollama server", "fail",
            "not reachable at http://localhost:11434",
            detail="Start Ollama to enable AI inference.",
            fix="ollama serve"))

    def _check_models(self):
        if not self._ollama_online:
            self.report.add(Check("models", "info", "skipped — Ollama offline"))
            return
        if not self._available_models:
            self.report.add(Check("models", "fail",
                "no models installed",
                detail="Pull at least one model to use Orca.",
                fix="ollama pull llama3.1:8b"))
            return

        shown = self._available_models[:5]
        label = ", ".join(shown)
        if len(self._available_models) > 5:
            label += f"  +{len(self._available_models) - 5} more"
        self.report.add(Check("models", "ok", label))

        orca_models = [m for m in self._available_models if "orca" in m.lower()]
        if orca_models:
            self.report.add(Check("orca fine-tune", "ok", ", ".join(orca_models)))
        else:
            self.report.add(Check("orca fine-tune", "warn",
                "no custom Orca model yet",
                detail="Generate data and fine-tune to build your personalized model.",
                fix="orca data seed --n 1000"))

    def _check_training_data(self):
        try:
            from orca.config import ORCA_HOME
            raw_dir = ORCA_HOME / "training" / "raw"
            total = 0
            if raw_dir.exists():
                for f in raw_dir.glob("*.jsonl"):
                    if f.stat().st_size > 0:
                        with open(f) as fh:
                            total += sum(1 for _ in fh)

            goal = 1000
            bar = _progress_bar(total, goal)

            if total >= goal:
                self.report.add(Check("training data", "ok",
                    f"{total} examples  {bar}  ready to fine-tune"))
            elif total > 0:
                needed = goal - total
                self.report.add(Check("training data", "warn",
                    f"{total}/{goal} examples  {bar}",
                    detail=f"Need {needed} more before fine-tuning.",
                    fix=f"orca data seed --n {needed}"))
            else:
                self.report.add(Check("training data", "warn",
                    "no training data generated yet",
                    fix="orca data seed --n 1000"))
        except Exception as e:
            self.report.add(Check("training data", "warn", f"could not read: {e}"))

    def _check_hardware(self):
        # RAM
        try:
            ram_gb = _get_ram_gb()
            if ram_gb >= 16:
                self.report.add(Check("RAM", "ok", f"{ram_gb:.0f} GB"))
            elif ram_gb >= 8:
                self.report.add(Check("RAM", "warn", f"{ram_gb:.0f} GB",
                    detail="8 GB is minimum — use phi3:mini or quantized 4-bit models."))
            elif ram_gb > 0:
                self.report.add(Check("RAM", "warn", f"{ram_gb:.0f} GB",
                    detail="Low RAM — use phi3:mini (2.3 GB) only."))
            else:
                self.report.add(Check("RAM", "info", "could not detect"))
        except Exception:
            self.report.add(Check("RAM", "info", "could not detect"))

        # GPU / Apple Silicon
        gpu_name, has_gpu = _detect_gpu()
        if has_gpu:
            self.report.add(Check("GPU / accelerator", "ok", gpu_name))
        else:
            self.report.add(Check("GPU / accelerator", "warn",
                "not detected — CPU inference only",
                detail="Fine-tuning requires a GPU. Use: orca train cloud --ssh ..."))

    def _check_web_server(self):
        try:
            import fastapi
            import uvicorn
            self.report.add(Check("web server", "ok",
                f"FastAPI {fastapi.__version__}  ·  uvicorn {uvicorn.__version__}"))
        except ImportError as e:
            self.report.add(Check("web server", "fail", str(e),
                fix="uv pip install fastapi uvicorn python-multipart"))


# ─────────────────────────────────────────────────────────────────────────────
#  Rich report renderer
# ─────────────────────────────────────────────────────────────────────────────

_ICON = {
    "ok":   "[green]✓[/green]",
    "warn": "[yellow]⚠[/yellow]",
    "fail": "[red]✗[/red]",
    "info": "[dim]·[/dim]",
}

_COLOR = {
    "ok":   "white",
    "warn": "yellow",
    "fail": "red",
    "info": "dim",
}

_SECTIONS: list[tuple[str, list[str]]] = [
    ("SYSTEM",          ["python", "data directory", "config"]),
    ("PACKAGES",        ["core packages", "optional: chromadb", "optional: mcp"]),
    ("BRAIN (OLLAMA)",  ["ollama binary", "ollama server", "models", "orca fine-tune"]),
    ("TRAINING DATA",   ["training data"]),
    ("HARDWARE",        ["RAM", "GPU / accelerator"]),
    ("WEB SERVER",      ["web server"]),
]


def print_report(report: DoctorReport, verbose: bool = False) -> None:
    console.print()
    console.print(Panel(
        "[bold white]ORCA DOCTOR — SYSTEM HEALTH REPORT[/bold white]",
        border_style="dim",
        box=box.SIMPLE,
        padding=(0, 1),
    ))

    check_map = {c.name: c for c in report.checks}

    for section_name, names in _SECTIONS:
        checks = [check_map[n] for n in names if n in check_map]
        if not checks:
            continue

        console.print(f"\n  [dim]{section_name}[/dim]")
        console.print(f"  [dim]{'─' * 48}[/dim]")

        for c in checks:
            icon  = _ICON[c.status]
            color = _COLOR[c.status]
            console.print(
                f"  {icon}  [{color}]{c.name:<28}[/{color}]  [dim]{c.message}[/dim]"
            )
            if c.detail and (verbose or c.status in ("fail", "warn")):
                console.print(f"            [dim italic]{c.detail}[/dim italic]")
            if c.fix and c.status in ("fail", "warn"):
                console.print(f"            [dim]→[/dim] [cyan]{c.fix}[/cyan]")

    # Summary bar
    console.print()
    console.print(f"  [dim]{'─' * 48}[/dim]")
    parts: list[str] = []
    if report.ok_count:
        parts.append(f"[green]{report.ok_count} passed[/green]")
    if report.warn_count:
        parts.append(f"[yellow]{report.warn_count} warning{'s' if report.warn_count != 1 else ''}[/yellow]")
    if report.fail_count:
        parts.append(f"[red]{report.fail_count} failed[/red]")
    console.print("  " + "  ·  ".join(parts))

    if report.fail_count:
        console.print(f"\n  Auto-repair: [cyan bold]orca doctor --fix[/cyan bold]")
        console.print(f"  Full wizard: [cyan]orca doctor --wizard[/cyan]")
    elif report.warn_count:
        console.print(f"\n  Warnings can be addressed with: [cyan]orca doctor --fix[/cyan]")
    else:
        console.print("\n  [green bold]All systems nominal. Orca is ready.[/green bold]")
        _mark_setup_complete()
    console.print()


# ─────────────────────────────────────────────────────────────────────────────
#  Auto-fixer
# ─────────────────────────────────────────────────────────────────────────────

def run_fix(report: DoctorReport, yes: bool = False) -> None:
    console.print()
    console.print(Panel(
        "[bold white]ORCA DOCTOR — AUTO FIX[/bold white]",
        border_style="dim",
        box=box.SIMPLE,
        padding=(0, 1),
    ))

    fixable = [c for c in report.checks if c.status in ("fail", "warn") and c.fix]
    if not fixable:
        console.print("\n  [green]Nothing to fix — all systems operational.[/green]\n")
        return

    for c in fixable:
        console.print(f"\n  {_ICON[c.status]}  [bold]{c.name}[/bold]")
        console.print(f"     issue: [dim]{c.message}[/dim]")
        if c.detail:
            console.print(f"     note:  [dim italic]{c.detail}[/dim italic]")
        console.print(f"     fix:   [cyan]{c.fix}[/cyan]")

        if not yes:
            try:
                answer = console.input("     Apply? [Y/n] ").strip().lower()
            except (KeyboardInterrupt, EOFError):
                console.print("\n  [dim]Aborted.[/dim]\n")
                return
            if answer and answer not in ("y", "yes"):
                console.print("     [dim]Skipped.[/dim]")
                continue

        _apply_fix(c)

    console.print()
    console.print("  Re-check: [cyan]orca doctor[/cyan]\n")


def _apply_fix(check: Check) -> None:
    fix = check.fix

    if fix == "ollama serve":
        _start_ollama_bg()
        return

    if fix.startswith("ollama pull "):
        _pull_model(fix.split()[-1])
        return

    if "orca data seed" in fix:
        console.print(f"     [dim]Run manually:[/dim] [cyan]{fix}[/cyan]")
        return

    # Generic shell execution
    console.print(f"     [dim]Running:[/dim] {fix}")
    try:
        result = subprocess.run(fix, shell=True, text=True)
        if result.returncode == 0:
            console.print("     [green]✓ Done.[/green]")
        else:
            console.print(f"     [red]✗ Failed (exit {result.returncode})[/red]")
    except Exception as e:
        console.print(f"     [red]✗ Error: {e}[/red]")


def _start_ollama_bg() -> None:
    console.print("     [dim]Starting ollama serve...[/dim]")
    try:
        subprocess.Popen(
            ["ollama", "serve"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
        )
        import httpx
        for i in range(15):
            time.sleep(1)
            console.print(f"     [dim]waiting... ({i+1}s)[/dim]", end="\r")
            try:
                r = httpx.get("http://localhost:11434/api/tags", timeout=2.0)
                if r.status_code == 200:
                    console.print("     [green]✓ Ollama is online.          [/green]")
                    return
            except Exception:
                pass
        console.print("     [yellow]⚠ Ollama may still be starting — wait a moment.[/yellow]")
    except FileNotFoundError:
        console.print("     [red]✗ ollama binary not found.[/red]")


def _pull_model(model: str) -> None:
    console.print(f"     [dim]Pulling {model}... (may take several minutes)[/dim]")
    try:
        subprocess.run(["ollama", "pull", model], check=True)
        console.print(f"     [green]✓ {model} ready.[/green]")
    except subprocess.CalledProcessError:
        console.print(f"     [red]✗ Pull failed. Check your internet connection.[/red]")
    except FileNotFoundError:
        console.print("     [red]✗ ollama not installed.[/red]")


# ─────────────────────────────────────────────────────────────────────────────
#  Interactive first-run wizard
# ─────────────────────────────────────────────────────────────────────────────

def run_wizard() -> None:
    console.print()
    console.print(Panel(
        "[bold white]ORCA — FIRST RUN SETUP WIZARD[/bold white]\n\n"
        "  [dim]Your private AI. Zero cloud. Zero APIs.[/dim]\n"
        "  [dim]Let's get your system ready.[/dim]",
        border_style="white",
        box=box.ROUNDED,
        padding=(1, 2),
    ))
    console.print()

    steps: list[tuple[str, object]] = [
        ("System check",          _wiz_system),
        ("Data directories",      _wiz_data_dirs),
        ("Ollama installation",   _wiz_ollama_install),
        ("Ollama server",         _wiz_ollama_running),
        ("Choose a model",        _wiz_pick_model),
        ("Test connection",       _wiz_test_connection),
        ("Quick-start guide",     _wiz_quickstart),
    ]

    total = len(steps)
    for i, (label, fn) in enumerate(steps, 1):
        console.print(f"  [dim]Step {i}/{total}[/dim]  [bold]{label}[/bold]")
        try:
            result = fn()
            if result is False:
                console.print(
                    f"\n  [yellow]Setup incomplete at step {i}.[/yellow]\n"
                    f"  Fix the issue above, then re-run: [cyan]orca doctor --wizard[/cyan]\n"
                )
                return
        except (KeyboardInterrupt, EOFError):
            console.print(f"\n  [dim]Wizard cancelled. Re-run: orca doctor --wizard[/dim]\n")
            return
        console.print()

    _mark_setup_complete()
    console.print(Panel(
        "[bold green]✓ Orca is ready.[/bold green]\n\n"
        "  [cyan bold]orca core chat[/cyan bold]              Start a chat session\n"
        "  [cyan bold]orca serve[/cyan bold]                  Browser UI at localhost:7337\n"
        "  [cyan bold]orca data seed --n 1000[/cyan bold]    Generate training data\n"
        "  [cyan bold]orca status[/cyan bold]                 Full system dashboard\n"
        "  [cyan bold]orca doctor[/cyan bold]                 Health check anytime",
        border_style="green",
        box=box.ROUNDED,
        padding=(1, 2),
    ))
    console.print()


def _wiz_system():
    v = sys.version_info
    ver = f"{v.major}.{v.minor}.{v.micro}"
    ok = v.major == 3 and v.minor >= 11
    icon = "[green]✓[/green]" if ok else "[yellow]⚠[/yellow]"
    console.print(f"    {icon}  Python {ver}")
    console.print(f"    [green]✓[/green]  Platform: {platform.system()} {platform.machine()}")

    ram = _get_ram_gb()
    if ram > 0:
        icon = "[green]✓[/green]" if ram >= 16 else "[yellow]⚠[/yellow]"
        note = "" if ram >= 16 else "  [dim](low RAM — use small models)[/dim]"
        console.print(f"    {icon}  RAM: {ram:.0f} GB{note}")

    _, has_gpu = _detect_gpu()
    if has_gpu:
        gpu_name, _ = _detect_gpu()
        console.print(f"    [green]✓[/green]  GPU: {gpu_name}")
    else:
        console.print("    [yellow]⚠[/yellow]  No GPU detected — CPU inference only")


def _wiz_data_dirs():
    try:
        from orca.config import ORCA_HOME
        _ensure_dirs(ORCA_HOME)
        console.print(f"    [green]✓[/green]  {ORCA_HOME}")
    except Exception as e:
        console.print(f"    [red]✗[/red]  Failed to create data directory: {e}")
        return False


def _wiz_ollama_install():
    path = shutil.which("ollama")
    if path:
        console.print(f"    [green]✓[/green]  Ollama found: {path}")
        return

    console.print("    [red]✗[/red]  Ollama is not installed.")
    console.print()
    console.print("    Install Ollama to continue:\n")
    console.print("      macOS:   [cyan]brew install ollama[/cyan]")
    console.print("      Linux:   [cyan]curl -fsSL https://ollama.ai/install.sh | sh[/cyan]")
    console.print("      Windows: download from [cyan]https://ollama.ai[/cyan]")
    console.print()
    console.print("    After installing, re-run: [cyan]orca doctor --wizard[/cyan]")
    return False


def _wiz_ollama_running():
    if _ollama_reachable():
        console.print("    [green]✓[/green]  Ollama server is online.")
        return

    console.print("    [yellow]⚠[/yellow]  Ollama is installed but not running.")
    try:
        ans = console.input("    Start Ollama now? [Y/n] ").strip().lower()
    except (EOFError, KeyboardInterrupt):
        raise

    if ans in ("", "y", "yes"):
        _start_ollama_bg()
        if not _ollama_reachable():
            console.print("    [red]✗[/red]  Could not start Ollama automatically.")
            console.print("    Start it manually: [cyan]ollama serve[/cyan]")
            return False
    else:
        console.print("    [dim]Start it with: ollama serve[/dim]")
        return False


def _wiz_pick_model():
    try:
        import httpx
        r = httpx.get("http://localhost:11434/api/tags", timeout=3.0)
        existing = [m["name"] for m in r.json().get("models", [])]
    except Exception:
        console.print("    [red]✗[/red]  Cannot connect to Ollama.")
        return False

    if existing:
        console.print(f"    [green]✓[/green]  Models already installed: {', '.join(existing[:4])}")
        if len(existing) > 4:
            console.print(f"         +{len(existing) - 4} more")
        return

    console.print("    No models installed yet.\n")
    console.print("    Recommended models:\n")
    for i, (name, size, desc) in enumerate(RECOMMENDED_MODELS, 1):
        console.print(f"    {i}.  [cyan]{name:<22}[/cyan]  {size:<8}  {desc}")
    console.print(f"    5.  Skip (pull a model manually later)")
    console.print()

    try:
        choice = console.input("    Select [1-5]: ").strip()
        idx = int(choice) - 1
        if idx == 4:
            console.print("    [dim]Skipped. Pull a model later: ollama pull llama3.1:8b[/dim]")
            return
        if 0 <= idx < len(RECOMMENDED_MODELS):
            _pull_model(RECOMMENDED_MODELS[idx][0])
        else:
            console.print("    [dim]Invalid choice.[/dim]")
    except (ValueError, IndexError):
        console.print("    [dim]Invalid input — skipping.[/dim]")


def _wiz_test_connection():
    try:
        from orca.brain.providers import OrcaBrain
        brain = OrcaBrain()
        if brain.is_available():
            console.print(f"    [green]✓[/green]  Connected to Ollama")
            console.print(f"    [green]✓[/green]  Active model: [bold]{brain.name}[/bold]")
        else:
            console.print("    [red]✗[/red]  Brain offline.")
            return False
    except Exception as e:
        console.print(f"    [red]✗[/red]  {e}")
        return False


def _wiz_quickstart():
    console.print("    Your Orca is ready. Here's how to begin:\n")
    cmds = [
        ("orca core chat",           "Start an intelligent chat session with tools + memory"),
        ("orca serve",               "Launch the browser UI  →  http://localhost:7337"),
        ("orca ultra chat",          "Deploy 6 specialist agents on complex tasks"),
        ("orca data seed --n 1000",  "Generate 1000 training examples to fine-tune Orca"),
        ("orca status",              "Full system dashboard at any time"),
        ("orca doctor",              "Health check whenever something feels off"),
    ]
    for cmd, desc in cmds:
        console.print(f"    [cyan]{cmd:<34}[/cyan]  [dim]{desc}[/dim]")


# ─────────────────────────────────────────────────────────────────────────────
#  First-run detection
# ─────────────────────────────────────────────────────────────────────────────

def is_first_run() -> bool:
    return not SETUP_SENTINEL.exists()


def _mark_setup_complete() -> None:
    try:
        SETUP_SENTINEL.parent.mkdir(parents=True, exist_ok=True)
        SETUP_SENTINEL.write_text("ok")
    except Exception:
        pass


def maybe_first_run_hint() -> None:
    """Print a one-liner hint if this looks like a first run. Call from CLI startup."""
    if is_first_run():
        console.print(
            "[dim]First run? Get set up in 2 minutes:[/dim]  "
            "[cyan bold]orca doctor --wizard[/cyan bold]"
        )


# ─────────────────────────────────────────────────────────────────────────────
#  Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _ensure_dirs(home: Path) -> None:
    for sub in [
        "training/raw",
        "training/curated",
        "training/formatted",
        "memory/episodic",
        "models",
        "logs",
    ]:
        (home / sub).mkdir(parents=True, exist_ok=True)


def _progress_bar(value: int, total: int, width: int = 20) -> str:
    filled = min(int((value / max(total, 1)) * width), width)
    return "[" + "█" * filled + "░" * (width - filled) + "]"


def _get_ram_gb() -> float:
    try:
        if platform.system() == "Darwin":
            r = subprocess.run(["sysctl", "-n", "hw.memsize"], capture_output=True, text=True)
            return int(r.stdout.strip()) / (1024 ** 3)
        if platform.system() == "Linux":
            with open("/proc/meminfo") as f:
                for line in f:
                    if line.startswith("MemTotal:"):
                        return int(line.split()[1]) / 1024 / 1024
    except Exception:
        pass
    return 0.0


def _detect_gpu() -> tuple[str, bool]:
    # NVIDIA
    try:
        r = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=5,
        )
        if r.returncode == 0 and r.stdout.strip():
            return r.stdout.strip().splitlines()[0], True
    except Exception:
        pass

    # Apple Silicon (unified memory GPU)
    try:
        r = subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"],
                           capture_output=True, text=True)
        brand = r.stdout.strip()
        if "Apple" in brand:
            return f"{brand} (unified GPU)", True
    except Exception:
        pass

    # AMD ROCm
    try:
        r = subprocess.run(["rocm-smi", "--showproductname"],
                           capture_output=True, text=True, timeout=5)
        if r.returncode == 0 and r.stdout.strip():
            return r.stdout.strip().splitlines()[0], True
    except Exception:
        pass

    return "", False


def _ollama_reachable() -> bool:
    try:
        import httpx
        r = httpx.get("http://localhost:11434/api/tags", timeout=3.0)
        return r.status_code == 200
    except Exception:
        return False
