"""
Orca Train Prepare — preflight validation before spending money on a GPU.

Checks:
  - Training data exists and is formatted
  - Example count (min 200, ideal 1000+)
  - Data quality (curated vs raw ratio)
  - Estimates cost and time per GPU tier
  - Generates a ready-to-paste cloud training command
"""
from __future__ import annotations

import json
from pathlib import Path

from rich.columns import Columns
from rich.console import Console
from rich.panel import Panel
from rich.rule import Rule
from rich.table import Table
from rich.text import Text
from rich import box

from orca.config import ORCA_HOME

console = Console()

FORMATTED_DIR = ORCA_HOME / "training" / "formatted"
CURATED_DIR   = ORCA_HOME / "training" / "curated"
RAW_DIR       = ORCA_HOME / "training" / "raw"

# ─────────────────────────────────────────────────────────────────────────────
#  GPU tiers with real-world estimates
# ─────────────────────────────────────────────────────────────────────────────

GPU_TIERS = [
    {
        "name": "Vast.ai A100 40GB",
        "provider": "vast.ai",
        "vram": 40,
        "price_hr": 1.89,
        "lora_rank": 128,
        "preset": "cloud",
        "min_per_1k_examples": 8,   # minutes per 1000 examples (3 epochs)
        "setup_min": 8,             # install deps + download base model
        "recommended": True,
    },
    {
        "name": "RunPod A100 40GB",
        "provider": "runpod.io",
        "vram": 40,
        "price_hr": 2.49,
        "lora_rank": 128,
        "preset": "cloud",
        "min_per_1k_examples": 8,
        "setup_min": 8,
        "recommended": False,
    },
    {
        "name": "Vast.ai RTX 4090",
        "provider": "vast.ai",
        "vram": 24,
        "price_hr": 0.69,
        "lora_rank": 64,
        "preset": "prosumer",
        "min_per_1k_examples": 20,
        "setup_min": 6,
        "recommended": False,
    },
    {
        "name": "Lambda A100 80GB",
        "provider": "lambdalabs.com",
        "vram": 80,
        "price_hr": 2.49,
        "lora_rank": 256,
        "preset": "cloud_xl",
        "min_per_1k_examples": 6,
        "setup_min": 10,
        "recommended": False,
    },
]


def _count_jsonl(path: Path) -> int:
    if not path.exists():
        return 0
    try:
        return sum(1 for _ in open(path))
    except Exception:
        return 0


def _check_data() -> dict:
    raw_total = sum(
        sum(1 for _ in open(f))
        for f in RAW_DIR.glob("*.jsonl")
        if f.stat().st_size > 0
    ) if RAW_DIR.exists() else 0

    curated = _count_jsonl(CURATED_DIR / "dataset.jsonl")
    train   = _count_jsonl(FORMATTED_DIR / "orca_llama3_train.jsonl")
    eval_   = _count_jsonl(FORMATTED_DIR / "orca_llama3_eval.jsonl")

    return {
        "raw": raw_total,
        "curated": curated,
        "train": train,
        "eval": eval_,
        "formatted": train + eval_,
        "train_file": str(FORMATTED_DIR / "orca_llama3_train.jsonl"),
        "eval_file":  str(FORMATTED_DIR / "orca_llama3_eval.jsonl"),
    }


def _readiness(data: dict) -> tuple[str, list[str], list[str]]:
    """Returns (status, blockers, warnings)."""
    blockers = []
    warnings = []

    if data["raw"] == 0:
        blockers.append("No training data. Run: orca data seed --n 1000")
    elif data["curated"] == 0:
        blockers.append("Data not curated. Run: orca data curate")
    elif data["formatted"] == 0:
        blockers.append("Data not formatted. Run: orca data format --format llama3 --split")
    elif data["train"] < 50:
        blockers.append(f"Only {data['train']} train examples — need at least 50 (200+ recommended)")

    if data["train"] < 200 and data["train"] >= 50:
        warnings.append(f"Only {data['train']} examples — model will learn but may be weak. 500+ is better.")
    if data["eval"] == 0:
        warnings.append("No eval split — add --split flag when formatting")
    if data["raw"] > data["curated"] * 2 and data["curated"] > 0:
        warnings.append(f"{data['raw']} raw examples but only {data['curated']} curated — re-run: orca data curate")

    if blockers:
        return "NOT READY", blockers, warnings
    if data["train"] >= 500:
        return "READY", [], warnings
    return "READY (minimal)", [], warnings


def estimate_cost(n_examples: int, gpu: dict, epochs: int = 3) -> dict:
    train_min = (n_examples / 1000) * gpu["min_per_1k_examples"] * epochs
    total_min = train_min + gpu["setup_min"]
    cost = (total_min / 60) * gpu["price_hr"]
    return {
        "train_min": round(train_min),
        "setup_min": gpu["setup_min"],
        "total_min": round(total_min),
        "cost_usd":  round(cost, 2),
    }


def run_preflight(epochs: int = 3) -> dict:
    """Run full preflight check and print the report."""
    data = _check_data()
    status, blockers, warnings = _readiness(data)
    n = data["train"]

    # ── Header ────────────────────────────────────────────────────────────────
    console.print()
    status_color = {"READY": "green", "READY (minimal)": "yellow", "NOT READY": "red"}[status]
    console.print(Panel(
        f"[{status_color} bold]{status}[/{status_color} bold]",
        title="[bold]◈ Orca Train — Preflight Check[/bold]",
        border_style=status_color,
        expand=False,
    ))
    console.print()

    # ── Data status ───────────────────────────────────────────────────────────
    dt = Table.grid(padding=(0, 3))
    dt.add_column(style="dim", min_width=16)
    dt.add_column(style="bold white")

    bar_len = 20
    filled  = min(int((n / 1000) * bar_len), bar_len)
    color   = "green" if n >= 500 else "yellow" if n >= 200 else "red"
    bar     = f"[{color}]" + "█" * filled + f"[/{color}][dim]" + "░" * (bar_len - filled) + "[/dim]"

    dt.add_row("raw collected",   str(data["raw"]))
    dt.add_row("after curation",  str(data["curated"]))
    dt.add_row("train examples",  f"{n}")
    dt.add_row("eval examples",   str(data["eval"]))
    dt.add_row("progress",        Text.from_markup(f"{bar} [dim]{n}/1000[/dim]"))
    console.print(Panel(dt, title="[bold]Training Data[/bold]", border_style="blue", box=box.ROUNDED))

    # Blockers / warnings
    if blockers:
        for b in blockers:
            console.print(f"  [red bold]✗[/red bold] {b}")
        console.print()
        return {"status": status, "data": data, "ready": False}

    if warnings:
        for w in warnings:
            console.print(f"  [yellow]⚠[/yellow]  {w}")
        console.print()

    # ── GPU cost table ─────────────────────────────────────────────────────────
    console.print(Rule("[dim]GPU Options[/dim]"))
    console.print()

    gt = Table(box=box.SIMPLE_HEAD, header_style="bold dim", show_header=True)
    gt.add_column("GPU", style="bold")
    gt.add_column("Provider", style="dim")
    gt.add_column("VRAM", justify="right")
    gt.add_column("$/hr", justify="right", style="dim")
    gt.add_column("Train time", justify="right")
    gt.add_column("Est. cost", justify="right", style="green")
    gt.add_column("LoRA rank", justify="right", style="dim")
    gt.add_column("Preset")

    for gpu in GPU_TIERS:
        est = estimate_cost(n, gpu, epochs)
        star = " [green]★[/green]" if gpu["recommended"] else ""
        gt.add_row(
            gpu["name"] + star,
            gpu["provider"],
            f"{gpu['vram']}GB",
            f"${gpu['price_hr']:.2f}",
            f"~{est['total_min']}min",
            f"~${est['cost_usd']:.2f}",
            str(gpu["lora_rank"]),
            gpu["preset"],
        )

    console.print(gt)
    console.print(f"  [dim]Estimates: {n} train examples × {epochs} epochs (includes setup)[/dim]")
    console.print()

    # ── Recommended flow ──────────────────────────────────────────────────────
    recommended = next(g for g in GPU_TIERS if g["recommended"])
    est = estimate_cost(n, recommended, epochs)

    console.print(Rule("[dim]Recommended Path[/dim]"))
    console.print()
    console.print(f"  [dim]1.[/dim] Rent a pod on [bold]vast.ai[/bold] → search for [bold]A100 40GB[/bold]")
    console.print(f"     Select template: [bold]PyTorch 2.x / CUDA 12[/bold]")
    console.print(f"     Estimated spend: [green bold]~${est['cost_usd']:.2f}[/green bold] total")
    console.print()
    console.print(f"  [dim]2.[/dim] Once your instance is up, copy the SSH command and run:")
    console.print()
    console.print(f'     [bold cyan]orca train cloud --ssh "ssh root@<IP> -p <PORT> -i ~/.ssh/id_rsa" --preset cloud[/bold cyan]')
    console.print()
    console.print(f"  [dim]3.[/dim] Orca will upload data, train, download model, and register it as [bold]orca[/bold] in Ollama.")
    console.print(f"     Total time: [bold]~{est['total_min']} minutes[/bold]")
    console.print()

    return {
        "status": status,
        "data": data,
        "ready": True,
        "recommended_gpu": recommended,
        "estimated_cost": est,
    }
