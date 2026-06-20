"""
Orca CLI — god-mode command interface.

  orca nano "what is entropy?"
  orca core chat
  orca core chat --thoughts          # show reasoning trace
  orca core think "design a cache"
  orca ultra run "build a REST API"
  cat data.csv | orca nano "analyze"
  orca backend
  orca data seed --n 500
  orca train run --preset prosumer
  orca train export ~/.orca/models/orca-8b-qlora/merged
"""
from __future__ import annotations

import sys
import typer
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel

from orca.character import banner, TAGLINE, NAME

app = typer.Typer(
    name="orca",
    help=f"[bold cyan]{NAME}[/bold cyan] — {TAGLINE}",
    no_args_is_help=True,
    rich_markup_mode="rich",
)
console = Console()


def _version_callback(value: bool):
    if value:
        from orca.__version__ import __version__
        typer.echo(f"orca {__version__}")
        raise typer.Exit()


@app.callback()
def _app_callback(
    ctx: typer.Context,
    version: bool = typer.Option(
        None, "--version", "-V",
        callback=_version_callback, is_eager=True, is_flag=True,
        help="Show version and exit.",
    ),
):
    """Orca — apex private intelligence. Your hardware. Your data."""
    if ctx.invoked_subcommand is None:
        from orca.doctor import maybe_first_run_hint
        maybe_first_run_hint()

nano_app  = typer.Typer(help="[cyan]Nano[/cyan]  — fast, precise, terminal-native")
core_app  = typer.Typer(help="[blue]Core[/blue]  — full intelligence, tools, memory")
ultra_app = typer.Typer(help="[magenta]Ultra[/magenta] — multi-agent apex orchestration")
data_app  = typer.Typer(help="[yellow]Data[/yellow]  — collect, curate, format training data")
train_app = typer.Typer(help="[red]Train[/red]  — fine-tune, evaluate, and export Orca model")

app.add_typer(nano_app,  name="nano")
app.add_typer(core_app,  name="core")
app.add_typer(ultra_app, name="ultra")
app.add_typer(data_app,  name="data")
app.add_typer(train_app, name="train")


def _piped() -> str | None:
    if not sys.stdin.isatty():
        return sys.stdin.read().strip()
    return None


def _check_ollama():
    from orca.brain.providers import OrcaBrain
    if not OrcaBrain().is_available():
        console.print("[red]Ollama is not running.[/red]")
        console.print("  Start: [bold]ollama serve[/bold]")
        console.print("  Install: [bold]curl -fsSL https://ollama.ai/install.sh | sh[/bold]")
        raise typer.Exit(1)


# ── Nano ─────────────────────────────────────────────────────────────────────

@nano_app.callback(invoke_without_command=True)
def nano_default(
    ctx: typer.Context,
    prompt: str = typer.Argument(None),
    model: str = typer.Option(None, "--model", "-m"),
    stream: bool = typer.Option(True, "--stream/--no-stream"),
):
    """Fast single-shot response."""
    if ctx.invoked_subcommand:
        return
    piped = _piped()
    if not prompt and not piped:
        typer.echo(ctx.get_help())
        return
    _check_ollama()
    from orca.variants.nano import OrcaNano
    bot = OrcaNano(model=model)
    if stream:
        for chunk in bot.stream(prompt or "", piped):
            print(chunk, end="", flush=True)
        print()
    else:
        print(bot.run(prompt or "", piped))


@nano_app.command("chat")
def nano_chat(model: str = typer.Option(None, "--model", "-m")):
    """Interactive Nano session."""
    _check_ollama()
    from orca.variants.nano import OrcaNano
    bot = OrcaNano(model=model)
    banner("nano", bot.brain.name)
    while True:
        try:
            prompt = console.input("\n[cyan]you ▸[/cyan] ").strip()
        except (KeyboardInterrupt, EOFError):
            console.print("\n[dim]Goodbye.[/dim]")
            break
        if not prompt or prompt.lower() in ("exit", "quit"):
            console.print("[dim]Goodbye.[/dim]")
            break
        print("[cyan]orca ▸[/cyan] ", end="")
        for chunk in bot.chat(prompt):
            print(chunk, end="", flush=True)
        print()


# ── Core ─────────────────────────────────────────────────────────────────────

@core_app.callback(invoke_without_command=True)
def core_default(ctx: typer.Context):
    if not ctx.invoked_subcommand:
        typer.echo(ctx.get_help())


@core_app.command("chat")
def core_chat(
    session: str = typer.Option(None, "--session", "-s", help="Resume a session by ID"),
    model: str = typer.Option(None, "--model", "-m"),
    thoughts: bool = typer.Option(False, "--thoughts", help="Show reasoning trace"),
    no_reflect: bool = typer.Option(False, "--no-reflect", help="Disable self-reflection"),
):
    """God-mode interactive chat with tools, memory, and self-reflection."""
    _check_ollama()
    from orca.variants.core import OrcaCore
    bot = OrcaCore(
        load_session=session,
        model=model,
        reflect=not no_reflect,
        show_thoughts=thoughts,
    )
    bot.chat()


@core_app.command("think")
def core_think(
    prompt: str = typer.Argument(...),
    model: str = typer.Option(None, "--model", "-m"),
    thoughts: bool = typer.Option(False, "--thoughts"),
    raw: bool = typer.Option(False, "--raw"),
):
    """Single-shot deep reasoning with tool access."""
    _check_ollama()
    from orca.variants.core import OrcaCore
    bot = OrcaCore(model=model, show_thoughts=thoughts)
    piped = _piped()
    with console.status("[dim]thinking...[/dim]", spinner="dots"):
        result = bot.think(prompt, piped)
    console.print(Markdown(result) if not raw else result)


@core_app.command("stream")
def core_stream(
    prompt: str = typer.Argument(...),
    model: str = typer.Option(None, "--model", "-m"),
):
    """Stream a Core response."""
    _check_ollama()
    from orca.variants.core import OrcaCore
    bot = OrcaCore(model=model)
    for chunk in bot.stream(prompt, _piped()):
        print(chunk, end="", flush=True)
    print()


# ── Ultra ─────────────────────────────────────────────────────────────────────

@ultra_app.callback(invoke_without_command=True)
def ultra_default(ctx: typer.Context):
    if not ctx.invoked_subcommand:
        typer.echo(ctx.get_help())


@ultra_app.command("run")
def ultra_run(
    task: str = typer.Argument(...),
    retries: int = typer.Option(2, "--retries", "-r"),
    model: str = typer.Option(None, "--model", "-m"),
    json_out: bool = typer.Option(False, "--json"),
):
    """Deploy the agent pod on a complex task."""
    from orca.license import gate
    gate("ultra")
    _check_ollama()
    import dataclasses, json
    from orca.variants.ultra import OrcaUltra

    def progress(msg: str):
        console.print(f"[dim]{msg}[/dim]")

    bot = OrcaUltra(on_progress=progress, model=model)

    console.print(Panel(
        f"[bold]Task:[/bold] {task}",
        title="[magenta]▓ ORCA ULTRA ▓[/magenta]",
        border_style="magenta",
    ))

    pipeline = bot.run(task, max_retries=retries)

    if json_out:
        print(json.dumps({
            "goal": pipeline.goal,
            "run_id": pipeline.run_id,
            "subtasks": [dataclasses.asdict(t) for t in pipeline.subtasks],
            "output": pipeline.final_output,
            "grade": pipeline.grade,
            "iterations": pipeline.iterations,
        }, indent=2))
    else:
        console.print("\n")
        console.print(Markdown(pipeline.final_output))
        score = pipeline.grade.get("score", "?")
        console.print(f"\n[dim]Score: {score}/100 · Agents: {len(pipeline.subtasks)} · Iterations: {pipeline.iterations}[/dim]")


@ultra_app.command("chat")
def ultra_chat(model: str = typer.Option(None, "--model", "-m")):
    """Interactive Ultra session — give tasks, watch the pod work."""
    from orca.license import gate
    gate("ultra")
    _check_ollama()
    from orca.variants.ultra import OrcaUltra

    def progress(msg: str):
        console.print(f"[dim]{msg}[/dim]")

    bot = OrcaUltra(on_progress=progress, model=model)
    bot.chat()


# ── Top-level commands ────────────────────────────────────────────────────────

@app.command("backend")
def backend_cmd():
    """Show which local model Orca is running on."""
    from orca.brain.providers import OrcaBrain
    b = OrcaBrain()
    try:
        models = b.list_models()
        active = b.name
        orca_models = [m for m in models if "orca" in m.lower()]
        console.print(Panel(
            f"[bold]Active model:[/bold]    [green]{active}[/green]\n"
            f"[bold]Your Orca models:[/bold] {', '.join(orca_models) or 'none — run: orca train export ...'}\n"
            f"[bold]All local models:[/bold] {', '.join(models[:10])}",
            title="[bold]Orca — Local Brain[/bold]",
            border_style="cyan",
        ))
    except RuntimeError as e:
        console.print(f"[red]{e}[/red]")


@app.command("sessions")
def sessions_cmd():
    """List saved sessions."""
    from orca.brain.memory import EpisodicMemory
    sids = EpisodicMemory.list_sessions()
    if not sids:
        console.print("[dim]No saved sessions.[/dim]")
        return
    console.print(f"[bold]{len(sids)} session(s):[/bold]")
    for sid in sids:
        console.print(f"  [cyan]{sid[:8]}[/cyan]  {sid}")


@app.command("search")
def search_cmd(query: str = typer.Argument(...)):
    """Search the web from the terminal."""
    from orca.tools.web import search_and_fetch
    with console.status("[dim]searching...[/dim]"):
        result = search_and_fetch(query)
    console.print(result)


@app.command("run")
def run_cmd(
    code: str = typer.Argument(None),
    file: str = typer.Option(None, "--file", "-f"),
    lang: str = typer.Option("python", "--lang", "-l"),
):
    """Execute code locally."""
    from orca.tools.code import run_code
    src = open(file).read() if file else (code or _piped() or "")
    if not src:
        console.print("[red]No code provided.[/red]")
        raise typer.Exit(1)
    result = run_code(src, lang)
    print(result.format())


@app.command("status")
def status_cmd():
    """Live dashboard — model, memory, data, tools at a glance."""
    from orca.tui import status_dashboard
    status_dashboard()


@app.command("history")
def history_cmd(n: int = typer.Option(20, "--n", help="Number of sessions to show")):
    """Browse saved session history."""
    from orca.tui import history_browser
    history_browser(n=n)


@app.command("info")
def info_cmd():
    """Show Orca configuration."""
    from orca.config import CONFIG
    console.print(Panel(
        f"[bold]Ollama host:[/bold]   {CONFIG.ollama.host}\n"
        f"[bold]Core model:[/bold]    {CONFIG.ollama.model_core}\n"
        f"[bold]Nano model:[/bold]    {CONFIG.ollama.model_nano}\n"
        f"[bold]Ultra model:[/bold]   {CONFIG.ollama.model_ultra}\n\n"
        f"[bold]Memory:[/bold]        ~/.orca/memory/\n"
        f"[bold]Training data:[/bold] ~/.orca/training/\n"
        f"[bold]Your models:[/bold]   ~/.orca/models/\n\n"
        f"[bold]Tagline:[/bold]       {TAGLINE}",
        title="[bold cyan]Orca[/bold cyan]",
        border_style="cyan",
    ))


# ── Data pipeline ─────────────────────────────────────────────────────────────

@data_app.command("stats")
def data_stats():
    """Show training data collection stats."""
    from orca.data.collector import DataCollector
    from orca.data.curator import DataCurator
    counts = DataCollector.count_examples()
    curated = DataCurator().stats()
    total_raw = sum(counts.values())
    console.print(Panel(
        f"[bold]Raw examples:[/bold]     {total_raw}\n"
        + "\n".join(f"  {k}: {v}" for k, v in counts.items())
        + f"\n[bold]Curated examples:[/bold] {curated.get('examples', 0)}\n"
        + (f"  Path: {curated.get('path', '')}" if curated.get('path') else ""),
        title="[yellow]Training Data[/yellow]",
        border_style="yellow",
    ))


@data_app.command("seed")
def data_seed(
    n: int = typer.Option(500, "--n", help="Total examples to generate"),
    workers: int = typer.Option(2, "--workers", "-w", help="Workers (note: Ollama serializes, >2 rarely helps)"),
    domains: str = typer.Option(None, "--domains", "-d", help="Comma-separated domains (default: all)"),
    model: str = typer.Option(None, "--model", "-m"),
    temperature: float = typer.Option(0.85, "--temp", "-t"),
    max_tokens: int = typer.Option(800, "--max-tokens", help="Max tokens per response (lower = faster)"),
    fast: bool = typer.Option(False, "--fast", help="Fast mode: 400 token cap, shorter answers"),
    preview: bool = typer.Option(False, "--preview", help="Preview one example per domain then exit"),
):
    """Generate synthetic training data in parallel using your local model."""
    from orca.brain.providers import get_brain
    from orca.data.pipeline import SeedPipeline, preview_domain
    from orca.data.seeds import ALL_DOMAINS, DOMAIN_MAP

    try:
        brain = get_brain(model)
    except RuntimeError as e:
        from orca.tui import ollama_offline_panel
        ollama_offline_panel()
        raise typer.Exit(1)

    domain_list = [d.strip() for d in domains.split(",")] if domains else None

    if preview:
        # Preview mode: generate one example per requested domain
        targets = domain_list or [d.name for d in ALL_DOMAINS[:5]]
        for dname in targets:
            console.print(f"\n[cyan]── {dname} ──[/cyan]")
            result = preview_domain(dname, brain)
            console.print(result[:500])
        return

    # Validate domain names
    if domain_list:
        for dname in domain_list:
            if dname not in DOMAIN_MAP:
                console.print(f"[red]Unknown domain: {dname}[/red]")
                console.print(f"Available: {', '.join(DOMAIN_MAP.keys())}")
                raise typer.Exit(1)

    console.print(f"\n[bold cyan]◈ Orca Parallel Seed Pipeline[/bold cyan]")
    console.print(f"  [dim]model:[/dim]   [bold]{brain.name}[/bold]")
    console.print(f"  [dim]target:[/dim]  [bold]{n}[/bold] examples")
    console.print(f"  [dim]workers:[/dim] [bold]{workers}[/bold] parallel")
    domains_label = ", ".join(domain_list) if domain_list else f"all {len(ALL_DOMAINS)} domains"
    console.print(f"  [dim]domains:[/dim] [bold]{domains_label}[/bold]")
    console.print(f"  [dim]100% local — no data leaves your machine[/dim]\n")

    effective_tokens = 400 if fast else max_tokens
    if fast:
        console.print(f"  [dim]fast mode:[/dim] [bold]{effective_tokens} token cap[/bold]\n")

    pipeline = SeedPipeline(
        brain=brain,
        n=n,
        workers=workers,
        domains=domain_list,
        temperature=temperature,
        max_tokens=effective_tokens,
    )

    result = pipeline.run()

    console.print(f"\n[bold green]✓ Done![/bold green]")
    console.print(f"  Generated: [bold green]{result.total_generated}[/bold green]")
    console.print(f"  Skipped:   [dim]{result.total_skipped}[/dim]")
    console.print(f"  Time:      {result.duration_sec:.0f}s")
    console.print(f"  Rate:      {result.total_generated / max(result.duration_sec, 1) * 60:.0f} examples/min\n")

    if result.by_domain:
        console.print("[dim]By domain:[/dim]")
        for domain, count in sorted(result.by_domain.items(), key=lambda x: -x[1]):
            console.print(f"  [cyan]{domain:<16}[/cyan] {count}")

    console.print(f"\nNext: [bold]orca data curate[/bold]")


@data_app.command("domains")
def data_domains():
    """List all available seed domains."""
    from orca.data.seeds import ALL_DOMAINS, TOTAL_WEIGHT
    from rich.table import Table
    from rich import box

    t = Table(box=box.SIMPLE_HEAD, header_style="bold dim")
    t.add_column("Domain", style="cyan")
    t.add_column("Weight", justify="right", style="dim")
    t.add_column("% of total", justify="right")
    t.add_column("Multi-turn", justify="center")
    t.add_column("Subtopics", justify="right", style="dim")

    for d in ALL_DOMAINS:
        pct = f"{d.weight / TOTAL_WEIGHT * 100:.0f}%"
        t.add_row(
            d.name,
            str(d.weight),
            pct,
            "✓" if d.multi_turn else "—",
            str(len(d.subtopics)),
        )

    console.print()
    console.print(t)
    console.print(f"\n[dim]Use:[/dim] orca data seed --domains python,sql,debugging --n 300")


@data_app.command("inspect")
def data_inspect(n: int = typer.Option(3, "--n", help="Number of examples to sample")):
    """Sample and display curated training examples."""
    from orca.data.curator import DataCurator
    import json as _json

    samples = DataCurator().inspect(n)
    if not samples:
        console.print("[dim]No curated data yet. Run: orca data curate[/dim]")
        return

    for i, conv in enumerate(samples):
        console.print(f"\n[bold cyan]── Example {i+1} ({conv.get('variant', '?')}) ──[/bold cyan]")
        for turn in conv.get("conversations", []):
            role = turn.get("role", "?")
            val = turn.get("value", turn.get("content", ""))[:300]
            if role == "system":
                continue
            color = "green" if role == "human" else "blue"
            console.print(f"  [{color}]{role}:[/{color}] {val}")
    console.print()


@data_app.command("curate")
def data_curate(
    local_judge: bool = typer.Option(False, "--local-judge", help="Use local Ollama model as quality judge"),
    min_score: int = typer.Option(7, "--min-score", help="Minimum quality score to keep (1-10)"),
    workers: int = typer.Option(4, "--workers", "-w", help="Parallel processing workers"),
    model: str = typer.Option(None, "--model", "-m", help="Model for local judge"),
):
    """Clean, deduplicate, and score raw training data."""
    from orca.data.curator import DataCurator

    brain = None
    if local_judge:
        from orca.brain.providers import get_brain
        try:
            brain = get_brain(model)
            console.print(f"[dim]Local judge: {brain.name}[/dim]")
        except RuntimeError:
            console.print("[yellow]Ollama not running — skipping AI judge[/yellow]")
            local_judge = False

    with console.status("[yellow]Curating...[/yellow]"):
        curator = DataCurator(
            use_local_judge=local_judge,
            min_score=min_score,
            workers=workers if not local_judge else 1,
            brain=brain,
        )
        result = curator.curate()

    console.print(f"[green]Kept:[/green] {result['kept']}  [red]Rejected:[/red] {result['rejected']}  [dim]({result.get('reject_rate', '?')} reject rate)[/dim]")
    console.print(f"Output: [dim]{result.get('output', '')}[/dim]")
    console.print("\nNext: [bold]orca data format[/bold]  [dim]or[/dim]  [bold]orca data inspect[/bold]")


@data_app.command("format")
def data_format(
    fmt: str = typer.Option("llama3", "--format", "-f", help="llama3 | chatml | alpaca"),
    split: bool = typer.Option(True, "--split/--no-split"),
):
    """Convert curated data to training format."""
    from orca.data.formatter import DataFormatter
    from pathlib import Path
    f = DataFormatter(fmt=fmt)  # type: ignore
    with console.status(f"[yellow]Formatting ({fmt})...[/yellow]"):
        result = f.format()
    console.print(f"[green]{result['converted']} examples[/green] → {result['output']}")
    if split:
        paths = f.split(Path(result["output"]))
        console.print(f"Train: {paths['train_examples']} | Eval: {paths['eval_examples']}")
    console.print("Next: [bold]orca train run --preset prosumer[/bold]")


# ── Train pipeline ─────────────────────────────────────────────────────────────

@train_app.command("prepare")
def train_prepare(
    epochs: int = typer.Option(3, "--epochs", "-e", help="Training epochs (for cost estimate)"),
):
    """Preflight check — validates data and estimates GPU cost before training."""
    from orca.train.prepare import run_preflight
    result = run_preflight(epochs=epochs)
    if not result["ready"]:
        raise typer.Exit(1)


@train_app.command("cloud")
def train_cloud(
    ssh: str = typer.Option(..., "--ssh", "-s", help='SSH command e.g. "ssh root@1.2.3.4 -p 22 -i ~/.ssh/id_rsa"'),
    preset: str = typer.Option("cloud", "--preset", "-p", help="cloud|cloud_xl|prosumer"),
    epochs: int = typer.Option(3, "--epochs", "-e"),
    name: str = typer.Option("orca", "--name", "-n", help="Ollama model name after training"),
):
    """Train on a rented GPU via SSH — uploads data, trains, downloads model."""
    from orca.license import gate
    gate("cloud_train")
    from orca.train.cloud import CloudTrainer
    from orca.train.prepare import run_preflight

    console.print()
    result = run_preflight(epochs=epochs)
    if not result.get("ready"):
        console.print("\n[red]Fix the issues above before training.[/red]")
        raise typer.Exit(1)

    console.print()
    trainer = CloudTrainer(ssh=ssh, preset=preset, epochs=epochs, model_name=name)
    try:
        trainer.run()
    except RuntimeError as e:
        console.print(f"\n[red bold]✗[/red bold] {e}")
        raise typer.Exit(1)
    except FileNotFoundError as e:
        console.print(f"\n[red bold]✗[/red bold] {e}")
        raise typer.Exit(1)


@train_app.command("run")
def train_run(
    preset: str = typer.Option("prosumer", "--preset", "-p", help="laptop|prosumer|cloud|cloud_xl"),
    epochs: int = typer.Option(None, "--epochs", "-e"),
    rank: int = typer.Option(None, "--rank", "-r"),
    model: str = typer.Option(None, "--model", "-m"),
):
    """Fine-tune Orca via QLoRA. Requires GPU + training deps."""
    from orca.train.config import TrainingConfig
    from orca.train.finetune import train

    cfg = TrainingConfig.preset(preset)
    if epochs:
        cfg.num_epochs = epochs
    if rank:
        cfg.lora.r = rank
        cfg.lora.lora_alpha = rank * 2
    if model:
        cfg.base_model = model

    console.print(Panel(
        f"[bold]Preset:[/bold]     {preset}\n"
        f"[bold]Base model:[/bold] {cfg.base_model}\n"
        f"[bold]LoRA rank:[/bold]  {cfg.lora.r}\n"
        f"[bold]Epochs:[/bold]     {cfg.num_epochs}\n"
        f"[bold]Output:[/bold]     {cfg.output_dir}",
        title="[bold red]Orca QLoRA Fine-Tune[/bold red]",
        border_style="red",
    ))

    try:
        meta = train(cfg, on_log=lambda m: console.print(f"[dim]{m}[/dim]"))
        console.print(f"\n[green]Training complete![/green]")
        console.print(f"Loss: {meta['train_loss']:.4f} | Time: {meta['duration_min']:.1f} min")
        console.print(f"\nNext: [bold]orca train export {meta['merged_path']}[/bold]")
    except FileNotFoundError as e:
        console.print(f"[red]{e}[/red]")
        raise typer.Exit(1)


@train_app.command("eval")
def train_eval(
    model: str = typer.Argument(..., help="Path to merged model"),
):
    """Evaluate a fine-tuned model."""
    from orca.train.eval import ModelEvaluator
    ev = ModelEvaluator(model, on_log=lambda m: console.print(f"[dim]{m}[/dim]"))
    report = ev.full_report()
    console.print(Panel(
        f"[bold]Score:[/bold]    [green]{report['overall_score']}/100[/green]\n"
        f"[bold]Accuracy:[/bold] {report['accuracy']['accuracy']*100:.0f}%\n"
        f"[bold]Style:[/bold]    {report['style']['style_score']}/10\n"
        f"[bold]Speed:[/bold]    {report['speed']['tokens_per_sec']:.1f} tok/s",
        title="[bold]Eval Report[/bold]",
        border_style="magenta",
    ))


@train_app.command("export")
def train_export(
    model: str = typer.Argument(..., help="Path to merged HF model"),
    name: str = typer.Option("orca", "--name", "-n"),
    quant: str = typer.Option("q4_k_m", "--quant", "-q"),
):
    """Export to GGUF and register with Ollama as 'orca'."""
    from orca.serve.export import ModelExporter
    exporter = ModelExporter(model, quantization=quant,
                              on_log=lambda m: console.print(f"[dim]{m}[/dim]"))
    result = exporter.export(model_name=name)
    console.print(Panel(
        f"[bold]Model:[/bold]  {result['model_name']}\n"
        f"[bold]GGUF:[/bold]   {result['gguf_path']}\n"
        f"[bold]Quant:[/bold]  {result['quantization']}\n\n"
        f"[bold]Test:[/bold]   ollama run {name}\n"
        f"[bold]Use:[/bold]    ORCA_CORE_MODEL={name} orca core chat",
        title="[green]Export Complete[/green]",
        border_style="green",
    ))


@app.command("activate")
def activate_cmd(
    key: str = typer.Argument(..., help="Your Orca license key  (ORCA-PRO-...)"),
    email: str = typer.Option("", "--email", "-e", help="Email address for records"),
):
    """Activate an Orca license key."""
    from orca.license import validate_key, save_license

    with console.status("[dim]Validating key...[/dim]"):
        lk = validate_key(key)

    if not lk.valid:
        console.print(Panel(
            f"[bold red]Activation Failed[/bold red]\n\n"
            f"  [dim]{lk.error}[/dim]\n\n"
            f"  Check the key and try again, or contact support.",
            border_style="red",
        ))
        raise typer.Exit(1)

    save_license(lk, email=email)

    from orca.license.keys import format_expiry
    expiry_str = format_expiry(lk)
    seats_str = "unlimited" if lk.tier == "enterprise" and lk.seats >= 255 else str(lk.seats)

    console.print(Panel(
        f"[bold green]✓ License Activated[/bold green]\n\n"
        f"  Tier:   [bold cyan]{lk.tier.upper()}[/bold cyan]\n"
        f"  Seats:  {seats_str}\n"
        f"  Valid:  {expiry_str}\n\n"
        f"  [dim]Run[/dim] [cyan]orca license[/cyan] [dim]to see full status.[/dim]",
        border_style="green",
    ))


@app.command("license")
def license_cmd(
    generate: bool   = typer.Option(False, "--generate", "-g", help="[Admin] Generate a new key"),
    tier:     str    = typer.Option("pro",   "--tier",   "-t", help="Tier: free | pro | enterprise"),
    seats:    int    = typer.Option(1,       "--seats",  "-s", help="Number of seats"),
    days:     int    = typer.Option(365,     "--days",   "-d", help="Validity in days (0=lifetime)"),
    deactivate: bool = typer.Option(False, "--deactivate", help="Remove stored license"),
    logs:     bool   = typer.Option(False, "--logs",    "-l", help="[Admin] Show issued key log"),
    buy:      bool   = typer.Option(False, "--buy",          help="Show pricing / purchase link"),
):
    """Show license status, generate admin keys, or manage activation."""
    from orca.license import get_active_license, current_tier, clear_license
    from orca.license.keys import format_expiry, TIER_FEATURES
    from orca.license.store import activation_email, load_record

    # ── Buy / pricing ────────────────────────────────────────────────────────
    if buy:
        console.print(Panel(
            "[bold white]ORCA PRICING[/bold white]\n\n"
            "  [bold cyan]Pro[/bold cyan]          $49 / month  ·  $399 / year\n"
            "  [dim]Ultra multi-agent, cloud training, web UI[/dim]\n\n"
            "  [bold cyan]Enterprise[/bold cyan]   $199 / month  ·  $1,499 / year\n"
            "  [dim]5 seats, all features, priority support[/dim]\n\n"
            "  Purchase: [cyan]https://orca.systems/pricing[/cyan]\n"
            "  Contact:  [cyan]team@orca.systems[/cyan]",
            border_style="dim",
        ))
        return

    # ── Deactivate ──────────────────────────────────────────────────────────
    if deactivate:
        record = load_record()
        if not record:
            console.print("[dim]No license stored.[/dim]")
            return
        try:
            ans = console.input("Remove stored license? [y/N] ").strip().lower()
        except (KeyboardInterrupt, EOFError):
            return
        if ans in ("y", "yes"):
            clear_license()
            console.print("[dim]License removed.[/dim]")
        return

    # ── Admin: generate key ─────────────────────────────────────────────────
    if generate:
        from orca.license.keys import generate_key
        key = generate_key(tier=tier, seats=seats, days=days)
        expiry_note = f"{days} days" if days > 0 else "lifetime"
        console.print(Panel(
            f"[bold white]GENERATED KEY[/bold white]\n\n"
            f"  [bold cyan]{key}[/bold cyan]\n\n"
            f"  Tier:   {tier.upper()}\n"
            f"  Seats:  {seats}\n"
            f"  Valid:  {expiry_note}\n\n"
            f"  Activate: [dim]orca activate {key}[/dim]",
            border_style="cyan",
        ))
        return

    # ── Admin: issued key log ───────────────────────────────────────────────
    if logs:
        from orca.license.stripe_hook import list_issued_keys
        from rich.table import Table
        from rich import box as _box
        records = list_issued_keys(50)
        if not records:
            console.print("[dim]No license keys issued yet.[/dim]")
            return
        t = Table(box=_box.SIMPLE_HEAD, header_style="bold dim")
        t.add_column("Date",  style="dim",  width=12)
        t.add_column("Tier",  style="cyan", width=12)
        t.add_column("Seats", width=6)
        t.add_column("Days",  width=6)
        t.add_column("Email", style="dim")
        t.add_column("Key (first 24)",  style="dim")
        for r in records:
            ts = r.get("ts", "")[:10]
            t.add_row(ts, r["tier"], str(r["seats"]), str(r["days"]),
                      r.get("email", ""), r["key"][:24] + "...")
        console.print()
        console.print(t)
        console.print(f"\n[dim]{len(records)} keys issued[/dim]\n")
        return

    # ── License status ──────────────────────────────────────────────────────
    lk = get_active_license()
    record = load_record()

    if not lk:
        if record:
            # Stored but invalid / expired
            console.print(Panel(
                "[bold red]License Invalid or Expired[/bold red]\n\n"
                f"  Stored key: [dim]{record.get('key', '?')[:30]}...[/dim]\n\n"
                "  Re-activate: [cyan]orca activate <new-key>[/cyan]\n"
                "  Purchase:    [cyan]orca license --buy[/cyan]",
                border_style="red",
            ))
        else:
            console.print(Panel(
                "[bold white]No License Activated[/bold white]\n\n"
                "  Running on [bold]FREE[/bold] tier.\n\n"
                "  Activate:  [cyan bold]orca activate <your-key>[/cyan bold]\n"
                "  Purchase:  [cyan]orca license --buy[/cyan]\n"
                "  Generate:  [cyan]orca license --generate[/cyan]  [dim](admin)[/dim]",
                border_style="dim",
            ))
        return

    expiry_str   = format_expiry(lk)
    email_str    = activation_email()
    activated_at = record.get("activated_at", "")[:10] if record else ""
    allowed      = TIER_FEATURES.get(lk.tier, set())
    feature_list = "all features" if "*" in allowed else "  " + "\n  ".join(sorted(allowed))

    console.print(Panel(
        f"[bold green]License Active[/bold green]\n\n"
        f"  Tier:         [bold cyan]{lk.tier.upper()}[/bold cyan]\n"
        f"  Seats:        {lk.seats}\n"
        f"  Valid:        {expiry_str}\n"
        f"  Activated:    {activated_at}\n"
        + (f"  Email:        [dim]{email_str}[/dim]\n" if email_str else "")
        + f"\n  Features:\n  {feature_list}",
        border_style="green",
    ))


@app.command("doctor")
def doctor_cmd(
    fix:    bool = typer.Option(False, "--fix",    "-f", help="Auto-repair failing checks"),
    wizard: bool = typer.Option(False, "--wizard", "-w", help="Interactive first-run setup wizard"),
    yes:    bool = typer.Option(False, "--yes",    "-y", help="Apply all fixes without prompting (use with --fix)"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Show detail for every check"),
):
    """System health check, auto-fixer, and first-run setup wizard."""
    from orca.doctor import OrcaDoctor, print_report, run_fix, run_wizard

    if wizard:
        run_wizard()
        return

    with console.status("[dim]Running checks...[/dim]", spinner="dots"):
        report = OrcaDoctor().run_all()

    print_report(report, verbose=verbose)

    if fix:
        run_fix(report, yes=yes)


@app.command("upgrade")
def upgrade_cmd(
    yes: bool = typer.Option(False, "--yes", "-y", help="Skip confirmation prompt"),
    check: bool = typer.Option(False, "--check", "-c", help="Only check, do not upgrade"),
):
    """Check for a newer Orca version and self-update from PyPI."""
    from orca.upgrade import is_update_available, self_update, _current_version

    with console.status("[dim]Checking PyPI...[/dim]"):
        available, current, latest = is_update_available()

    if not latest:
        console.print(Panel(
            f"[yellow]Could not reach PyPI.[/yellow]\n\n"
            f"  Current version: [bold]{current}[/bold]\n"
            f"  Check your internet connection.",
            border_style="yellow",
        ))
        raise typer.Exit(1)

    if not available:
        console.print(Panel(
            f"[green]Orca is up to date.[/green]\n\n"
            f"  Version: [bold]{current}[/bold]",
            border_style="green",
        ))
        return

    console.print(Panel(
        f"[bold white]Update Available[/bold white]\n\n"
        f"  Current: [dim]{current}[/dim]\n"
        f"  Latest:  [bold cyan]{latest}[/bold cyan]",
        border_style="cyan",
    ))

    if check:
        return

    if not yes:
        try:
            ans = console.input(f"\n  Upgrade to {latest}? [Y/n] ").strip().lower()
        except (KeyboardInterrupt, EOFError):
            console.print("\n[dim]Cancelled.[/dim]")
            return
        if ans and ans not in ("y", "yes"):
            console.print("[dim]Cancelled.[/dim]")
            return

    with console.status(f"[dim]Upgrading orca-ai to {latest}...[/dim]"):
        try:
            self_update(yes=True)
        except RuntimeError as e:
            console.print(f"[red]Upgrade failed:[/red] {e}")
            console.print("[dim]Try manually: pip install --upgrade orca-ai[/dim]")
            raise typer.Exit(1)

    console.print(Panel(
        f"[bold green]Upgraded to {latest}[/bold green]\n\n"
        f"  Restart the terminal to use the new version.\n"
        f"  Changes: [dim]https://orca.systems/changelog[/dim]",
        border_style="green",
    ))


@app.command("serve")
def serve(
    host: str = typer.Option("127.0.0.1", "--host", "-H"),
    port: int = typer.Option(7337, "--port", "-p"),
    open_browser: bool = typer.Option(True, "--open/--no-open"),
):
    """Start the Orca web server — browser-based chat UI."""
    import uvicorn
    from orca.serve.api import create_app

    console.print(Panel(
        f"[bold]Orca Web UI[/bold]\n\n"
        f"  [dim]URL:[/dim]    [cyan bold]http://{host}:{port}[/cyan bold]\n"
        f"  [dim]Model:[/dim]  checking...\n\n"
        f"  Stop: [dim]Ctrl+C[/dim]",
        title="[bold blue]◈ Orca Server[/bold blue]",
        border_style="blue",
        expand=False,
    ))

    if open_browser:
        import threading, webbrowser, time
        def _open():
            time.sleep(1.2)
            webbrowser.open(f"http://{host}:{port}")
        threading.Thread(target=_open, daemon=True).start()

    uvicorn.run(create_app(), host=host, port=port, log_level="warning")


if __name__ == "__main__":
    app()
