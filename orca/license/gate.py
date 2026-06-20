"""
Orca feature gating — check license tier before running premium commands.

Usage (decorator):
    @require_license("ultra", tier="pro")
    def ultra_run(...): ...

Usage (inline):
    gate("cloud_train")   # raises SystemExit if not licensed

Usage (boolean):
    if has_feature("ultra"):
        ...
"""
from __future__ import annotations

import functools
from typing import Callable, TypeVar

from orca.license.keys import TIER_FEATURES
from orca.license.store import current_tier

F = TypeVar("F", bound=Callable)

# Order matters for tier comparison
_TIER_ORDER = ["free", "pro", "enterprise"]


def has_feature(feature: str) -> bool:
    """True if the current active license includes this feature."""
    tier = current_tier()
    allowed = TIER_FEATURES.get(tier, set())
    return "*" in allowed or feature in allowed


def required_tier_for(feature: str) -> str:
    """Returns the minimum tier that grants a feature."""
    for tier in _TIER_ORDER:
        allowed = TIER_FEATURES.get(tier, set())
        if "*" in allowed or feature in allowed:
            return tier
    return "enterprise"


def gate(feature: str, soft: bool = False) -> bool:
    """
    Inline license gate.
    If soft=True: print warning and return False instead of raising.
    If soft=False (default): print panel and raise SystemExit(1).
    Returns True if licensed, False if not (only in soft mode).
    """
    if has_feature(feature):
        return True

    needed = required_tier_for(feature)
    _print_gate_panel(feature, needed)

    if soft:
        return False

    import typer
    raise typer.Exit(1)


def require_license(feature: str, tier: str = "pro"):
    """
    Decorator that blocks the function if the feature isn't licensed.

    @require_license("ultra")
    def ultra_run(...): ...
    """
    def decorator(fn: F) -> F:
        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            gate(feature)
            return fn(*args, **kwargs)
        return wrapper  # type: ignore
    return decorator


def _print_gate_panel(feature: str, needed_tier: str) -> None:
    from rich.console import Console
    from rich.panel import Panel
    from rich import box

    active = current_tier()
    feature_label = feature.replace("_", " ").title()

    Console().print(Panel(
        f"[bold yellow]LICENSE REQUIRED[/bold yellow]\n\n"
        f"  [bold]{feature_label}[/bold] is a [bold cyan]{needed_tier.upper()}[/bold cyan] feature.\n\n"
        f"  Your tier:  [dim]{active.upper()}[/dim]\n\n"
        f"  Activate:   [cyan bold]orca activate <your-key>[/cyan bold]\n"
        f"  Get a key:  [cyan]orca license --buy[/cyan]",
        title="[yellow]Orca License[/yellow]",
        border_style="yellow",
        box=box.ROUNDED,
        padding=(0, 2),
    ))
