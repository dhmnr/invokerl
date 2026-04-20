"""Rich-formatted logging that plays well with the stdlib logger.

Public API:
    rl.setup_logging(level="INFO", verbose=False)
    rl.log_step(step, dt, metrics, is_disagg=False, is_fsdp=False)

`setup_logging()` installs a `rich.logging.RichHandler` as the root handler,
so every `logger.info(...)` call across the package (and vLLM / transformers)
gets colored level columns + pretty tracebacks. No other code has to change.

`log_step()` renders a grouped `Panel` via a shared `rich.Console` —
bypassing the logger's time/level prefix so the panel lays out cleanly.
Interleaves correctly with regular log lines since both share the same
Console instance.
"""

from __future__ import annotations

import logging
from contextlib import contextmanager

from rich.console import Console
from rich.logging import RichHandler
from rich.panel import Panel
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)
from rich.table import Table
from rich.text import Text

# Module-level Console, shared by RichHandler and log_step(). Using stdout so
# redirection (`| tee`) captures both log lines and panels.
_console = Console()


def setup_logging(level: str | int = "INFO", verbose: bool = False) -> None:
    """Install a RichHandler on the root logger.

    Idempotent — safe to call multiple times.

    Args:
        level: Log level for invokerl loggers ("INFO", "DEBUG", int).
        verbose: If True, force level to DEBUG.
    """
    if verbose:
        level = "DEBUG"

    # Remove any pre-existing handlers (e.g. from a prior basicConfig call)
    # so we don't get duplicate lines.
    root = logging.getLogger()
    for h in list(root.handlers):
        root.removeHandler(h)

    handler = RichHandler(
        console=_console,
        show_time=True,
        show_level=True,
        show_path=False,
        rich_tracebacks=True,
        markup=True,
        log_time_format="[%H:%M:%S]",
    )
    root.addHandler(handler)
    root.setLevel(level)


def _kv(key: str, value: str) -> tuple[Text, Text]:
    return Text(key, style="dim"), Text(value, style="bold")


def log_step(
    step: int,
    dt: float,
    metrics: dict,
    is_disagg: bool = False,
    is_fsdp: bool = False,
) -> None:
    """Render a compact grouped panel summarizing one training step.

    Args:
        step: Current step index (0-based).
        dt: Wall-clock seconds for this step.
        metrics: Dict produced by Trainer.train() with loss / reward / kl /
            grad_norm / lr / staleness / weight_version / (disagg extras).
        is_disagg: Include disagg timing breakdown (wait/train/sync/queue).
        is_fsdp: Include FSDP-specific fields (world_size).
    """
    table = Table.grid(padding=(0, 2), expand=False)
    # Three (key, value) columns side-by-side.
    for _ in range(6):
        table.add_column(no_wrap=True)

    # -- Row 1: loss-family -------------------------------------------------
    loss = metrics.get("loss", 0.0)
    reward = metrics.get("reward", 0.0)
    kl = metrics.get("kl", 0.0)
    table.add_row(
        Text("loss", style="dim"),     Text(f"{loss:+.4f}", style="bold"),
        Text("reward", style="dim"),   Text(f"{reward:.3f}", style="bold green"),
        Text("kl", style="dim"),       Text(f"{kl:.4f}", style="bold yellow"),
    )

    # -- Row 2: optimizer health --------------------------------------------
    gnorm = metrics.get("grad_norm", 0.0)
    clip_frac = metrics.get("clip_frac", 0.0)
    lr = metrics.get("lr", 0.0)
    table.add_row(
        Text("gnorm", style="dim"),    Text(f"{gnorm:.2f}", style="bold"),
        Text("clip_frac", style="dim"), Text(f"{clip_frac:.3f}", style="bold"),
        Text("lr", style="dim"),       Text(f"{lr:.1e}", style="bold"),
    )

    # -- Row 3: versioning --------------------------------------------------
    weight_ver = metrics.get("weight_version", 0)
    staleness = metrics.get("staleness", 0.0)
    r3 = [
        Text("version", style="dim"),   Text(f"{weight_ver}", style="bold"),
        Text("staleness", style="dim"), Text(f"{staleness:.1f}", style="bold"),
    ]
    if is_fsdp:
        r3.extend([
            Text("world", style="dim"),
            Text(f"{metrics.get('world_size', 1)}", style="bold"),
        ])
    else:
        r3.extend([Text(""), Text("")])
    table.add_row(*r3)

    # -- Row 4 (disagg only): timing breakdown + queue ----------------------
    if is_disagg:
        t_wait = metrics.get("t_wait", 0.0)
        t_train = metrics.get("t_train", 0.0)
        sync_ms = metrics.get("sync_ms", 0.0)
        queue = metrics.get("queue_size", 0)
        table.add_row(
            Text("wait", style="dim"),  Text(f"{t_wait:.1f}s", style="bold"),
            Text("train", style="dim"), Text(f"{t_train:.1f}s", style="bold"),
            Text("sync", style="dim"),  Text(f"{sync_ms:.0f}ms", style="bold"),
        )
        table.add_row(
            Text("queue", style="dim"), Text(f"{queue}", style="bold"),
            Text(""), Text(""),
            Text(""), Text(""),
        )

    title = Text.assemble(
        ("step ", "dim"),
        (f"{step:>4d}", "bold cyan"),
        (" · ", "dim"),
        (f"{dt:.1f}s", "bold"),
    )
    panel = Panel(table, title=title, title_align="left", border_style="cyan", expand=False)
    _console.print(panel)


def get_console() -> Console:
    """Expose the shared Console for callers that want to `.print` directly."""
    return _console


@contextmanager
def training_progress(total_steps: int, start_step: int = 0):
    """Live progress bar for the training loop.

    Usage inside Trainer.train():

        with training_progress(cfg.total_steps, start_step) as advance:
            for step in range(start_step, cfg.total_steps):
                ...
                advance()

    Panels and log lines printed through the shared Console interleave
    naturally above the bar.
    """
    with Progress(
        SpinnerColumn(style="cyan"),
        TextColumn("[bold]training[/]"),
        BarColumn(bar_width=None),
        MofNCompleteColumn(),
        TextColumn("·"),
        TimeElapsedColumn(),
        TextColumn("· eta"),
        TimeRemainingColumn(),
        console=_console,
        transient=False,
    ) as progress:
        task = progress.add_task("train", total=total_steps, completed=start_step)

        def advance(n: int = 1) -> None:
            progress.advance(task, n)

        yield advance
