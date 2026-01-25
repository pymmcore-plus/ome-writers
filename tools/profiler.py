"""Profile ome-writers backends to identify performance bottlenecks.

Profiles the write path (append loop + finalize) with setup excluded.

Usage:
    uv run python tools/profiler.py -b zarr-python \
        -d t:20,c:2,y:1024:512,x:1024:512
    uv run python tools/profiler.py -b zarrs-python \
        -d t:40:1:10,c:2,z:10:1:5,y:1024:512:2,x:1024:512:2
    uv run python tools/profiler.py -b tensorstore \
        -f settings.json --top-n 10 --sort cumulative
"""

from __future__ import annotations

import contextlib
import cProfile
import pstats
import shutil
import site
import sys
import sysconfig
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Literal

import typer
from rich.console import Console
from rich.table import Table

from ome_writers import AcquisitionSettings, _schema, create_stream
from ome_writers._stream import AVAILABLE_BACKENDS, get_format_for_backend

sys.path.insert(0, str(Path(__file__).parent))
from _common import generate_frames, parse_dimensions, parse_settings_file

if TYPE_CHECKING:
    from typing import TypedDict

    import numpy as np

    from ome_writers._stream import Stream

    class ProfileItem(TypedDict):
        filename: str
        lineno: int
        funcname: str
        ncalls: str
        tottime: float
        cumtime: float


app = typer.Typer(
    no_args_is_help=True, add_completion=False, pretty_exceptions_show_locals=False
)
console = Console()

WORKING_DIR = Path.cwd()

SITE_PACKAGES = Path(pkg[0]).resolve() if (pkg := site.getsitepackages()) else None
STDLIB = Path(sysconfig.get_path("stdlib") or "").resolve()
COLORS = {
    "project": "bold green",
    "package": "blue",
    "stdlib": "dim white",
    "other": "white",
}


def shorten_path(filepath: str) -> tuple[str, str]:
    """Return (shortened_path, category) for filepath."""
    p = Path(filepath)
    if SITE_PACKAGES:
        with contextlib.suppress(ValueError):
            return f"<sitepackages>/{p.relative_to(SITE_PACKAGES)}", "package"
    with contextlib.suppress(ValueError):
        return f"<stdlib>/{p.relative_to(STDLIB)}", "stdlib"
    with contextlib.suppress(ValueError):
        return str(p.relative_to(WORKING_DIR)), "project"
    return filepath, "other"


def write_frames(stream: Stream, frames: list[np.ndarray]) -> None:
    """Profile target: append loop + finalize."""
    for frame in frames:
        stream.append(frame)
    stream._backend.finalize()


def do_profiling(settings: AcquisitionSettings) -> cProfile.Profile:
    """Run profiler and return profiler object."""
    console.print("[dim]Generating frames...[/dim]")
    frames = generate_frames(settings)

    console.print("[dim]Setting up stream...[/dim]")
    tmp_path = Path(tempfile.mkdtemp())
    try:
        settings = settings.model_copy(
            update={"root_path": str(tmp_path / settings.root_path)}
        )
        stream = create_stream(settings)

        console.print("[bold yellow]Profiling append + finalize...[/bold yellow]")
        profiler = cProfile.Profile()
        profiler.runcall(write_frames, stream, frames)
    finally:
        shutil.rmtree(tmp_path, ignore_errors=True)

    return profiler


def extract_profile_items(profiler: cProfile.Profile, sort: str) -> list[ProfileItem]:
    """Extract and sort profile items from profiler."""
    stats = pstats.Stats(profiler)

    items: list[ProfileItem] = []
    for (filename, lineno, funcname), (cc, nc, tt, ct, _) in stats.stats.items():  # type: ignore
        items.append(
            {
                "filename": filename,
                "lineno": lineno,
                "funcname": funcname,
                "ncalls": f"{cc}/{nc}" if cc != nc else str(nc),
                "tottime": tt,
                "cumtime": ct,
            }
        )

    sort_field = {"time": "tottime", "cumulative": "cumtime", "calls": "ncalls"}[sort]
    if sort == "calls":
        items.sort(key=lambda x: int(x["ncalls"].split("/")[0]), reverse=True)
    else:
        items.sort(key=lambda x: x[sort_field], reverse=True)

    return items


def print_results(items: list[ProfileItem], top_n: int, sort: str) -> None:
    """Display profiling results in a Rich table."""
    table = Table(title=f"Top {top_n} functions by {sort}")
    table.add_column("Calls", justify="right", style="dim")
    table.add_column("Time", justify="right", style="bold cyan")
    table.add_column("Cumulative", justify="right", style="yellow")
    table.add_column("Location", style="white")

    for item in items[:top_n]:
        short_path, category = shorten_path(item["filename"])
        location = f"{short_path}:{item['lineno']}({item['funcname']})"
        table.add_row(
            item["ncalls"],
            f"{item['tottime']:.3f}s",
            f"{item['cumtime']:.3f}s",
            f"[{COLORS[category]}]{location}[/{COLORS[category]}]",
        )

    console.print()
    console.print(table)


@app.command()
def main(
    backend: Annotated[str, typer.Option("--backend", "-b")],
    dimensions: Annotated[
        str | None,
        typer.Option("--dims", "-d", help="e.g. t:10,c:2,y:1024:512,x:1024:512"),
    ] = None,
    settings_file: Annotated[Path | None, typer.Option("--settings-file", "-f")] = None,
    dtype: Annotated[str, typer.Option()] = "uint16",
    compression: Annotated[_schema.Compression | None, typer.Option("-c")] = None,
    top_n: Annotated[int, typer.Option(help="Number of top functions to show")] = 20,
    sort: Annotated[
        Literal["time", "cumulative", "calls"],
        typer.Option(help="Sort by: time (self time), cumulative, or calls"),
    ] = "time",
) -> None:
    """Profile ome-writers backend to find performance bottlenecks."""
    if backend not in AVAILABLE_BACKENDS:
        console.print(f"[red]Unknown backend: {backend}[/red]")
        console.print(f"Available: {', '.join(AVAILABLE_BACKENDS)}")
        raise typer.Exit(1)

    if dimensions and settings_file:
        console.print("[red]Cannot specify both --dims and --settings-file[/red]")
        raise typer.Exit(1)
    if not dimensions and not settings_file:
        console.print("[red]Must specify either --dims or --settings-file[/red]")
        raise typer.Exit(1)

    if settings_file:
        settings = parse_settings_file(settings_file, dtype, compression)
    else:
        dims = parse_dimensions(dimensions)
        settings = AcquisitionSettings(
            root_path="tmp", dimensions=dims, dtype=dtype, compression=compression
        )

    root = f"test_{backend}.ome.{get_format_for_backend(backend)}"
    settings = settings.model_copy(update={"backend": backend, "root_path": root})

    console.print(f"\n[bold]Profiling {backend}[/bold]")
    console.print(f"  Shape: {tuple(d.count for d in settings.dimensions)}")
    console.print(f"  Chunks: {tuple(d.chunk_size for d in settings.dimensions)}")
    console.print(f"  Frames: {settings.num_frames}\n")

    profiler = do_profiling(settings)
    items = extract_profile_items(profiler, sort)
    print_results(items, top_n, sort)


if __name__ == "__main__":
    app()
