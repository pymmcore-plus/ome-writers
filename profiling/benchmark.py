"""Benchmark ome-writers backends with granular phase timing.

Usage examples:
    # Single backend, simple 3D acquisition
    python profiling/benchmark.py \\
        --dims "c:3,y:512:128,x:512:128" \\
        --backend zarr-python

    # Multiple backends with compression
    python profiling/benchmark.py \\
        --dims "t:100,c:2,y:2048:256,x:2048:256" \\
        -b zarr-python -b zarrs-python -b tensorstore \\
        --compression blosc-zstd \\
        --iterations 10 --warmups 3

    # With sharding
    python profiling/benchmark.py \\
        --dims "t:100:1:10,c:4,z:20:1:5,y:1024:128,x:1024:128" \\
        -b zarrs-python \\
        --dtype uint8
"""

from __future__ import annotations

import shutil
import tempfile
import time
from pathlib import Path
from typing import TYPE_CHECKING, Annotated

import numpy as np
import typer
from rich.console import Console
from rich.progress import track
from rich.table import Table

from ome_writers import (
    AcquisitionSettings,
    Dimension,
    PositionDimension,
    create_stream,
    dims_from_standard_axes,
)
from ome_writers._stream import AVAILABLE_BACKENDS, get_format_for_backend

if TYPE_CHECKING:
    from jedi.inference.gradual.typing import TypedDict

    class TimingDict(TypedDict):
        create_stream: float
        append: float
        finalize: float
        total: float
        write: float

    class SummaryDict(TypedDict):
        mean: float
        std: float
        min: float
        max: float
        median: float

    class ResultsDict(TypedDict):
        create_stream: SummaryDict
        append: SummaryDict
        finalize: SummaryDict
        write: SummaryDict
        total: SummaryDict


app = typer.Typer(help="Benchmark ome-writers backends")
console = Console()


def parse_dimensions(dim_spec: str) -> list[Dimension | PositionDimension]:
    """Parse compact dimension specification.

    Format: name:count[:chunk[:shard]]
    Example: t:10:1,c:3,z:5,y:512:64,x:512:64
    """
    sizes: dict[str, int] = {}
    chunk_shapes: dict[str, int] = {}
    shard_shapes: dict[str, int | None] = {}
    for spec in dim_spec.split(","):
        if len(parts := spec.split(":")) < 2:
            raise ValueError(f"Invalid dimension spec: {spec} (need name:count)")

        name = parts[0]
        sizes[name] = int(parts[1])
        chunk_shapes[name] = int(parts[2]) if len(parts) > 2 else 1
        shard_shapes[name] = int(parts[3]) if len(parts) > 3 else None

    return dims_from_standard_axes(sizes, chunk_shapes, shard_shapes)


def run_benchmark_iteration(
    settings: AcquisitionSettings, frames: list[np.ndarray]
) -> TimingDict:
    """Run a single benchmark iteration and return phase timings."""
    tmp_path = Path(tempfile.mkdtemp())
    settings = settings.model_copy(
        update={"root_path": str(tmp_path / settings.root_path)}
    )
    try:
        # Time create_stream
        t0 = time.perf_counter()
        stream = create_stream(settings)
        t1 = time.perf_counter()

        # Time all append calls (cumulative)
        for frame in frames:
            stream.append(frame)
        t2 = time.perf_counter()

        # Time finalize
        stream._backend.finalize()
        t3 = time.perf_counter()

        # print warning if tmp_path is empty (no data written)
        if not any(tmp_path.rglob("*")):
            console.print(f"[yellow]Warning: No data written to {tmp_path}[/yellow]")
    finally:
        shutil.rmtree(tmp_path, ignore_errors=True)

    return {  # ty: ignore
        "create_stream": t1 - t0,
        "append": t2 - t1,
        "finalize": t3 - t2,
        "write": t3 - t1,
        "total": t3 - t0,
    }


def _generate_frames(settings: AcquisitionSettings) -> list[np.ndarray]:
    """Generate random frames based on acquisition settings."""
    dtype = np.dtype(settings.dtype)
    iinfo = np.iinfo(dtype)
    size = (settings.num_frames or 1, *settings.shape[-2:])
    return list(np.random.randint(iinfo.min, iinfo.max, size=size, dtype=dtype))


def run_benchmark(
    settings: AcquisitionSettings,
    frames: list[np.ndarray],
    backend: str,
    warmups: int,
    iterations: int,
) -> ResultsDict:
    """Run benchmark for a single backend with multiple iterations."""
    root = f"test_{backend}.ome.{get_format_for_backend(backend)}"
    settings = settings.model_copy(update={"backend": backend, "root_path": root})

    # Warmup runs
    if warmups > 0:
        console.print(f"  [dim]Running {warmups} warmup(s)...[/dim]")
        for _ in range(warmups):
            run_benchmark_iteration(settings, frames)
            # Clean up warmup data

    # Actual benchmark iterations
    console.print(f"  [dim]Running {iterations} iteration(s)...[/dim]")
    all_timings = [
        run_benchmark_iteration(settings, frames)
        for _ in track(range(iterations), description="  Progress", console=console)
    ]

    # Compute statistics for each phase
    results = {}
    for phase in list(all_timings[0]):
        values = [t[phase] for t in all_timings]  # ty: ignore
        results[phase] = {
            "mean": np.mean(values),
            "std": np.std(values),
            "min": np.min(values),
            "max": np.max(values),
            "median": np.median(values),
        }

    return results  # ty: ignore


def run_all_benchmarks(
    settings: AcquisitionSettings, backends: list[str], warmups: int, iterations: int
) -> tuple[dict[str, ResultsDict | str], list[np.ndarray]]:
    # Run benchmarks
    frames = _generate_frames(settings)

    results: dict[str, ResultsDict | str] = {}
    for b in backends:
        console.print(f"[bold yellow]Benchmarking {b}[/bold yellow]")
        try:
            results[b] = run_benchmark(
                settings=settings,
                frames=frames,
                backend=b,
                warmups=warmups,
                iterations=iterations,
            )
            console.print(f"[green]✓ {b} complete[/green]\n")
        except Exception as e:
            console.print(f"[red]✗ {b} failed: {e}[/red]\n")
            results[b] = str(e)
            raise
    return results, frames


def print_results(
    results: dict[str, ResultsDict | str],
    settings: AcquisitionSettings,
    frames: list[np.ndarray],
) -> None:
    """Display benchmark results in a table with backends as columns."""
    console.print("\n[bold]Benchmark Results[/bold]\n")

    # Calculate metrics
    num_frames = len(frames)
    frame_bytes = frames[0].nbytes
    total_bytes = num_frames * frame_bytes

    # Create table with backends as columns
    table = Table()
    table.add_column("Metric", style="cyan", no_wrap=True)

    # Add a column for each backend
    backend_names = list(results.keys())
    for backend in backend_names:
        is_error = isinstance(results[backend], str)
        style = "dim" if is_error else "bold yellow"
        table.add_column(backend, justify="right", style=style)

    # Build all rows in a single pass through results
    create_row = ["create"]
    write_row = ["write"]
    throughput_row = ["throughput (fps)"]
    bandwidth_row = ["bandwidth (GB/s)"]

    for result in results.values():
        if isinstance(result, str):
            create_row.append("ERROR")
            write_row.append("ERROR")
            throughput_row.append("ERROR")
            bandwidth_row.append("ERROR")
        else:
            # Create time
            create = result["create_stream"]
            create_row.append(f"{create['mean']:.4f} ± {create['std']:.4f}")

            # Write time
            write = result["write"]
            write_row.append(f"{write['mean']:.4f} ± {write['std']:.4f}")

            # Throughput and bandwidth
            if (write_time := write["mean"]) > 0:
                fps = num_frames / write_time
                gb_per_sec = (total_bytes / 1e9) / write_time
                throughput_row.append(f"{fps:,.1f}")
                bandwidth_row.append(f"{gb_per_sec:.3f}")
            else:
                throughput_row.append("N/A")
                bandwidth_row.append("N/A")

    table.add_row(*create_row)
    table.add_row(*write_row)
    table.add_row(*throughput_row)
    table.add_row(*bandwidth_row)
    console.print(table)


@app.command()
def main(
    dimensions: Annotated[
        str,
        typer.Option(
            "--dims",
            "-d",
            help=(
                "Dimension spec: name:count[:chunk[:shard]] (comma-separated). "
                "Example: c:3,y:512:128,x:512:128"
            ),
        ),
    ],
    backends: Annotated[
        list[str],
        typer.Option(
            "--backend",
            "-b",
            help="Backend to benchmark (can be specified multiple times).  "
            "Use 'all' for all available backends.",
        ),
    ],
    dtype: Annotated[
        str,
        typer.Option("--dtype", help="Data type"),
    ] = "uint16",
    compression: Annotated[
        str | None,
        typer.Option("--compression", "-c", help="Compression algorithm"),
    ] = None,
    warmups: Annotated[
        int,
        typer.Option("--warmups", "-w", help="Number of warmup runs"),
    ] = 1,
    iterations: Annotated[
        int,
        typer.Option("--iterations", "-n", help="Number of benchmark iterations"),
    ] = 5,
) -> None:
    """Benchmark ome-writers backends with granular phase timing.

    Each backend is benchmarked multiple times, and timing is collected for
    three phases:
    - create_stream(): Stream initialization
    - append(): Cumulative time for all frame writes
    - finalize(): Backend finalization
    """
    if not backends:
        console.print("[red]Error: At least one --backend must be specified[/red]")
        raise typer.Exit(1)

    # Parse dimensions
    try:
        dims = parse_dimensions(dimensions)
    except ValueError as e:
        console.print(f"[red]Error parsing dimensions: {e}[/red]")
        raise typer.Exit(1) from e

    settings = AcquisitionSettings(
        root_path="tmp",
        dimensions=dims,
        dtype=dtype,
        compression=compression,
    )

    if "all" in backends:
        backends = list(AVAILABLE_BACKENDS)

    # Display configuration
    console.print("\n[bold cyan]Benchmark Configuration[/bold cyan]")
    console.print(f"  Backends: {', '.join(backends)}")
    console.print(f"  Dimensions: {dimensions}")
    console.print(f"  Total frames: {settings.num_frames:,}")
    console.print(f"  Dtype: {dtype}")
    console.print(f"  Compression: {compression or 'none'}")
    console.print(f"  Warmups: {warmups}")
    console.print(f"  Iterations: {iterations}\n")

    results, frames = run_all_benchmarks(
        settings=settings,
        backends=backends,
        warmups=warmups,
        iterations=iterations,
    )

    print_results(results, settings, frames)


if __name__ == "__main__":
    app()
