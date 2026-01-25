"""Benchmark ome-writers backends with granular phase timing.

run `uv run tools/benchmark.py --help` for usage details.

You can specify data shape either via `--dims`/`-d` or `--settings-file`/`-f`.

Format for --dims:
    `name:count[:chunk[:shard]]` (comma-separated)

Format for --settings-file (JSON) is any AcquisitionSettings dict, for example:
    {
        "dimensions": [
            {"name": "t", "count": 2},
            {"name": "c", "count": 3},
            {"name": "z", "count": 4},
            {"name": "y", "count": 256, "chunk_size": 64},
            {"name": "x", "count": 256, "chunk_size": 64}
        ],
        "dtype": "uint16"
    }

Usage examples:

    # Single backend, simple 3D acquisition
    uv run tools/benchmark.py -d c:3,y:512:128,x:512:128 -b zarr-python

    # 10 timepoints, 2 channels, (2048,2048) frame with (512,512) chunks
    # Multiple backends with compression
    uv run tools/benchmark.py -d t:10,c:2,y:2048:512,x:2048:512 \
        -b zarr-python -b zarrs-python -b tensorstore \
        --compression blosc-zstd \
        --iterations 5

    # With sharding and dtype
    uv run tools/benchmark.py \
        -d t:40:1:10,c:2,z:10:1:5,y:1024:512:2,x:1024:512:2 \
        -b zarrs-python \
        --dtype uint8 --warmups 0 --iterations 3

    # Using a settings file
    uv run tools/benchmark.py -f settings.json -b zarr-python
"""

from __future__ import annotations

import shutil
import sys
import tempfile
import time
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, get_args

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import typer
from _common import generate_frames, parse_dimensions, parse_settings_file
from rich.console import Console
from rich.progress import track
from rich.table import Table

from ome_writers import (
    AcquisitionSettings,
    _schema,
    create_stream,
)
from ome_writers._stream import AVAILABLE_BACKENDS, get_format_for_backend

if TYPE_CHECKING:
    from typing import TypedDict

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


COMPRESSIONS = set(get_args(_schema.Compression))
app = typer.Typer(
    help="Benchmark ome-writers backends",
    pretty_exceptions_enable=False,
    add_completion=False,
    no_args_is_help=True,
)
console = Console()


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
    frames = generate_frames(settings)

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
    frame_shape = frames[0].shape
    outer_shape = [dim.count for dim in settings.dimensions]

    # Display test conditions
    console.print("[bold cyan]Test Conditions:[/bold cyan]")
    console.print(f"  Total shape: {tuple(outer_shape)}")
    console.print(f"  Frame shape: {tuple(frame_shape)}")
    console.print(f"  Number of frames: {num_frames:,}")
    console.print(f"  Data type: {settings.dtype}")

    # Chunk shape - extract from dimensions
    chunk_shape = []
    shard_shape_chunks = []
    has_sharding = False
    for dim in settings.dimensions:
        chunk_shape.append(dim.chunk_size)
        if hasattr(dim, "shard_size_chunks") and dim.shard_size_chunks is not None:
            shard_shape_chunks.append(dim.shard_size_chunks)
            has_sharding = True
        else:
            shard_shape_chunks.append(None)

    console.print(f"  Chunk shape: {tuple(chunk_shape)}")

    # Shard shape (in chunks per shard) if present
    if has_sharding:
        shard_display = tuple(s if s is not None else 1 for s in shard_shape_chunks)
        console.print(f"  Shard shape (chunks/shard): {shard_display}")

    # MB per chunk
    dtype_size = np.dtype(settings.dtype).itemsize
    chunk_elements = np.prod(chunk_shape)
    mb_per_chunk = (chunk_elements * dtype_size) / (1024 * 1024)
    console.print(f"  MB per chunk: {mb_per_chunk:.3f}")

    # Total data size
    console.print(f"  Total data: {total_bytes / (1024**3):.3f} GB")
    console.print(f"  Compression: {settings.compression or 'none'}")
    console.print()

    # Create table with backends as columns
    table = Table()
    table.add_column("Metric", style="cyan", no_wrap=True)

    # Calculate throughput for each backend to determine fastest/slowest
    backend_names = list(results.keys())
    throughputs: dict[str, float | None] = {}
    for backend, result in results.items():
        if isinstance(result, str):
            throughputs[backend] = None
        else:
            write_time = result["write"]["mean"]
            throughputs[backend] = num_frames / write_time if write_time > 0 else None

    # Find fastest and slowest (excluding errors)
    valid_throughputs = {k: v for k, v in throughputs.items() if v is not None}
    fastest_backend = (
        max(valid_throughputs, key=lambda k: valid_throughputs[k])
        if valid_throughputs
        else None
    )
    slowest_backend = (
        min(valid_throughputs, key=lambda k: valid_throughputs[k])
        if valid_throughputs
        else None
    )

    # Add a column for each backend with appropriate header styling
    for backend in backend_names:
        is_error = isinstance(results[backend], str)
        style = "dim" if is_error else "light_cyan3"
        if is_error:
            header_style = "dim"
        elif backend == fastest_backend:
            header_style = "bold green"
        elif backend == slowest_backend:
            header_style = "bold red"
        else:
            header_style = "bold white"
        table.add_column(
            backend, justify="right", header_style=header_style, style=style
        )

    # Build all rows in a single pass through results
    create_row = ["create [dim](mean±std s)"]
    write_row = ["write  [dim](mean±std s)"]
    throughput_row = ["throughput    [dim](fps)"]
    bandwidth_row = ["bandwidth    [dim](GB/s)"]

    p = 3
    for result in results.values():
        if isinstance(result, str):
            create_row.append("ERROR")
            write_row.append("ERROR")
            throughput_row.append("ERROR")
            bandwidth_row.append("ERROR")
        else:
            # Create time
            create = result["create_stream"]
            create_row.append(f"{create['mean']:.{p}f} ± {create['std']:.{p}f}")

            # Write time
            write = result["write"]
            write_row.append(f"{write['mean']:.{p}f} ± {write['std']:.{p}f}")

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


@app.command(no_args_is_help=True)
def main(
    backends: Annotated[
        list[str],
        typer.Option(
            "--backend",
            "-b",
            help="Backend to benchmark (can be specified multiple times).  "
            f"Must be one of {list(AVAILABLE_BACKENDS)}.  Or, "
            "use 'all' for all available backends. ",
        ),
    ] = ["all"],  # noqa
    dimensions: Annotated[
        str | None,
        typer.Option(
            "--dims",
            "-d",
            help=(
                "Dimension spec: name:count[:chunk[:shard]] (comma-separated). "
                "For example, 'c:3,y:512:128,x:512:128:4' defines a 3-channel "
                "image with 512x512 pixels, chunks of (128, 128) and sharding of 4 "
                "chunks per shard in only the x dimension. "
                "Mutually exclusive with --settings-file."
            ),
        ),
    ] = None,
    settings_file: Annotated[
        Path | None,
        typer.Option(
            "--settings-file",
            "-f",
            exists=True,
            file_okay=True,
            dir_okay=False,
            help=(
                "Path to JSON file containing settings. "
                "Expected format: {'dimensions': [...], 'dtype': 'uint16'}. "
                "Mutually exclusive with --dims."
            ),
        ),
    ] = None,
    dtype: Annotated[
        str,
        typer.Option("--dtype", help="Data type (overridden by settings file)"),
    ] = "uint16",
    compression: Annotated[
        _schema.Compression | None,
        typer.Option(
            "--compression",
            "-c",
            help="Compression algorithm.",
        ),
    ] = None,
    warmups: Annotated[
        int,
        typer.Option("--warmups", "-w", help="Number of warmup runs"),
    ] = 1,
    iterations: Annotated[
        int,
        typer.Option("--iterations", "-n", help="Number of benchmark iterations"),
    ] = 10,
) -> None:
    """Benchmark ome-writers backends with granular phase timing.

    Define a synthetic acquisition using --dims or --settings-file, and specify
    one or more backends to benchmark.

    Each backend is benchmarked `iterations` times, and timing is reported for:
    - create: Time for stream initialization and directory creation
    - write: Cumulative time for all frame writes (plus finalization)
    """
    if not backends:
        console.print("[red]Error: At least one --backend must be specified[/red]")
        raise typer.Exit(1)

    # Validate mutual exclusivity of dimensions and settings_file
    if dimensions is None:
        if settings_file is None:
            console.print(
                "[red]Error: Either --dims or --settings-file must be specified[/red]"
            )
            raise typer.Exit(1)
    elif settings_file is not None:
        console.print(
            "[red]Error: --dims and --settings-file are mutually exclusive[/red]"
        )
        raise typer.Exit(1)

    # Parse dimensions and dtype from appropriate source
    if settings_file is not None:
        try:
            settings = parse_settings_file(settings_file, dtype, compression)
        except Exception as e:
            console.print(f"[red]Error parsing settings file: {e}[/red]")
            raise typer.Exit(1) from e
    else:
        try:
            dims = parse_dimensions(dimensions)
        except ValueError as e:
            console.print(f"[red]Error parsing settings: {e}[/red]")
            raise typer.Exit(1) from e

        settings = AcquisitionSettings(
            root_path="tmp",
            dimensions=dims,
            dtype=dtype,
            compression=compression,
        )

    if "all" in backends:
        backends = list(AVAILABLE_BACKENDS)

    dims = "".join(dim.name for dim in settings.dimensions)
    shape = tuple(dim.count for dim in settings.dimensions)
    chunks = tuple(dim.chunk_size for dim in settings.dimensions)

    # Display configuration
    console.print("\n[bold cyan]Benchmark Configuration[/bold cyan]")
    console.print(f"  Backends: {', '.join(backends)}")
    console.print(f"  Dimensions: {dims!r} {shape}")
    console.print(f"  Chunk shape: {chunks}")
    console.print(f"  Total frames: {settings.num_frames:,}")
    console.print(f"  Dtype: {settings.dtype}")
    console.print(f"  Compression: {settings.compression}")
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
