"""Benchmarks for ome-writers write performance.

To run this file, explicitly run:
pytest tests/test_bench.py

or use `pytest --benchmark-only` or `--codspeed` with pytest-codspeed.

These measure ome-writers and how it drives each backend, not the disk, so output
goes to RAM (`/dev/shm`) where available. `tools/benchmark.py` measures real-disk
throughput.
"""

from __future__ import annotations

import os
import shutil
import sys
import tempfile
from itertools import count
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pytest

from ome_writers import AcquisitionSettings, Dimension, create_stream
from tests import conftest

if TYPE_CHECKING:
    from collections.abc import Iterator

    from pytest_benchmark.fixture import BenchmarkFixture

    from ome_writers import OMEStream


if all(
    x not in {"--codspeed", "--benchmark-only", "tests/test_bench.py"} for x in sys.argv
):
    pytest.skip("use --benchmark to run benchmark", allow_module_level=True)


pytestmark = pytest.mark.benchmark
np.random.seed(0)  # For reproducible benchmarks

D = Dimension


# Benchmark cases: subset of integration test cases focused on common patterns
# Optimized for CI speed while preserving chunk crossing and async queue behaviors
BENCHMARK_CASES = [
    pytest.param(
        AcquisitionSettings(
            root_path="tmp",
            dimensions=[
                D(name="y", count=2048, chunk_size=64, type="space"),
                D(name="x", count=2048, chunk_size=64, type="space"),
            ],
            dtype="uint16",
        ),
        id="large_2d",
    ),
    pytest.param(
        AcquisitionSettings(
            root_path="tmp",
            dimensions=[
                D(name="t", count=200, chunk_size=1, type="time"),
                D(name="y", count=128, chunk_size=128, type="space"),
                D(name="x", count=128, chunk_size=128, type="space"),
            ],
            dtype="uint16",
        ),
        id="time_lapse_small_2d",
    ),
    pytest.param(
        AcquisitionSettings(
            root_path="tmp",
            dimensions=[
                D(name="t", count=200, chunk_size=1, shard_size_chunks=20, type="time"),
                D(name="y", count=128, chunk_size=128, type="space"),
                D(name="x", count=128, chunk_size=128, type="space"),
            ],
            dtype="uint16",
        ),
        id="time_lapse_t_shards",
    ),
    pytest.param(
        AcquisitionSettings(
            root_path="tmp",
            dimensions=[
                D(name="z", count=64, chunk_size=16, type="space"),
                D(name="y", count=512, chunk_size=128, type="space"),
                D(name="x", count=512, chunk_size=128, type="space"),
            ],
            dtype="uint16",
        ),
        id="3d_chunking",
    ),
    pytest.param(
        AcquisitionSettings(
            root_path="tmp",
            dimensions=[
                D(name="t", count=5, chunk_size=1, type="time"),
                D(name="c", count=2, chunk_size=1, type="channel"),
                D(name="z", count=3, chunk_size=1, type="space"),
                D(name="y", count=256, chunk_size=128, type="space"),
                D(name="x", count=256, chunk_size=128, type="space"),
            ],
            dtype="uint16",
        ),
        id="t5_c2_z3_xy256",
    ),
]

# Whole acquisitions with metadata on every frame (as pymmcore-plus sends it),
# where the bookkeeping of ome-writers matters as much as the pixels written:
# (dimensions, number of frames appended)
ACQUISITION_CASES = [
    pytest.param(
        [
            D(name="t", count=4, type="time"),
            D(name="p", type="position", coords=[f"P{i}" for i in range(16)]),
            D(name="y", count=64, type="space"),
            D(name="x", count=64, type="space"),
        ],
        64,
        id="16_positions",
    ),
    pytest.param(
        [
            D(name="t", count=None, type="time"),
            D(name="c", count=2, type="channel"),
            D(name="y", count=128, type="space"),
            D(name="x", count=128, type="space"),
        ],
        40,
        id="unbounded_t",
    ),
]


@pytest.fixture
def out_dir(tmp_path: Path) -> Iterator[Path]:
    """Directory for benchmark output, in RAM where possible.

    On a real disk, fsync and writeback make the walltimes bimodal from run to run.
    """
    shm = Path("/dev/shm")
    if not (shm.is_dir() and os.access(shm, os.W_OK)):
        yield tmp_path
        return
    path = Path(tempfile.mkdtemp(prefix="ome-writers-bench-", dir=shm))
    try:
        yield path
    finally:
        shutil.rmtree(path, ignore_errors=True)


def _name_case(case: AcquisitionSettings) -> str:
    """Generate a readable name for a benchmark case."""
    dims = case.dimensions
    dim_names = "_".join(f"{d.name}{d.count}" for d in dims)
    return f"{case.dtype}-{dim_names}"


def _make_frames(settings: AcquisitionSettings) -> list[np.ndarray]:
    """Generate all frame data in memory to isolate write performance."""

    # Calculate total frames needed
    num_frames = settings.num_frames
    if num_frames is None:
        raise NotImplementedError("Dynamic frame counts not supported in benchmarks")

    # Pre-allocate all frames
    dims = settings.dimensions
    frame_shape = tuple(d.count or 1 for d in dims[-2:])
    iinfo = np.iinfo(settings.dtype)
    return [
        np.random.randint(
            iinfo.min, iinfo.max, frame_shape, dtype=np.dtype(settings.dtype)
        )
        for _ in range(num_frames)
    ]


@pytest.mark.parametrize("backend", conftest.AVAILABLE_BACKENDS)
@pytest.mark.parametrize("case", BENCHMARK_CASES)
def test_bench_append(
    backend: str,
    case: AcquisitionSettings,
    benchmark: BenchmarkFixture,
    out_dir: Path,
) -> None:
    """Benchmark append() loop performance with pre-generated data."""
    if len(case.array_dimensions) < 3 and backend == "acquire-zarr":
        pytest.skip("acquire-zarr requires at least 3 array dimensions")
        return

    benchmark.group = _name_case(case)
    settings = AcquisitionSettings.model_validate(
        {
            **case.model_dump(),
            "root_path": str(out_dir / "output"),
            "format": backend,
            "overwrite": True,
        }
    )

    frames = _make_frames(case)

    def setup() -> tuple[tuple, dict]:
        """Create a fresh stream for each benchmark round (not timed)."""
        stream = create_stream(settings)
        return (frames, stream), {}

    def append_all_frames(frames: list[np.ndarray], stream: OMEStream) -> None:
        """Only the append loop is timed."""
        for frame in frames:
            stream.append(frame)
        stream.close()  # flush async writes

    benchmark.pedantic(append_all_frames, setup=setup, rounds=15)


@pytest.mark.parametrize("backend", conftest.AVAILABLE_BACKENDS)
@pytest.mark.parametrize(("dims", "n_frames"), ACQUISITION_CASES)
def test_bench_acquisition(
    backend: str,
    dims: list[Dimension],
    n_frames: int,
    benchmark: BenchmarkFixture,
    out_dir: Path,
) -> None:
    """Benchmark a whole acquisition: create_stream, append with metadata, close."""
    shape = (dims[-2].count or 1, dims[-1].count or 1)
    frame = np.random.randint(0, np.iinfo("uint16").max, shape, dtype="uint16")
    metadata = [
        {"delta_t": i * 0.1, "exposure_time": 10.0, "temperature": 37.0}
        for i in range(n_frames)
    ]
    rounds = count()

    def setup() -> tuple[tuple, dict]:
        """Settings writing to a fresh path for each round (not timed)."""
        root = out_dir / f"round{next(rounds)}" / "output"
        settings = AcquisitionSettings(
            root_path=str(root), dimensions=dims, dtype="uint16", format=backend
        )
        return (settings,), {}

    def acquire(settings: AcquisitionSettings) -> None:
        with create_stream(settings) as stream:
            for meta in metadata:
                stream.append(frame, frame_metadata=meta)

    # (rounds vary by < 3% in RAM, and the slowest round takes > 1 s on CI)
    benchmark.pedantic(acquire, setup=setup, rounds=5)
