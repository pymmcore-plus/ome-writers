"""Benchmarks for ome-writers write performance.

To run this file, explicitly run:
pytest tests/test_bench.py

or use `pytest --benchmark-only` or `--codspeed` with pytest-codspeed.
"""

from __future__ import annotations

import sys
from typing import TYPE_CHECKING, cast

import numpy as np
import pytest

from ome_writers import AcquisitionSettings, Dimension, _stream, create_stream

if TYPE_CHECKING:
    from pathlib import Path

    from pytest_benchmark.fixture import BenchmarkFixture

    from ome_writers import OMEStream

if all(
    x not in {"--codspeed", "--benchmark-only", "tests/test_bench.py"} for x in sys.argv
):
    pytest.skip("use --benchmark to run benchmark", allow_module_level=True)


pytestmark = pytest.mark.benchmark
np.random.seed(0)  # For reproducible benchmarks

D = Dimension

ZARR_BACKENDS = [
    b.name for b in _stream.BACKENDS if b.is_available() and b.format == "zarr"
]

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
    frame_shape = cast("tuple[int, ...]", tuple(d.count for d in dims[-2:]))
    iinfo = np.iinfo(settings.dtype)
    return [
        np.random.randint(iinfo.min, iinfo.max, frame_shape, dtype=settings.dtype)  # ty: ignore
        for _ in range(num_frames)
    ]


@pytest.mark.parametrize("backend", ZARR_BACKENDS)
@pytest.mark.parametrize("case", BENCHMARK_CASES)
def test_bench_append(
    backend: str,
    case: AcquisitionSettings,
    benchmark: BenchmarkFixture,
    tmp_path: Path,
) -> None:
    """Benchmark append() loop performance with pre-generated data."""
    if len(case.array_dimensions) < 3 and backend == "acquire-zarr":
        pytest.skip("acquire-zarr requires at least 3 array dimensions")
        return

    benchmark.group = _name_case(case)
    settings = case.model_copy(
        update={
            "root_path": str(tmp_path / f"output.ome.{case.format}"),
            "backend": backend,
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
        stream._backend.finalize()

    benchmark.pedantic(append_all_frames, setup=setup, rounds=15)
