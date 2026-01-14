"""Tests for ZarrBackend."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest
import yaozarrs

from ome_writers.backends._zarr import ZarrBackend
from ome_writers.router import FrameRouter
from ome_writers.schema_pydantic import (
    AcquisitionSettings,
    ArraySettings,
    Dimension,
    Position,
    PositionDimension,
    dims_from_standard_axes,
)

if TYPE_CHECKING:
    from collections.abc import Mapping
    from pathlib import Path

zarr = pytest.importorskip("zarr")
pytest.importorskip("yaozarrs")

# ---------------------------------------------------------------------------
# Test cases as data
# ---------------------------------------------------------------------------

# (sizes_dict, expected_shape_per_position)
WRITE_CASES = [
    pytest.param(
        {"t": 2, "c": 2, "y": 64, "x": 64},
        {"0": (2, 2, 64, 64)},
        id="single-position-tcyx",
    ),
    pytest.param(
        {"t": 3, "y": 32, "x": 32},
        {"0": (3, 32, 32)},
        id="single-position-tyx",
    ),
    pytest.param(
        {"p": ["A1", "B1"], "t": 2, "y": 16, "x": 16},
        {"A1": (2, 16, 16), "B1": (2, 16, 16)},
        id="multi-position",
    ),
    pytest.param(
        {"t": 2, "p": ["X", "Y", "Z"], "c": 2, "y": 8, "x": 8},
        {"X": (2, 2, 8, 8), "Y": (2, 2, 8, 8), "Z": (2, 2, 8, 8)},
        id="position-interleaved",
    ),
]


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("sizes,expected_shapes", WRITE_CASES)
def test_zarr_backend_write(
    sizes: Mapping[str, int | list[str]],
    expected_shapes: dict[str, tuple[int, ...]],
    tmp_path: Path,
) -> None:
    """Test backend writes correct shapes for various configurations."""
    array_settings = ArraySettings(
        dimensions=dims_from_standard_axes(sizes), dtype="uint16"
    )
    settings = AcquisitionSettings(
        root_path=str(tmp_path / "test.zarr"),
        array_settings=array_settings,
        overwrite=True,
    )
    router = FrameRouter(array_settings)
    backend = ZarrBackend()

    backend.prepare(settings, router)

    for pos_key, idx in router:
        shape = expected_shapes[pos_key][-2:]  # Y, X from expected shape
        backend.write(pos_key, idx, np.zeros(shape, dtype=np.uint16))

    backend.finalize()

    # Verify output shapes
    store = zarr.open(settings.root_path, mode="r")
    for pos_key, expected_shape in expected_shapes.items():
        array_path = f"{pos_key}/0" if len(expected_shapes) > 1 else "0"
        assert store[array_path].shape == expected_shape

    yaozarrs.validate_zarr_store(settings.root_path)


def test_zarr_backend_unlimited_dimension(tmp_path: Path) -> None:
    """Test backend handles unlimited dimensions with auto-resizing."""
    # Create settings with unlimited time dimension
    array_settings = ArraySettings(
        dimensions=[
            Dimension(name="t", count=None, type="time"),  # Unlimited
            Dimension(name="c", count=2, type="channel"),
            Dimension(name="y", count=32, type="space"),
            Dimension(name="x", count=32, type="space"),
        ],
        dtype="uint16",
    )
    settings = AcquisitionSettings(
        root_path=str(tmp_path / "test.zarr"),
        array_settings=array_settings,
        overwrite=True,
    )
    router = FrameRouter(array_settings)
    backend = ZarrBackend()

    # Verify compatibility
    assert not backend.is_incompatible(settings)

    backend.prepare(settings, router)

    # Write more frames than initial size (starts at 1 for unlimited dim)
    # Manually iterate and break after N frames to test unlimited behavior
    frame_count = 0
    max_frames = 10  # Write 10 timepoints x 2 channels = 20 frames

    for pos_key, idx in router:
        if frame_count >= max_frames:
            break
        backend.write(pos_key, idx, np.zeros((32, 32), dtype=np.uint16))
        frame_count += 1

    backend.finalize()

    # Verify output shape - should have grown to accommodate 5 timepoints
    store = zarr.open(settings.root_path, mode="r")
    assert store["0"].shape == (5, 2, 32, 32)  # t=5 (10 frames / 2 channels)


def test_zarr_backend_unlimited_multiposition(tmp_path: Path) -> None:
    """Test unlimited dimensions with multiple positions."""
    array_settings = ArraySettings(
        dimensions=[
            Dimension(name="t", count=None, type="time"),  # Unlimited
            PositionDimension(positions=[Position(name="A1"), Position(name="B2")]),
            Dimension(name="y", count=16, type="space"),
            Dimension(name="x", count=16, type="space"),
        ],
        dtype="uint16",
    )
    settings = AcquisitionSettings(
        root_path=str(tmp_path / "test.zarr"),
        array_settings=array_settings,
        overwrite=True,
    )
    router = FrameRouter(array_settings)
    backend = ZarrBackend()

    backend.prepare(settings, router)

    # Write 6 frames: 3 timepoints x 2 positions
    frame_count = 0
    for pos_key, idx in router:
        if frame_count >= 6:
            break
        backend.write(pos_key, idx, np.zeros((16, 16), dtype=np.uint16))
        frame_count += 1

    backend.finalize()

    # Verify both positions have t=3
    store = zarr.open(settings.root_path, mode="r")
    assert store["A1/0"].shape == (3, 16, 16)
    assert store["B2/0"].shape == (3, 16, 16)
