"""Tests for ZarrBackend."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest
import yaozarrs

from ome_writers.backends._zarr import ZarrBackend
from ome_writers.router import FrameRouter
from ome_writers.schema import (
    AcquisitionSettings,
    ArraySettings,
    Dimension,
    Plate,
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

    for pos_info, idx in router:
        pos_name = pos_info[1].name
        shape = expected_shapes[pos_name][-2:]  # Y, X from expected shape
        backend.write(pos_info, idx, np.zeros(shape, dtype=np.uint16))

    backend.finalize()

    # Verify output shapes
    store = zarr.open(settings.root_path, mode="r")
    for pos_name, expected_shape in expected_shapes.items():
        array_path = f"{pos_name}/0" if len(expected_shapes) > 1 else "0"
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

    for pos_info, idx in router:
        if frame_count >= max_frames:
            break
        backend.write(pos_info, idx, np.zeros((32, 32), dtype=np.uint16))
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
    for pos_info, idx in router:
        if frame_count >= 6:
            break
        backend.write(pos_info, idx, np.zeros((16, 16), dtype=np.uint16))
        frame_count += 1

    backend.finalize()

    # Verify both positions have t=3
    store = zarr.open(settings.root_path, mode="r")
    assert store["A1/0"].shape == (3, 16, 16)
    assert store["B2/0"].shape == (3, 16, 16)


def test_zarr_backend_plate(tmp_path: Path) -> None:
    """Test backend creates plate structure with wells and FOVs."""
    array_settings = ArraySettings(
        dimensions=[
            Dimension(name="t", count=2, type="time"),
            PositionDimension(
                positions=[
                    Position(name="fov0", row="A", column="1"),
                    Position(name="fov0", row="A", column="2"),
                    Position(name="fov0", row="C", column="4"),
                    Position(name="fov1", row="C", column="4"),  # 2 FOVs in same well
                ]
            ),
            Dimension(name="c", count=2, type="channel"),
            Dimension(name="y", count=16, type="space"),
            Dimension(name="x", count=16, type="space"),
        ],
        dtype="uint16",
    )
    settings = AcquisitionSettings(
        root_path=str(tmp_path / "plate.ome.zarr"),
        array_settings=array_settings,
        plate=Plate(
            name="Test Plate",
            row_names=["A", "B", "C"],
            column_names=["1", "2", "3", "4"],
        ),
        overwrite=True,
    )
    router = FrameRouter(array_settings)
    backend = ZarrBackend()

    assert not backend.is_incompatible(settings)
    backend.prepare(settings, router)

    # Write all frames
    for pos_info, idx in router:
        backend.write(pos_info, idx, np.zeros((16, 16), dtype=np.uint16))

    backend.finalize()

    # Verify plate structure - FOV paths use Position.name
    store = zarr.open(settings.root_path, mode="r")

    # Check well A/1 has 1 FOV named "fov0"
    assert store["A/1/fov0/0"].shape == (2, 2, 16, 16)

    # Check well A/2 has 1 FOV named "fov0"
    assert store["A/2/fov0/0"].shape == (2, 2, 16, 16)

    # Check well C/4 has 2 FOVs named "fov0" and "fov1"
    assert store["C/4/fov0/0"].shape == (2, 2, 16, 16)
    assert store["C/4/fov1/0"].shape == (2, 2, 16, 16)

    # Validate the zarr store
    yaozarrs.validate_zarr_store(settings.root_path)


def test_zarr_backend_plate_duplicate_names_in_well_rejected(tmp_path: Path) -> None:
    """Test that duplicate position names within the same well are rejected."""
    with pytest.raises(
        ValueError, match="Position names must be unique within each well"
    ):
        ArraySettings(
            dimensions=[
                Dimension(name="t", count=2, type="time"),
                PositionDimension(
                    positions=[
                        Position(name="fov0", row="C", column="4"),
                        Position(
                            name="fov0", row="C", column="4"
                        ),  # Duplicate in same well
                    ]
                ),
                Dimension(name="y", count=16, type="space"),
                Dimension(name="x", count=16, type="space"),
            ],
            dtype="uint16",
        )


def test_zarr_backend_plate_same_name_different_wells(tmp_path: Path) -> None:
    """Test that the same position name can be used in different wells."""
    array_settings = ArraySettings(
        dimensions=[
            Dimension(name="t", count=2, type="time"),
            PositionDimension(
                positions=[
                    Position(name="fov0", row="A", column="1"),
                    Position(
                        name="fov0", row="B", column="2"
                    ),  # Same name, different well
                ]
            ),
            Dimension(name="y", count=16, type="space"),
            Dimension(name="x", count=16, type="space"),
        ],
        dtype="uint16",
    )
    settings = AcquisitionSettings(
        root_path=str(tmp_path / "plate.ome.zarr"),
        array_settings=array_settings,
        plate=Plate(row_names=["A", "B"], column_names=["1", "2"]),
        overwrite=True,
    )
    router = FrameRouter(array_settings)
    backend = ZarrBackend()

    backend.prepare(settings, router)

    for pos_info, idx in router:
        backend.write(pos_info, idx, np.zeros((16, 16), dtype=np.uint16))

    backend.finalize()

    # Both wells have FOV named "fov0"
    store = zarr.open(settings.root_path, mode="r")
    assert store["A/1/fov0/0"].shape == (2, 16, 16)
    assert store["B/2/fov0/0"].shape == (2, 16, 16)

    yaozarrs.validate_zarr_store(settings.root_path)


def test_zarr_backend_plate_requires_row_column(tmp_path: Path) -> None:
    """Test that plate mode requires row/column on positions."""
    array_settings = ArraySettings(
        dimensions=[
            Dimension(name="t", count=2, type="time"),
            PositionDimension(
                positions=[
                    Position(name="A1"),  # Missing row/column
                ]
            ),
            Dimension(name="y", count=16, type="space"),
            Dimension(name="x", count=16, type="space"),
        ],
        dtype="uint16",
    )
    settings = AcquisitionSettings(
        root_path=str(tmp_path / "plate.ome.zarr"),
        array_settings=array_settings,
        plate=Plate(row_names=["A"], column_names=["1"]),
        overwrite=True,
    )
    router = FrameRouter(array_settings)
    backend = ZarrBackend()

    with pytest.raises(ValueError, match="must have row and column"):
        backend.prepare(settings, router)
