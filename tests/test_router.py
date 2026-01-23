"""Tests for schema and router."""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import pytest

from ome_writers._router import FrameRouter
from ome_writers._schema import (
    AcquisitionSettings,
    Dimension,
    Position,
    PositionDimension,
    dims_from_standard_axes,
)

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

# ---------------------------------------------------------------------------
# Test cases as data
# ---------------------------------------------------------------------------

# (sizes_dict, expected_outputs) where expected_outputs is list of (pos_key, idx)
ROUTER_CASES = [
    pytest.param(
        # Simple: T=2, C=3
        {"t": 2, "c": 3, "y": 64, "x": 64},
        [
            ("0", (0, 0)),
            ("0", (0, 1)),
            ("0", (0, 2)),
            ("0", (1, 0)),
            ("0", (1, 1)),
            ("0", (1, 2)),
        ],
        id="no-position",
    ),
    pytest.param(
        # Position outermost: P, T, C (Z-stack per position pattern)
        {"p": ["A1", "B2"], "t": 2, "c": 2, "y": 64, "x": 64},
        [
            ("A1", (0, 0)),
            ("A1", (0, 1)),
            ("A1", (1, 0)),
            ("A1", (1, 1)),
            ("B2", (0, 0)),
            ("B2", (0, 1)),
            ("B2", (1, 0)),
            ("B2", (1, 1)),
        ],
        id="position-outermost",
    ),
    pytest.param(
        # Position interleaved: T, P, C (time-lapse across positions pattern)
        {"t": 2, "p": ["A1", "B2"], "c": 2, "y": 64, "x": 64},
        [
            ("A1", (0, 0)),
            ("A1", (0, 1)),
            ("B2", (0, 0)),
            ("B2", (0, 1)),
            ("A1", (1, 0)),
            ("A1", (1, 1)),
            ("B2", (1, 0)),
            ("B2", (1, 1)),
        ],
        id="position-interleaved",
    ),
    pytest.param(
        # Position innermost: T, C, P
        {"t": 2, "c": 2, "p": ["A1", "B2"], "y": 64, "x": 64},
        [
            ("A1", (0, 0)),
            ("B2", (0, 0)),
            ("A1", (0, 1)),
            ("B2", (0, 1)),
            ("A1", (1, 0)),
            ("B2", (1, 0)),
            ("A1", (1, 1)),
            ("B2", (1, 1)),
        ],
        id="position-innermost",
    ),
]

STORAGE_ORDER_CASES = [
    pytest.param(
        # Acquisition TZC, storage TCZ (ngff)
        {"t": 2, "z": 2, "c": 2, "y": 64, "x": 64},
        "ome",
        [
            ("0", (0, 0, 0)),
            ("0", (0, 1, 0)),  # z varies, but stored as c
            ("0", (0, 0, 1)),
            ("0", (0, 1, 1)),
            ("0", (1, 0, 0)),
            ("0", (1, 1, 0)),
            ("0", (1, 0, 1)),
            ("0", (1, 1, 1)),
        ],
        id="tzc-to-tcz",
    ),
]


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("sizes,expected", ROUTER_CASES)
def test_router_iteration(
    sizes: dict[str, int | list[str] | None],
    expected: list[tuple[str, tuple[int, ...]]],
) -> None:
    """Test router yields correct (position_info, index) sequence."""
    settings = AcquisitionSettings(
        root_path="test.zarr",
        dimensions=dims_from_standard_axes(sizes),
        dtype="uint16",
    )
    router = FrameRouter(settings)
    # Extract (pos_name, idx) for comparison with expected
    results = [(settings.positions[pos_idx].name, idx) for pos_idx, idx in router]
    assert results == expected


@pytest.mark.parametrize("sizes,storage_order,expected", STORAGE_ORDER_CASES)
def test_router_storage_order(
    sizes: Mapping[str, int | Sequence[str] | None],
    storage_order: Literal["acquisition", "ome"],
    expected: list[tuple[str, tuple[int, ...]]],
) -> None:
    """Test router applies storage order permutation correctly."""
    settings = AcquisitionSettings(
        root_path="test.zarr",
        dimensions=dims_from_standard_axes(sizes),
        dtype="uint16",
        storage_order=storage_order,
    )
    results = [
        (settings.positions[pos_idx].name, idx)
        for pos_idx, idx in FrameRouter(settings)
    ]
    assert results == expected


def test_settings_positions() -> None:
    """Test settings.positions property."""
    sizes = {"t": 2, "p": ["well_A", "well_B", "well_C"], "y": 64, "x": 64}
    settings = AcquisitionSettings(
        root_path="test.zarr",
        dimensions=dims_from_standard_axes(sizes),
        dtype="uint16",
    )
    positions = settings.positions
    assert [p.name for p in positions] == ["well_A", "well_B", "well_C"]


def test_router_unlimited_dimension() -> None:
    """Test router with unlimited dimension doesn't auto-stop."""
    settings = AcquisitionSettings(
        root_path="test.zarr",
        dimensions=[
            Dimension(name="t", count=None, type="time"),  # Unlimited
            Dimension(name="c", count=2, type="channel"),
            Dimension(name="y", count=64, type="space"),
            Dimension(name="x", count=64, type="space"),
        ],
        dtype="uint16",
    )
    router = FrameRouter(settings)

    # Iterate manually, collecting first N frames
    results = []
    for i, (pos_idx, idx) in enumerate(router):
        results.append((settings.positions[pos_idx].name, idx))
        if i >= 9:  # Stop after 10 frames (5 timepoints x 2 channels)
            break

    # Should get frames for t=0..4, c=0..1 (rightmost varies fastest)
    expected = [
        ("0", (0, 0)),
        ("0", (0, 1)),
        ("0", (1, 0)),
        ("0", (1, 1)),
        ("0", (2, 0)),
        ("0", (2, 1)),
        ("0", (3, 0)),
        ("0", (3, 1)),
        ("0", (4, 0)),
        ("0", (4, 1)),
    ]
    assert results == expected


def test_router_unlimited_with_positions() -> None:
    """Test router with unlimited dimension and positions."""
    settings = AcquisitionSettings(
        root_path="test.zarr",
        dimensions=[
            Dimension(name="t", count=None, type="time"),
            PositionDimension(positions=[Position(name="A1"), Position(name="B2")]),
            Dimension(name="y", count=16, type="space"),
            Dimension(name="x", count=16, type="space"),
        ],
        dtype="uint16",
    )
    router = FrameRouter(settings)

    # Get first 6 frames
    results = []
    for i, (pos_idx, idx) in enumerate(router):
        results.append((settings.positions[pos_idx].name, idx))
        if i >= 5:
            break

    # t varies slowest, positions vary fastest (rightmost)
    expected = [
        ("A1", (0,)),
        ("B2", (0,)),
        ("A1", (1,)),
        ("B2", (1,)),
        ("A1", (2,)),
        ("B2", (2,)),
    ]
    assert results == expected


def test_plate_acquisition_patterns() -> None:
    """Same shape (3 wells x 3 timepoints), different acquisition orders.

    Pattern 1 - Burst timelapse per well:
        Complete a 3-frame timelapse at each well before moving to the next.
        Dimension order: P, T (position outermost)

    Pattern 2 - Round-robin across wells:
        Visit all 3 wells, then repeat 3 times.
        Dimension order: T, P (time outermost)
    """
    wells = [
        Position(name="A1", plate_row="A", plate_column="1"),
        Position(name="B1", plate_row="B", plate_column="1"),
        Position(name="C1", plate_row="C", plate_column="1"),
    ]

    # Pattern 1: Burst timelapse at each well (P outermost)
    # Acquire: A1(t0,t1,t2), B1(t0,t1,t2), C1(t0,t1,t2)
    burst = AcquisitionSettings(
        root_path="test.zarr",
        dimensions=dims_from_standard_axes({"p": wells, "t": 3, "y": 64, "x": 64}),
        dtype="uint16",
    )
    burst_results = [
        (burst.positions[pos_idx].name, idx) for pos_idx, idx in FrameRouter(burst)
    ]
    assert burst_results == [
        ("A1", (0,)),
        ("A1", (1,)),
        ("A1", (2,)),
        ("B1", (0,)),
        ("B1", (1,)),
        ("B1", (2,)),
        ("C1", (0,)),
        ("C1", (1,)),
        ("C1", (2,)),
    ]

    # Pattern 2: Round-robin across wells (T outermost)
    # Acquire: A1(t0), B1(t0), C1(t0), A1(t1), B1(t1), C1(t1), ...
    roundrobin = AcquisitionSettings(
        root_path="test.zarr",
        dimensions=dims_from_standard_axes({"t": 3, "p": wells, "y": 64, "x": 64}),
        dtype="uint16",
    )
    rr_results = [
        (roundrobin.positions[pos_idx].name, idx)
        for pos_idx, idx in FrameRouter(roundrobin)
    ]
    assert rr_results == [
        ("A1", (0,)),
        ("B1", (0,)),
        ("C1", (0,)),
        ("A1", (1,)),
        ("B1", (1,)),
        ("C1", (1,)),
        ("A1", (2,)),
        ("B1", (2,)),
        ("C1", (2,)),
    ]
