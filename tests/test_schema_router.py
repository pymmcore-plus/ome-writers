"""Tests for schema and router."""

from __future__ import annotations

from typing import Literal

import pytest
from pydantic import ValidationError

from ome_writers.router import FrameRouter
from ome_writers.schema_pydantic import (
    ArraySettings,
    Dimension,
    Plate,
    Position,
    PositionDimension,
    dims_from_standard_axes,
)

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
        "ngff",
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
    """Test router yields correct (position_key, index) sequence."""
    settings = ArraySettings(dimensions=dims_from_standard_axes(sizes), dtype="uint16")
    router = FrameRouter(settings)
    assert list(router) == expected


@pytest.mark.parametrize("sizes,storage_order,expected", STORAGE_ORDER_CASES)
def test_router_storage_order(
    sizes: dict[str, int | list[str] | None],
    storage_order: Literal["acquisition", "ngff"],
    expected: list[tuple[str, tuple[int, ...]]],
) -> None:
    """Test router applies storage order permutation correctly."""
    settings = ArraySettings(
        dimensions=dims_from_standard_axes(sizes),
        dtype="uint16",
        storage_order=storage_order,
    )
    assert list(FrameRouter(settings)) == expected


def test_router_position_keys() -> None:
    """Test router.position_keys property."""
    sizes = {"t": 2, "p": ["well_A", "well_B", "well_C"], "y": 64, "x": 64}
    settings = ArraySettings(dimensions=dims_from_standard_axes(sizes), dtype="uint16")
    assert FrameRouter(settings).position_keys == ["well_A", "well_B", "well_C"]


def test_schema_unique_dimension_names() -> None:
    """Test that duplicate dimension names raise ValueError."""
    with pytest.raises(ValidationError, match="unique"):
        ArraySettings(
            dimensions=[
                Dimension(name="t", count=10),
                Dimension(name="t", count=5),
                Dimension(name="y", count=64, type="space"),
                Dimension(name="x", count=64, type="space"),
            ],
            dtype="uint16",
        )


def test_schema_unlimited_first_only() -> None:
    """Test that only first dimension can be unlimited (count=None)."""
    # First dimension unlimited - OK
    ArraySettings(
        dimensions=[
            Dimension(name="t", count=None),
            Dimension(name="y", count=64, type="space"),
            Dimension(name="x", count=64, type="space"),
        ],
        dtype="uint16",
    )

    # Second dimension unlimited - error
    with pytest.raises(ValidationError, match="Only one dimension may be unlimited"):
        ArraySettings(
            dimensions=[
                Dimension(name="t", count=None),
                Dimension(name="c", count=None),
                Dimension(name="y", count=64, type="space"),
                Dimension(name="x", count=64, type="space"),
            ],
            dtype="uint16",
        )


def test_plate_metadata() -> None:
    """Test Plate is structural metadata only."""
    plate = Plate(
        row_names=["A", "B", "C", "D"],
        column_names=["1", "2", "3"],
        name="MyPlate",
    )
    assert len(plate.row_names) == 4
    assert len(plate.column_names) == 3
    assert plate.name == "MyPlate"


def test_position_with_plate_context() -> None:
    """Test Position can carry plate row/column info."""
    pos_dim = PositionDimension(
        positions=[
            Position(name="A1/0", row="A", column="1"),
            Position(name="A1/1", row="A", column="1"),
            Position(name="B2/0", row="B", column="2"),
        ]
    )

    assert pos_dim.count == 3
    assert pos_dim.names == ["A1/0", "A1/1", "B2/0"]
    assert pos_dim.positions[0].row == "A"
    assert pos_dim.positions[2].column == "2"


def test_plate_acquisition_patterns() -> None:
    """Same shape (3 wells × 3 timepoints), different acquisition orders.

    Pattern 1 - Burst timelapse per well:
        Complete a 3-frame timelapse at each well before moving to the next.
        Dimension order: P, T (position outermost)

    Pattern 2 - Round-robin across wells:
        Visit all 3 wells, then repeat 3 times.
        Dimension order: T, P (time outermost)
    """
    wells = [
        Position(name="A1", row="A", column="1"),
        Position(name="B1", row="B", column="1"),
        Position(name="C1", row="C", column="1"),
    ]

    # Pattern 1: Burst timelapse at each well (P outermost)
    # Acquire: A1(t0,t1,t2), B1(t0,t1,t2), C1(t0,t1,t2)
    burst = ArraySettings(
        dimensions=dims_from_standard_axes({"p": wells, "t": 3, "y": 64, "x": 64}),
        dtype="uint16",
    )
    assert list(FrameRouter(burst)) == [
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
    roundrobin = ArraySettings(
        dimensions=dims_from_standard_axes({"t": 3, "p": wells, "y": 64, "x": 64}),
        dtype="uint16",
    )
    assert list(FrameRouter(roundrobin)) == [
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
