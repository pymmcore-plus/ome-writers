from __future__ import annotations

import pytest
from pydantic import ValidationError

from ome_writers.schema import (
    AcquisitionSettings,
    Dimension,
    Plate,
    Position,
    PositionDimension,
    dims_from_standard_axes,
)


def test_schema_unique_dimension_names() -> None:
    """Test that duplicate dimension names raise ValueError."""
    with pytest.raises(ValidationError, match="unique"):
        AcquisitionSettings(
            root_path="test.zarr",
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
    AcquisitionSettings(
        root_path="test.zarr",
        dimensions=[
            Dimension(name="t", count=None),
            Dimension(name="y", count=64, type="space"),
            Dimension(name="x", count=64, type="space"),
        ],
        dtype="uint16",
    )

    # Second dimension unlimited - error
    with pytest.raises(
        ValidationError, match=" Only the first dimension may be unbounded"
    ):
        AcquisitionSettings(
            root_path="test.zarr",
            dimensions=[
                Dimension(name="t", count=10),
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


def test_plate_requires_row_column() -> None:
    """Test that plate mode requires row/column on positions."""
    with pytest.raises(
        ValueError, match="All positions must have row and column for plate mode"
    ):
        AcquisitionSettings(
            root_path="plate.ome.zarr",
            dimensions=[
                Dimension(name="t", count=2, type="time"),
                PositionDimension(
                    # Missing row/column
                    positions=[Position(name="A1")]
                ),
                Dimension(name="y", count=16, type="space"),
                Dimension(name="x", count=16, type="space"),
            ],
            dtype="uint16",
            plate=Plate(row_names=["A"], column_names=["1"]),
            overwrite=True,
        )


def test_duplicate_names_rejected() -> None:
    """Test that duplicate position names within the same well are rejected."""
    with pytest.raises(
        ValueError, match="Position names must be unique within each well"
    ):
        AcquisitionSettings(
            root_path="test.zarr",
            dimensions=[
                PositionDimension(
                    positions=[
                        Position(name="fov0", row="C", column="4"),
                        Position(name="fov0", row="C", column="4"),
                    ]
                ),
                Dimension(name="y", count=16, type="space"),
                Dimension(name="x", count=16, type="space"),
            ],
            dtype="uint16",
        )

    with pytest.raises(
        ValueError, match="positions without row/column must have unique names"
    ):
        AcquisitionSettings(
            root_path="test.zarr",
            dimensions=[
                PositionDimension(
                    positions=[
                        Position(name="fov0"),
                        Position(name="fov0"),
                    ]
                ),
                Dimension(name="y", count=16, type="space"),
                Dimension(name="x", count=16, type="space"),
            ],
            dtype="uint16",
        )


def test_acquisition_settings_properties() -> None:
    """Test AcquisitionSettings properties for coverage."""
    # Test is_unbounded property
    settings = AcquisitionSettings(
        root_path="test.zarr",
        dimensions=[
            Dimension(name="t", count=None),
            Dimension(name="y", count=64, type="space"),
            Dimension(name="x", count=64, type="space"),
        ],
        dtype="uint16",
    )
    assert settings.is_unbounded is True

    # Test plate without PositionDimension raises error
    with pytest.raises(
        ValueError, match="Plate mode requires a PositionDimension in dimensions"
    ):
        AcquisitionSettings(
            root_path="plate.zarr",
            dimensions=[
                Dimension(name="t", count=2, type="time"),
                Dimension(name="y", count=16, type="space"),
                Dimension(name="x", count=16, type="space"),
            ],
            dtype="uint16",
            plate=Plate(row_names=["A"], column_names=["1"]),
        )

    # Test dims_from_standard_axes with invalid axis name
    with pytest.raises(ValueError, match="All axes must be one of"):
        dims_from_standard_axes({"invalid": 10, "y": 64, "x": 64})

    # Test dims_from_standard_axes with invalid position value
    with pytest.raises(ValueError, match="Invalid position value"):
        dims_from_standard_axes({"p": "invalid", "y": 64, "x": 64})

    # Test dimension with type="other" for _ngff_sort_key coverage
    settings_with_other = AcquisitionSettings(
        root_path="test.zarr",
        dimensions=[
            Dimension(name="custom", count=5, type="other"),
            Dimension(name="y", count=64, type="space"),
            Dimension(name="x", count=64, type="space"),
        ],
        dtype="uint16",
        storage_order="ngff",
    )
    assert settings_with_other.storage_index_dimensions[0].name == "custom"
