from __future__ import annotations

import pytest
from pydantic import ValidationError

from ome_writers import (
    AcquisitionSettings,
    Channel,
    Dimension,
    Plate,
    Position,
    StandardAxis,
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
    pos_dim = Dimension(
        name="p",
        type="position",
        coords=[
            Position(name="A1/0", plate_row="A", plate_column="1"),
            Position(name="A1/1", plate_row="A", plate_column="1"),
            Position(name="B2/0", plate_row="B", plate_column="2"),
        ],
    )

    assert pos_dim.count == 3
    assert pos_dim.coords
    assert [p.name for p in pos_dim.coords] == ["A1/0", "A1/1", "B2/0"]
    assert pos_dim.coords[0].plate_row == "A"
    assert pos_dim.coords[2].plate_column == "2"


def test_plate_requires_row_column() -> None:
    """Test that plate mode requires row/column on positions."""
    with pytest.raises(
        ValueError, match="All positions must have row and column for plate mode"
    ):
        AcquisitionSettings(
            root_path="plate.ome.zarr",
            dimensions=[
                Dimension(name="t", count=2, type="time"),
                Dimension(
                    name="p",
                    type="position",
                    # Missing row/column
                    coords=[Position(name="A1")],
                ),
                Dimension(name="y", count=16, type="space"),
                Dimension(name="x", count=16, type="space"),
            ],
            dtype="uint16",
            plate=Plate(row_names=["A"], column_names=["1"]),
            overwrite=True,
        )


def test_plate_requires_position_dimension() -> None:
    with pytest.raises(
        ValueError, match="Plate mode requires a position dimension in dimensions"
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


def test_plate_position_warnings(caplog: pytest.LogCaptureFixture) -> None:
    """Test warnings for plate position row/column issues."""
    # Test bad row/column values
    with pytest.warns(
        UserWarning, match="Some positions have row/column values not in the plate"
    ):
        AcquisitionSettings(
            root_path="plate.zarr",
            dimensions=[
                Dimension(name="t", count=2, type="time"),
                Dimension(
                    name="p",
                    type="position",
                    coords=[
                        Position(name="Pos1", plate_row="A", plate_column="1"),
                        Position(name="Pos2", plate_row="C", plate_column="3"),  # Bad
                    ],
                ),
                Dimension(name="y", count=16, type="space"),
                Dimension(name="x", count=16, type="space"),
            ],
            dtype="uint16",
            plate=Plate(row_names=["A"], column_names=["1"]),
        )


def test_duplicate_names_rejected() -> None:
    """Test that duplicate position names within the same well are rejected."""
    with pytest.raises(
        ValueError, match="Position names must be unique within each group"
    ):
        AcquisitionSettings(
            root_path="test.zarr",
            dimensions=[
                Dimension(
                    name="p",
                    type="position",
                    coords=[
                        Position(name="fov0", plate_row="C", plate_column="4"),
                        Position(name="fov0", plate_row="C", plate_column="4"),
                    ],
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
                Dimension(
                    name="p",
                    type="position",
                    coords=[
                        Position(name="fov0"),
                        Position(name="fov0"),
                    ],
                ),
                Dimension(name="y", count=16, type="space"),
                Dimension(name="x", count=16, type="space"),
            ],
            dtype="uint16",
        )


def test_same_name_allowed_in_different_groups() -> None:
    """Test that same position name is allowed across different plate/grid groups.

    This is a real-world scenario: grid scans within multiple wells where each
    well has the same grid layout with the same site names.
    """
    # Same name, same grid coords, but different plate coords
    # (e.g., "fov0" at grid position (0,0) in wells A/1 and B/2)
    pd = Dimension(
        name="p",
        type="position",
        coords=[
            Position(
                name="fov0", plate_row="A", plate_column="1", grid_row=0, grid_column=0
            ),
            Position(
                name="fov0", plate_row="B", plate_column="2", grid_row=0, grid_column=0
            ),
        ],
    )
    assert len(pd.coords) == 2

    # Same name, same plate coords, but different grid coords
    # (e.g., "fov0" at different grid positions within the same well)
    pd = Dimension(
        name="p",
        type="position",
        coords=[
            Position(
                name="fov0", plate_row="A", plate_column="1", grid_row=0, grid_column=0
            ),
            Position(
                name="fov0", plate_row="A", plate_column="1", grid_row=1, grid_column=1
            ),
        ],
    )
    assert len(pd.coords) == 2

    # Same name, same plate AND same grid coords
    with pytest.raises(
        ValueError, match="Position names must be unique within each group"
    ):
        Dimension(
            name="p",
            type="position",
            coords=[
                Position(
                    name="fov0",
                    plate_row="A",
                    plate_column="1",
                    grid_row=0,
                    grid_column=0,
                ),
                Position(
                    name="fov0",
                    plate_row="A",
                    plate_column="1",
                    grid_row=0,
                    grid_column=0,
                ),
            ],
        )


def test_dims_from_standard_axes_names_values() -> None:
    # Test dims_from_standard_axes with invalid axis name
    with pytest.raises(ValueError, match="Standard axes names must be one of"):
        dims_from_standard_axes({"invalid": 10, "y": 64, "x": 64})


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

    # Test dimension with type="other" for _ngff_sort_key coverage
    settings_with_other = AcquisitionSettings(
        root_path="test.zarr",
        dimensions=[
            Dimension(name="custom", count=5, type="other"),
            Dimension(name="y", count=64, type="space"),
            Dimension(name="x", count=64, type="space"),
        ],
        dtype="uint16",
    )
    assert settings_with_other.storage_index_dimensions[0].name == "custom"


def test_invalid_dtype() -> None:
    """Test that invalid dtype raises ValueError."""
    with pytest.raises(ValidationError, match="Invalid dtype"):
        AcquisitionSettings(
            root_path="test.zarr",
            dimensions=[
                Dimension(name="y", count=64, type="space"),
                Dimension(name="x", count=64, type="space"),
            ],
            dtype="not_a_valid_dtype",
        )


def test_storage_order_cannot_permute_last_two_dims() -> None:
    """Test that storage_order cannot permute the last two dimensions."""
    with pytest.raises(
        ValidationError, match="storage_order may not \\(yet\\) permute the last two"
    ):
        AcquisitionSettings(
            root_path="test.zarr",
            dimensions=[
                Dimension(name="t", count=10, type="time"),
                Dimension(name="c", count=3, type="channel"),
                Dimension(name="y", count=64, type="space"),
                Dimension(name="x", count=64, type="space"),
            ],
            dtype="uint16",
            storage_order=["y", "x", "t", "c"],  # Incorrectly puts t,c at the end
        )


def test_standard_axis_methods() -> None:
    """Test StandardAxis dimension_type and unit methods."""
    assert StandardAxis.Z.dimension_type() == "space"
    assert StandardAxis.TIME.dimension_type() == "time"
    assert StandardAxis.CHANNEL.dimension_type() == "channel"
    assert StandardAxis.POSITION.dimension_type() == "position"
    assert StandardAxis.Z.unit() == "micrometer"
    assert StandardAxis.TIME.unit() == "second"
    assert StandardAxis.CHANNEL.unit() is None


def test_multiple_position_dimensions_error() -> None:
    """Test that only one position dimension is allowed."""
    with pytest.raises(ValueError, match="Only one position dimension is allowed"):
        AcquisitionSettings(
            root_path="test.zarr",
            dimensions=[
                Dimension(name="p", type="position", coords=[Position(name="p1")]),
                Dimension(name="p2", type="position", coords=[Position(name="p2")]),
                Dimension(name="y", count=64, type="space"),
                Dimension(name="x", count=64, type="space"),
            ],
            dtype="uint16",
        )


def test_too_few_dimensions_error() -> None:
    """Test that at least 2 dimensions are required."""
    with pytest.raises(
        ValueError, match="At least 2 non-position dimensions are required"
    ):
        AcquisitionSettings(
            root_path="test.zarr",
            dimensions=[
                Dimension(name="x", count=64, type="space"),
            ],
            dtype="uint16",
        )


def test_too_many_dimensions_error() -> None:
    """Test that at most 5 non-position dimensions are allowed."""
    with pytest.raises(
        ValueError, match="At most 5 non-position dimensions are allowed"
    ):
        AcquisitionSettings(
            root_path="test.zarr",
            dimensions=[
                Dimension(name="t", count=10),
                Dimension(name="c", count=3),
                Dimension(name="z", count=20),
                Dimension(name="a", count=2),
                Dimension(name="y", count=64, type="space"),
                Dimension(name="x", count=64, type="space"),
            ],
            dtype="uint16",
        )


def test_last_two_must_be_spatial() -> None:
    """Test that the last two dimensions must be spatial."""
    with pytest.raises(
        ValueError, match="The last two dimensions must be spatial dimensions"
    ):
        AcquisitionSettings(
            root_path="test.zarr",
            dimensions=[
                Dimension(name="y", count=64, type="space"),
                Dimension(name="x", count=64, type="space"),
                Dimension(name="t", count=10, type="time"),
            ],
            dtype="uint16",
        )


def test_non_xy_dimension_type_none_warning() -> None:
    """Test warning when non-x/y dimension has type=None."""
    # if the last two dimension are not x/y, and type is None, a warning is issued
    # and type is set to 'space'
    with pytest.warns(UserWarning, match="expected to have type='space'"):
        settings = AcquisitionSettings(
            root_path="test.zarr",
            dimensions=[
                Dimension(name="custom1", count=64),  # type=None, non-standard name
                Dimension(name="custom2", count=64),
            ],
            dtype="uint16",
        )
        assert [d.type for d in settings.dimensions] == ["space", "space"]


def test_tiff_backend_format(tiff_backend: str) -> None:
    """Test format property returns 'tiff' for tiff backend."""
    settings = AcquisitionSettings(
        root_path="test",
        dimensions=[
            Dimension(name="y", count=64, type="space"),
            Dimension(name="x", count=64, type="space"),
        ],
        dtype="uint16",
        format={"name": "ome-tiff", "backend": tiff_backend},
    )
    assert settings.format.name == "ome-tiff"

    settings2 = settings.model_copy(deep=True)
    settings2.format = "acquire-zarr"  # type: ignore
    assert settings2.format.name == "ome-zarr"


def test_acquisition_settings_shape() -> None:
    """Test shape property."""
    settings = AcquisitionSettings(
        root_path="test.zarr",
        dimensions=[
            Dimension(name="t", count=10, type="time"),
            Dimension(name="c", count=3, type="channel"),
            Dimension(name="y", count=64, type="space"),
            Dimension(name="x", count=128, type="space"),
        ],
        dtype="uint16",
    )
    assert settings.shape == (10, 3, 64, 128)


def test_num_frames_with_unbounded() -> None:
    """Test num_frames returns None when first dimension is unbounded."""
    settings = AcquisitionSettings(
        root_path="test.zarr",
        dimensions=[
            Dimension(name="t", count=None),
            Dimension(name="y", count=64, type="space"),
            Dimension(name="x", count=64, type="space"),
        ],
        dtype="uint16",
    )
    assert settings.num_frames is None


def test_array_dimensions() -> None:
    """Test array_dimensions excludes position dimension."""
    settings = AcquisitionSettings(
        root_path="test.zarr",
        dimensions=[
            Dimension(name="t", count=10, type="time"),
            Dimension(name="p", type="position", coords=[Position(name="pos1")]),
            Dimension(name="y", count=64, type="space"),
            Dimension(name="x", count=64, type="space"),
        ],
        dtype="uint16",
    )
    assert len(settings.array_dimensions) == 3
    assert all(isinstance(d, Dimension) for d in settings.array_dimensions)
    assert settings.storage_index_permutation is None  # already be sorted correctly
    assert len(settings.array_storage_dimensions) == 3


def test_dims_from_standard_axes_with_positions_list() -> None:
    """Test dims_from_standard_axes with position list."""
    dims = dims_from_standard_axes({"p": ["pos1", "pos2"], "y": 64, "x": 128})
    assert len(dims) == 3
    assert dims[0].type == "position"
    assert dims[0].coords is not None
    assert [p.name for p in dims[0].coords] == ["pos1", "pos2"]

    dims = dims_from_standard_axes({"p": 3, "y": 64, "x": 128})
    assert len(dims) == 3
    assert dims[0].type == "position"
    assert dims[0].count == 3


def test_storage_order_acquisition() -> None:
    """Test storage_order='acquisition' preserves acquisition order."""
    settings = AcquisitionSettings(
        root_path="test.zarr",
        dimensions=[
            Dimension(name="z", count=3, type="channel"),
            Dimension(name="c", count=3, type="channel"),
            Dimension(name="t", count=10, type="time"),
            Dimension(name="y", count=64, type="space"),
            Dimension(name="x", count=64, type="space"),
        ],
        dtype="uint16",
        storage_order="acquisition",
    )
    # Should preserve the order
    assert [d.name for d in settings.storage_index_dimensions] == ["z", "c", "t"]


def test_storage_order_ome_with_tiff(tiff_backend: str) -> None:
    """Test storage_order='ome' with tiff backend."""
    settings = AcquisitionSettings(
        root_path="test.ome.tiff",
        dimensions=[
            Dimension(name="c", count=3, type="channel"),
            Dimension(name="t", count=10, type="time"),
            Dimension(name="y", count=64, type="space"),
            Dimension(name="x", count=64, type="space"),
        ],
        dtype="uint16",
        storage_order="ome",
        format={"name": "ome-tiff", "backend": tiff_backend},
    )
    # For TIFF, the sort key should use _ome_tiff_sort_key
    assert settings.format.name == "ome-tiff"
    assert len(settings.storage_index_dimensions) == 2
    assert not settings.storage_index_permutation  # CTYX is already correct


def test_storage_order_valid_list() -> None:
    """Test storage_order with mismatched dimension names."""
    settings = AcquisitionSettings(
        root_path="test.zarr",
        dimensions=[
            Dimension(name="z", count=3, type="space"),
            Dimension(name="c", count=3, type="channel"),
            Dimension(name="t", count=10, type="time"),
            Dimension(name="y", count=64, type="space"),
            Dimension(name="x", count=64, type="space"),
        ],
        dtype="uint16",
        storage_order=["t", "c", "z", "y", "x"],  # 'z' doesn't exist in index dims
    )
    assert [d.name for d in settings.storage_index_dimensions] == ["t", "c", "z"]


def test_storage_order_invalid_list() -> None:
    """Test storage_order with mismatched dimension names."""
    with pytest.raises(ValueError, match=r"storage_order names .* don't match"):
        AcquisitionSettings(
            root_path="test.zarr",
            dimensions=[
                Dimension(name="t", count=10, type="time"),
                Dimension(name="c", count=3, type="channel"),
                Dimension(name="y", count=64, type="space"),
                Dimension(name="x", count=64, type="space"),
            ],
            dtype="uint16",
            storage_order=["z", "t", "y", "x"],  # 'z' doesn't exist in index dims
        )


def test_dimension_coords_unit_conflicts() -> None:
    """Test that conflicting coords and unit/type combinations are rejected."""

    # Channel coords with spatial unit (conflicting inference)
    with pytest.raises(
        ValueError,
        match=r"inference.*coords suggests.*channel.*unit suggests.*space",
    ):
        Dimension(name="test", coords=[Channel(name="red")], unit="micrometer")

    # Position coords with temporal unit (conflicting inference)
    with pytest.raises(
        ValueError,
        match=r"inference.*coords suggests.*position.*unit suggests.*time",
    ):
        Dimension(name="test", coords=[Position(name="A1")], unit="second")

    # Channel coords with explicit incompatible type
    with pytest.raises(
        ValueError,
        match=r"Channel objects in coords require type='channel'.*got type='time'",
    ):
        Dimension(name="test", coords=[Channel(name="red")], type="time")

    # Position coords with explicit incompatible type
    with pytest.raises(
        ValueError,
        match=r"Position objects in coords require type='position'.*got type='space'",
    ):
        Dimension(name="test", coords=[Position(name="A1")], type="space")


def test_dimension_coords_mixing_types() -> None:
    """Test that mixing incompatible types in coords is rejected."""

    # Mix Channel with non-str, non-Channel types
    with pytest.raises(ValueError, match=r"May not mix Channel objects"):
        Dimension(name="test", coords=[Channel(name="red"), 123])
    with pytest.raises(ValueError, match=r"May not mix Channel objects"):
        Dimension(name="test", coords=[Channel(name="red"), Position(name="A1")])

    # Mix Position with non-str, non-Position types
    with pytest.raises(ValueError, match=r"May not mix Position objects"):
        Dimension(name="test", coords=[Position(name="A1"), 456])

    # But mixing with strings is OK
    dim1 = Dimension(name="c", coords=[Channel(name="red"), "blue"])
    assert dim1.type == "channel"
    assert len(dim1.coords) == 2

    dim2 = Dimension(name="p", coords=[Position(name="A1"), "B2"])
    assert dim2.type == "position"
    assert len(dim2.coords) == 2


def test_position_dimension_deprecation() -> None:
    from ome_writers import PositionDimension

    with pytest.warns(DeprecationWarning, match="PositionDimension is deprecated"):
        pdims1 = PositionDimension(positions=["Pos0", "Pos1"])  # type: ignore

    pdims2 = Dimension(name="p", type="position", coords=["Pos0", "Pos1"])
    assert pdims1.model_dump() == pdims2.model_dump()

    pdims3 = Dimension(name="p", coords=[Position(name="Pos0"), Position(name="Pos1")])
    assert pdims1.model_dump() == pdims3.model_dump()
