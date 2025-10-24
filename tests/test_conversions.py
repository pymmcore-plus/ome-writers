"""Tests for Dimension and Plate conversion methods."""

from __future__ import annotations

import numpy as np
import pytest

import ome_writers as omew


def test_dimension_to_ome_types() -> None:
    """Test dims_to_ome() conversion."""
    pytest.importorskip("ome_types")
    from ome_types import model as ome

    # Create test dimensions
    dims = [
        omew.Dimension(label="t", size=3, unit=(1.0, "s")),
        omew.Dimension(label="c", size=2),
        omew.Dimension(label="z", size=5, unit=(0.5, "um")),
        omew.Dimension(label="y", size=64, unit=(0.1, "um")),
        omew.Dimension(label="x", size=64, unit=(0.1, "um")),
    ]

    # Convert to OME types
    ome_obj = omew.dims_to_ome(dims, dtype=np.uint16)

    # Validate the result
    assert isinstance(ome_obj, ome.OME)
    assert len(ome_obj.images) == 1
    assert ome_obj.images[0].pixels is not None

    pixels = ome_obj.images[0].pixels
    assert pixels.size_t == 3
    assert pixels.size_c == 2
    assert pixels.size_z == 5
    assert pixels.size_y == 64
    assert pixels.size_x == 64
    assert pixels.type.value == "uint16"

    # The OME object structure is valid if we can access all fields
    assert ome_obj.creator is not None


def test_dimension_to_ome_types_with_positions() -> None:
    """Test dims_to_ome() with position dimension."""
    pytest.importorskip("ome_types")
    from ome_types import model as ome

    dims = [
        omew.Dimension(label="t", size=2, unit=(1.0, "s")),
        omew.Dimension(label="c", size=1),
        omew.Dimension(label="y", size=32, unit=(0.2, "um")),
        omew.Dimension(label="x", size=32, unit=(0.2, "um")),
        omew.Dimension(label="p", size=3),
    ]

    ome_obj = omew.dims_to_ome(dims, dtype=np.uint8)

    assert isinstance(ome_obj, ome.OME)
    # Should create 3 images for 3 positions
    assert len(ome_obj.images) == 3

    for i, img in enumerate(ome_obj.images):
        assert img.id == f"Image:{i}"
        assert img.pixels is not None
        assert img.pixels.size_t == 2
        assert img.pixels.size_c == 1
        assert img.pixels.size_y == 32
        assert img.pixels.size_x == 32

    # The OME object structure is valid if we can access all fields
    assert ome_obj.creator is not None


def test_dimension_to_yaozarrs_v5() -> None:
    """Test dims_to_yaozarrs_v5() conversion."""
    pytest.importorskip("yaozarrs")
    import json

    from yaozarrs import validate_ome_json

    dims = [
        omew.Dimension(label="t", size=3, unit=(1.0, "s")),
        omew.Dimension(label="c", size=2),
        omew.Dimension(label="y", size=64, unit=(0.1, "um")),
        omew.Dimension(label="x", size=64, unit=(0.1, "um")),
    ]

    # Convert to yaozarrs v0.5 format
    array_dims = {"0": dims}
    image = omew.dims_to_yaozarrs_v5(array_dims)

    # Validate structure - image is a yaozarrs.v05.Image object
    assert image.version == "0.5"
    assert len(image.multiscales) == 1

    multiscale = image.multiscales[0]
    assert len(multiscale.axes) == 4
    assert len(multiscale.datasets) == 1
    assert multiscale.datasets[0].path == "0"

    # Check axes
    axes = multiscale.axes
    assert len(axes) == 4
    assert axes[0].name == "t"
    assert axes[0].type == "time"
    assert axes[1].name == "c"
    assert axes[1].type == "channel"
    assert axes[2].name == "y"
    assert axes[2].type == "space"
    assert axes[3].name == "x"
    assert axes[3].type == "space"

    # Validate using yaozarrs (convert Image object to dict first)
    zarr_meta = {"ome": image.model_dump(exclude_unset=True, by_alias=True)}
    validate_ome_json(json.dumps(zarr_meta))


def test_dimension_to_yaozarrs_v5_multiple_arrays() -> None:
    """Test dims_to_yaozarrs_v5() with multiple arrays."""
    pytest.importorskip("yaozarrs")
    import json

    from yaozarrs import validate_ome_json

    dims = [
        omew.Dimension(label="t", size=2, unit=(1.0, "s")),
        omew.Dimension(label="c", size=1),
        omew.Dimension(label="y", size=32, unit=(0.2, "um")),
        omew.Dimension(label="x", size=32, unit=(0.2, "um")),
    ]

    # Create multiple arrays with same dimensions (e.g., multi-position)
    array_dims = {"0": dims, "1": dims, "2": dims}
    image = omew.dims_to_yaozarrs_v5(array_dims)

    # Validate structure - image is a yaozarrs.v05.Image object
    assert image.version == "0.5"

    multiscale = image.multiscales[0]
    assert len(multiscale.datasets) == 3
    assert multiscale.datasets[0].path == "0"
    assert multiscale.datasets[1].path == "1"
    assert multiscale.datasets[2].path == "2"

    # Validate using yaozarrs (convert Image object to dict first)
    zarr_meta = {"ome": image.model_dump(exclude_unset=True, by_alias=True)}
    validate_ome_json(json.dumps(zarr_meta))


def test_plate_to_ome_types() -> None:
    """Test plate_to_ome_types() conversion."""
    pytest.importorskip("ome_types")
    from ome_types import model as ome

    # Create a 2x2 plate
    plate = omew.Plate(
        rows=["A", "B"],
        columns=["01", "02"],
        wells=[
            omew.WellPosition("A/01", 0, 0),
            omew.WellPosition("A/02", 0, 1),
            omew.WellPosition("B/01", 1, 0),
            omew.WellPosition("B/02", 1, 1),
        ],
        name="Test Plate",
        field_count=1,
        acquisitions=[
            omew.PlateAcquisition(
                id=0,
                name="Scan1",
                description="Test acquisition",
                start_time=1234567890,
                end_time=1234567900,
                maximum_field_count=1,
            )
        ],
    )

    # Convert to OME types
    ome_plate = omew.plate_to_ome_types(plate)

    # Validate the result
    assert isinstance(ome_plate, ome.Plate)
    assert ome_plate.name == "Test Plate"
    assert ome_plate.rows == 2
    assert ome_plate.columns == 2
    assert len(ome_plate.wells) == 4

    # Check wells
    for well in ome_plate.wells:
        assert isinstance(well, ome.Well)
        assert well.row in [0, 1]
        assert well.column in [0, 1]
        assert len(well.well_samples) == 1

    # Check acquisitions
    assert ome_plate.plate_acquisitions is not None
    assert len(ome_plate.plate_acquisitions) == 1
    acq = ome_plate.plate_acquisitions[0]
    assert acq.name == "Scan1"
    assert acq.description == "Test acquisition"
    assert acq.maximum_field_count == 1
    # Check that timestamps were converted from epoch to datetime
    assert acq.start_time is not None
    assert acq.end_time is not None

    # The plate object structure is valid if we can access all fields
    assert ome_plate.id is not None


def test_plate_to_yaozarrs_v5() -> None:
    """Test plate_to_yaozarrs_v5() conversion."""
    pytest.importorskip("yaozarrs")
    from yaozarrs import v05

    # Create a 2x3 plate
    plate = omew.Plate(
        rows=["A", "B"],
        columns=["01", "02", "03"],
        wells=[
            omew.WellPosition("A/01", 0, 0),
            omew.WellPosition("A/02", 0, 1),
            omew.WellPosition("A/03", 0, 2),
            omew.WellPosition("B/01", 1, 0),
            omew.WellPosition("B/02", 1, 1),
            omew.WellPosition("B/03", 1, 2),
        ],
        name="Test Plate",
        field_count=2,
        acquisitions=[
            omew.PlateAcquisition(
                id=0, name="Scan1", start_time=1234567890, maximum_field_count=2
            )
        ],
    )

    # Convert to yaozarrs v0.5
    yao_plate = omew.plate_to_yaozarrs_v5(plate)

    # Validate the result
    assert isinstance(yao_plate, v05.Plate)
    assert yao_plate.version == "0.5"
    assert yao_plate.plate is not None

    plate_def = yao_plate.plate
    assert plate_def.name == "Test Plate"
    assert plate_def.field_count == 2
    assert len(plate_def.rows) == 2
    assert len(plate_def.columns) == 3
    assert len(plate_def.wells) == 6

    # Check rows and columns
    assert plate_def.rows[0].name == "A"
    assert plate_def.rows[1].name == "B"
    assert plate_def.columns[0].name == "01"
    assert plate_def.columns[1].name == "02"
    assert plate_def.columns[2].name == "03"

    # Check wells
    for well in plate_def.wells:
        assert isinstance(well, v05.PlateWell)
        assert "/" in well.path
        assert 0 <= well.rowIndex <= 1
        assert 0 <= well.columnIndex <= 2

    # Check acquisitions
    assert plate_def.acquisitions is not None
    assert len(plate_def.acquisitions) == 1
    acq = plate_def.acquisitions[0]
    assert acq.id == 0
    assert acq.name == "Scan1"
    assert acq.starttime == 1234567890
    assert acq.maximumfieldcount == 2

    # Validate using yaozarrs validation
    # The yaozarrs Plate object has a model_validate method
    # We can also convert to dict and validate that way
    plate_dict = yao_plate.model_dump(exclude_unset=True, by_alias=True)
    assert "version" in plate_dict
    assert plate_dict["version"] == "0.5"
    assert "plate" in plate_dict


def test_plate_minimal() -> None:
    """Test Plate with minimal parameters."""
    pytest.importorskip("ome_types")
    pytest.importorskip("yaozarrs")

    # Create a minimal plate (no acquisitions, no name, no field_count)
    plate = omew.Plate(
        rows=["A"],
        columns=["01"],
        wells=[omew.WellPosition("A/01", 0, 0)],
    )

    # Test OME types conversion
    ome_plate = omew.plate_to_ome_types(plate)
    assert ome_plate.name is None
    assert ome_plate.rows == 1
    assert ome_plate.columns == 1
    assert len(ome_plate.wells) == 1
    assert (
        ome_plate.plate_acquisitions is None or len(ome_plate.plate_acquisitions) == 0
    )
    # The plate object structure is valid if we can access all fields
    assert ome_plate.id is not None

    # Test yaozarrs conversion
    yao_plate = omew.plate_to_yaozarrs_v5(plate)
    assert yao_plate.plate is not None
    assert yao_plate.plate.name is None
    assert yao_plate.plate.field_count is None
    assert (
        yao_plate.plate.acquisitions is None or len(yao_plate.plate.acquisitions) == 0
    )


def test_backward_compatibility_dims_to_ome() -> None:
    """Test that the old dims_to_ome function still works."""
    pytest.importorskip("ome_types")

    dims = [
        omew.Dimension(label="y", size=32, unit=(0.2, "um")),
        omew.Dimension(label="x", size=32, unit=(0.2, "um")),
    ]

    # Old function should still work
    ome_obj = omew.dims_to_ome(dims, dtype=np.uint16)
    assert ome_obj is not None
    assert len(ome_obj.images) == 1


def test_backward_compatibility_dims_to_yaozarrs_v5() -> None:
    """Test that the old dims_to_ngff_v5 function still works."""
    pytest.importorskip("yaozarrs")

    dims = [
        omew.Dimension(label="y", size=32, unit=(0.2, "um")),
        omew.Dimension(label="x", size=32, unit=(0.2, "um")),
    ]

    # Old function should still work and return a dict
    zarr_meta = omew.dims_to_ngff_v5({"0": dims})
    assert zarr_meta is not None
    assert "ome" in zarr_meta
