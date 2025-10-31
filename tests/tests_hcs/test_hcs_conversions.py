"""Tests for Dimension and Plate conversion methods."""

from __future__ import annotations

import numpy as np
import pytest

import ome_writers as omew


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


def test_plate_to_acquire_zarr() -> None:
    """Test plate_to_acquire_zarr() conversion."""
    pytest.importorskip("acquire_zarr")
    import acquire_zarr as aqz

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
        field_count=2,
        acquisitions=[
            omew.PlateAcquisition(
                id=0,
                name="Scan1",
                description="Test acquisition",
                start_time=1234567890,
                end_time=1234567900,
                maximum_field_count=2,
            )
        ],
    )

    # Create some dummy acquire-zarr dimensions
    az_dims = [
        aqz.Dimension(
            name="t",
            kind=aqz.DimensionType.TIME,
            array_size_px=3,
            chunk_size_px=3,
            shard_size_chunks=1,
        ),
        aqz.Dimension(
            name="c",
            kind=aqz.DimensionType.CHANNEL,
            array_size_px=2,
            chunk_size_px=2,
            shard_size_chunks=1,
        ),
        aqz.Dimension(
            name="y",
            kind=aqz.DimensionType.SPACE,
            array_size_px=64,
            chunk_size_px=64,
            shard_size_chunks=1,
        ),
        aqz.Dimension(
            name="x",
            kind=aqz.DimensionType.SPACE,
            array_size_px=64,
            chunk_size_px=64,
            shard_size_chunks=1,
        ),
    ]

    # Convert to acquire-zarr Plate
    aqz_plate = omew.plate_to_acquire_zarr(plate, az_dims, np.uint16)

    # Validate the result
    assert isinstance(aqz_plate, aqz.Plate)
    assert aqz_plate.name == "Test Plate"
    assert aqz_plate.path == "Test_Plate"  # spaces replaced with underscores
    assert aqz_plate.row_names == ["A", "B"]
    assert aqz_plate.column_names == ["01", "02"]
    assert len(aqz_plate.wells) == 4

    # Check wells
    for well in aqz_plate.wells:
        assert isinstance(well, aqz.Well)
        assert well.row_name in ["A", "B"]
        assert well.column_name in ["01", "02"]
        assert len(well.images) == 2  # field_count = 2

        # Check field of view entries
        for fov in well.images:
            assert isinstance(fov, aqz.FieldOfView)
            assert fov.path in ["fov0", "fov1"]
            assert fov.array_settings is not None
            assert fov.array_settings.data_type == aqz.DataType.UINT16
            assert len(fov.array_settings.dimensions) == 4

    # Check acquisitions
    assert aqz_plate.acquisitions is not None
    assert len(aqz_plate.acquisitions) == 1
    acq = aqz_plate.acquisitions[0]
    assert acq.id == 0
    assert acq.name == "Scan1"
    assert acq.start_time == 1234567890
    assert acq.end_time == 1234567900


def test_plate_from_useq_basic() -> None:
    """Test basic plate creation from useq.MDASequence."""
    pytest.importorskip("useq")
    from useq import MDASequence, WellPlatePlan

    # Create a simple plate plan with 3 wells
    plate_plan = WellPlatePlan(
        plate="96-well",
        a1_center_xy=(0.0, 0.0),
        selected_wells=([0, 0, 1], [0, 1, 0]),  # A1, A2, B1
    )
    seq = MDASequence(stage_positions=plate_plan)

    # Convert to ome-writers Plate
    plate = omew.plate_from_useq(seq)

    assert plate is not None
    assert plate.name == "96-well"
    assert len(plate.wells) == 3
    assert plate.field_count is None  # Only 1 FOV per well

    # Check that wells are correctly created
    well_paths = {w.path for w in plate.wells}
    assert "A/01" in well_paths
    assert "A/02" in well_paths
    assert "B/01" in well_paths


def test_plate_from_useq_multiple_fov_per_well() -> None:
    """Test plate with multiple fields of view per well."""
    pytest.importorskip("useq")
    from useq import GridRowsColumns, MDASequence, WellPlatePlan

    # Create a plate plan with 2x2 grid per well
    grid_plan = GridRowsColumns(rows=2, columns=2)
    plate_plan = WellPlatePlan(
        plate="24-well",
        a1_center_xy=(0.0, 0.0),
        selected_wells=([0, 1], [0, 0]),  # A1, B1
        well_points_plan=grid_plan,
    )
    seq = MDASequence(stage_positions=plate_plan)

    # Convert to ome-writers Plate
    plate = omew.plate_from_useq(seq)

    assert plate is not None
    assert plate.name == "24-well"
    assert len(plate.wells) == 2  # 2 unique wells
    assert plate.field_count == 4  # 2x2 grid = 4 FOV per well

    # Check wells
    well_paths = {w.path for w in plate.wells}
    assert "A/01" in well_paths
    assert "B/01" in well_paths


def test_plate_from_useq_with_time_plan() -> None:
    """Test plate with time acquisition."""
    pytest.importorskip("useq")
    from useq import MDASequence, TIntervalLoops, WellPlatePlan

    plate_plan = WellPlatePlan(
        plate="6-well",
        a1_center_xy=(0.0, 0.0),
        selected_wells=([0], [0]),  # A1
    )
    time_plan = TIntervalLoops(interval=1.0, loops=10)
    seq = MDASequence(stage_positions=plate_plan, time_plan=time_plan)

    plate = omew.plate_from_useq(seq)

    assert plate is not None
    assert plate.acquisitions is not None
    assert len(plate.acquisitions) == 1
    assert plate.acquisitions[0].name == "Acquisition 1"
    assert plate.acquisitions[0].maximum_field_count == 1


def test_plate_from_useq_different_plate_types() -> None:
    """Test different plate types."""
    pytest.importorskip("useq")
    from useq import MDASequence, WellPlatePlan

    for plate_type in ["6-well", "12-well", "24-well", "48-well", "96-well"]:
        plate_plan = WellPlatePlan(
            plate=plate_type,
            a1_center_xy=(0.0, 0.0),
            selected_wells=([0], [0]),  # Just A1
        )
        seq = MDASequence(stage_positions=plate_plan)
        plate = omew.plate_from_useq(seq)

        assert plate is not None
        assert plate.name == plate_type
        assert len(plate.wells) == 1


def test_plate_from_useq_non_plate_sequence() -> None:
    """Test that non-plate sequences return None."""
    pytest.importorskip("useq")
    from useq import MDASequence, Position

    # Sequence with simple positions (not a plate)
    seq = MDASequence(stage_positions=[Position(x=0, y=0), Position(x=1, y=1)])

    plate = omew.plate_from_useq(seq)
    assert plate is None

    # Sequence with no stage positions
    seq = MDASequence()
    plate = omew.plate_from_useq(seq)
    assert plate is None


def test_plate_from_useq_complex_well_selection() -> None:
    """Test complex well selection patterns."""
    pytest.importorskip("useq")
    from useq import MDASequence, WellPlatePlan

    # Select every other well in first two rows
    plate_plan = WellPlatePlan(
        plate="96-well",
        a1_center_xy=(0.0, 0.0),
        selected_wells=(
            [0, 0, 0, 1, 1, 1],  # rows: A, A, A, B, B, B
            [0, 2, 4, 1, 3, 5],  # cols: 1, 3, 5, 2, 4, 6
        ),
    )
    seq = MDASequence(stage_positions=plate_plan)
    plate = omew.plate_from_useq(seq)

    assert plate is not None
    assert len(plate.wells) == 6

    # Check specific wells
    well_paths = {w.path for w in plate.wells}
    assert "A/01" in well_paths
    assert "A/03" in well_paths
    assert "A/05" in well_paths
    assert "B/02" in well_paths
    assert "B/04" in well_paths
    assert "B/06" in well_paths


def test_plate_from_useq_integration_with_plate_to_yaozarrs() -> None:
    """Test integration with plate_to_yaozarrs_v5."""
    pytest.importorskip("useq")
    pytest.importorskip("yaozarrs")
    from useq import GridRowsColumns, MDASequence, WellPlatePlan
    from yaozarrs import v05

    # Create a plate with multiple FOV per well
    grid_plan = GridRowsColumns(rows=2, columns=2)
    plate_plan = WellPlatePlan(
        plate="96-well",
        a1_center_xy=(0.0, 0.0),
        selected_wells=([0, 0], [0, 1]),  # A1, A2
        well_points_plan=grid_plan,
    )
    seq = MDASequence(stage_positions=plate_plan)

    # Convert to ome-writers Plate
    plate = omew.plate_from_useq(seq)
    assert plate is not None

    # Convert to yaozarrs format
    yao_plate = omew.plate_to_yaozarrs_v5(plate)

    assert isinstance(yao_plate, v05.Plate)
    assert yao_plate.version == "0.5"
    assert yao_plate.plate is not None
    assert yao_plate.plate.name == "96-well"
    assert yao_plate.plate.field_count == 4  # 2x2 grid
    assert len(yao_plate.plate.wells) == 2


def test_plate_from_useq_large_plate_with_multi_letter_rows() -> None:
    """Test 1536-well plate with multi-letter row names (A-Z, AA-AF)."""
    pytest.importorskip("useq")
    from useq import (
        MDASequence,
        WellPlatePlan,
        register_well_plates,
        registered_well_plate_keys,
    )

    register_well_plates(
        {
            "1536-well": {
                "rows": 32,
                "columns": 48,
                "well_spacing": 2.25,
                "well_size": 1.55,
            }
        }
    )

    assert "1536-well" in registered_well_plate_keys()

    # Create a 1536-well plate plan with wells that have multi-letter row names
    # 1536-well plates have 32 rows (A-Z, AA-AF) and 48 columns
    plate_plan = WellPlatePlan(
        plate="1536-well",
        a1_center_xy=(0.0, 0.0),
        selected_wells=([0, 25, 30], [0, 23, 47]),  # A1, Z24, AE48
    )
    seq = MDASequence(stage_positions=plate_plan)

    # Convert to ome-writers Plate
    plate = omew.plate_from_useq(seq)

    assert plate is not None
    assert plate.name == "1536-well"
    assert len(plate.wells) == 3
    assert len(plate.rows) == 32  # A-Z (26) + AA-AF (6) = 32
    assert len(plate.columns) == 48

    # Check that row labels are correct
    assert plate.rows[0] == "A"
    assert plate.rows[25] == "Z"
    assert plate.rows[26] == "AA"
    assert plate.rows[30] == "AE"
    assert plate.rows[31] == "AF"

    # Check that wells are correctly created
    well_paths = {w.path for w in plate.wells}
    assert "A/01" in well_paths  # First well (row 0, col 0)
    assert "Z/24" in well_paths  # Row Z (index 25), col 24
    assert "AE/48" in well_paths  # Row AE (index 30), col 48
