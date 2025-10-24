"""Tests for plate_from_useq function."""

from __future__ import annotations

import pytest

import ome_writers as omew


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
