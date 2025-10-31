"""Tests for HCS (High Content Screening) support with acquire-zarr backend."""

from __future__ import annotations

import shutil
from typing import TYPE_CHECKING

import numpy as np
import pytest

from ome_writers import Dimension, Plate, PlateAcquisition, WellPosition, create_stream

if TYPE_CHECKING:
    from pathlib import Path

pytest.importorskip("acquire_zarr", reason="acquire-zarr backend not available")
pytest.importorskip("useq", reason="useq not available")
pytest.importorskip("yaozarrs", reason="yaozarrs not available")


def test_hcs_plate_with_acquire_zarr(tmp_path: Path) -> None:
    """Test creating HCS plate structure with acquire-zarr backend."""
    # Create a simple plate with 3 wells
    plate = Plate(
        rows=["A", "B"],
        columns=["01", "02", "03"],
        wells=[
            WellPosition("A/01", 0, 0),
            WellPosition("A/02", 0, 1),
            WellPosition("B/01", 1, 0),
        ],
        name="Test Plate",
        field_count=1,
        acquisitions=[PlateAcquisition(id=0, name="Scan1")],
    )

    # Create dimensions for the data (position dimension for wells)
    dimensions = [
        Dimension(label="p", size=3),  # 3 wells × 1 field per well
        Dimension(label="t", size=2, unit=(1.0, "s")),
        Dimension(label="c", size=2),
        Dimension(label="z", size=3, unit=(1, "um")),
        Dimension(label="y", size=64, unit=(0.1, "um")),
        Dimension(label="x", size=64, unit=(0.1, "um")),
    ]

    output_path = tmp_path / "test_plate.zarr"

    # Create stream with plate metadata
    with create_stream(
        path=str(output_path),
        dimensions=dimensions,
        dtype=np.uint16,
        backend="acquire-zarr",
        plate=plate,
        overwrite=True,
    ) as stream:
        # Write data for each well (3 wells × 2 time × 2 channels × 3 z-slices)
        for _ in range(3 * 2 * 2 * 3):
            frame = np.random.randint(0, 100, (64, 64), dtype=np.uint16)
            stream.append(frame)

    # Verify the structure was created
    assert output_path.exists()

    # Check that plate metadata was written (plate name is sanitized)
    plate_path = output_path / "Test_Plate"
    assert plate_path.exists(), "Plate directory should exist"

    # Check wells exist
    well_a01 = plate_path / "A" / "01"
    well_a02 = plate_path / "A" / "02"
    well_b01 = plate_path / "B" / "01"

    assert well_a01.exists(), "Well A/01 should exist"
    assert well_a02.exists(), "Well A/02 should exist"
    assert well_b01.exists(), "Well B/01 should exist"

    # Check field of view exists (single field per well)
    assert (well_a01 / "0").exists(), "Field 0 in well A/01 should exist"
    assert (well_a02 / "0").exists(), "Field 0 in well A/02 should exist"
    assert (well_b01 / "0").exists(), "Field 0 in well B/01 should exist"

    # Cleanup
    shutil.rmtree(output_path)


def test_hcs_plate_with_multiple_fields_per_well(tmp_path: Path) -> None:
    """Test HCS plate with multiple fields of view per well."""
    # Create plate with 2 fields per well
    plate = Plate(
        rows=["A"],
        columns=["01", "02"],
        wells=[
            WellPosition("A/01", 0, 0),
            WellPosition("A/02", 0, 1),
        ],
        name="MultiField Plate",
        field_count=2,  # 2 fields per well
        acquisitions=[PlateAcquisition(id=0, name="Acquisition 1")],
    )

    # Dimensions: position will be expanded to 2 wells × 2 fields = 4 positions
    dimensions = [
        Dimension(label="p", size=4),  # 2 wells × 2 fields
        Dimension(label="t", size=2, unit=(1.0, "s")),
        Dimension(label="c", size=1),
        Dimension(label="y", size=32, unit=(0.2, "um")),
        Dimension(label="x", size=32, unit=(0.2, "um")),
    ]

    output_path = tmp_path / "multifield_plate.zarr"

    with create_stream(
        path=str(output_path),
        dimensions=dimensions,
        dtype=np.uint16,
        backend="acquire-zarr",
        plate=plate,
        overwrite=True,
    ) as stream:
        # Write data: 4 positions × 2 time × 1 channel = 8 frames
        for _ in range(8):
            frame = np.random.randint(0, 100, (32, 32), dtype=np.uint16)
            stream.append(frame)

    # Verify structure
    assert output_path.exists()
    plate_path = output_path / "MultiField_Plate"  # Sanitized name
    assert plate_path.exists()

    # Check that both fields exist for each well
    well_a01 = plate_path / "A" / "01"
    well_a02 = plate_path / "A" / "02"

    assert (well_a01 / "fov0").exists(), "Field fov0 in well A/01 should exist"
    assert (well_a01 / "fov1").exists(), "Field fov1 in well A/01 should exist"
    assert (well_a02 / "fov0").exists(), "Field fov0 in well A/02 should exist"
    assert (well_a02 / "fov1").exists(), "Field fov1 in well A/02 should exist"

    # Cleanup
    shutil.rmtree(output_path)


def test_hcs_plate_from_useq(tmp_path: Path) -> None:
    """Test creating HCS plate from useq.MDASequence using plate_from_useq."""
    from useq import GridRowsColumns, MDASequence, WellPlatePlan

    from ome_writers import plate_from_useq

    # Create a plate plan with multiple FOV per well
    plate_plan = WellPlatePlan(
        plate="96-well",
        a1_center_xy=(0.0, 0.0),
        selected_wells=([0, 0, 1], [0, 1, 0]),  # A1, A2, B1
        well_points_plan=GridRowsColumns(rows=2, columns=2),  # 4 FOV per well
    )

    seq = MDASequence(
        stage_positions=plate_plan,
        time_plan={"interval": 0.1, "loops": 2},
        channels=["DAPI", "GFP"],
    )

    # Extract plate metadata from sequence
    plate = plate_from_useq(seq)
    assert plate is not None
    assert len(plate.wells) == 3
    assert plate.field_count == 4  # 2×2 grid

    # Create dimensions from sequence
    from ome_writers import dims_from_useq

    dims = dims_from_useq(seq, image_width=128, image_height=128)

    output_path = tmp_path / "useq_plate.zarr"

    # Create stream with plate from useq
    with create_stream(
        path=str(output_path),
        dimensions=dims,
        dtype=np.uint16,
        backend="acquire-zarr",
        plate=plate,
        overwrite=True,
    ) as stream:
        # Calculate total frames: 3 wells × 4 fields × 2 time × 2 channels
        total_frames = 3 * 4 * 2 * 2
        for _ in range(total_frames):
            frame = np.random.randint(0, 200, (128, 128), dtype=np.uint16)
            stream.append(frame)

    # Verify the plate structure
    assert output_path.exists()
    plate_path = output_path / plate.name
    assert plate_path.exists()

    # Verify wells exist
    well_a01 = plate_path / "A" / "01"
    well_a02 = plate_path / "A" / "02"
    well_b01 = plate_path / "B" / "01"

    assert well_a01.exists()
    assert well_a02.exists()
    assert well_b01.exists()

    # Verify all 4 fields exist in each well
    for well in [well_a01, well_a02, well_b01]:
        for fov_idx in range(4):
            fov_path = well / f"fov{fov_idx}"
            assert fov_path.exists(), f"Field fov{fov_idx} should exist in {well}"

    # Cleanup
    shutil.rmtree(output_path)


def test_hcs_large_plate_from_useq(tmp_path: Path) -> None:
    """Test HCS with large plate (1536-well) using multi-letter row names."""
    from useq import GridRowsColumns, MDASequence, WellPlatePlan, register_well_plates

    from ome_writers import dims_from_useq, plate_from_useq

    # Register 1536-well plate
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

    # Create plate plan with wells that have multi-letter row names
    plate_plan = WellPlatePlan(
        plate="1536-well",
        a1_center_xy=(0.0, 0.0),
        selected_wells=([30, 28, 1], [32, 10, 0]),  # AE33, AC11, B01
        well_points_plan=GridRowsColumns(rows=1, columns=2),  # 2 FOV per well
    )

    seq = MDASequence(
        stage_positions=plate_plan,
        channels=["DAPI"],
    )

    # Extract plate metadata
    plate = plate_from_useq(seq)
    assert plate is not None
    assert len(plate.wells) == 3
    assert plate.field_count == 2

    # Verify multi-letter row names are handled
    well_paths = {w.path for w in plate.wells}
    assert "AE/33" in well_paths, "Should have well AE/33"
    assert "AC/11" in well_paths, "Should have well AC/11"
    assert "B/01" in well_paths, "Should have well B/01"

    # Create dimensions
    dims = dims_from_useq(seq, image_width=64, image_height=64)

    output_path = tmp_path / "large_plate.zarr"

    # Create stream
    with create_stream(
        path=str(output_path),
        dimensions=dims,
        dtype=np.uint16,
        backend="acquire-zarr",
        plate=plate,
        overwrite=True,
    ) as stream:
        # 3 wells × 2 fields × 1 channel = 6 frames
        for _ in range(6):
            frame = np.random.randint(0, 100, (64, 64), dtype=np.uint16)
            stream.append(frame)

    # Verify structure
    assert output_path.exists()
    plate_path = output_path / plate.name
    assert plate_path.exists()

    # Check multi-letter row wells exist
    well_ae33 = plate_path / "AE" / "33"
    well_ac11 = plate_path / "AC" / "11"
    well_b01 = plate_path / "B" / "01"

    assert well_ae33.exists(), "Well AE/33 should exist"
    assert well_ac11.exists(), "Well AC/11 should exist"
    assert well_b01.exists(), "Well B/01 should exist"

    # Verify fields exist
    for well in [well_ae33, well_ac11, well_b01]:
        assert (well / "fov0").exists(), f"Field fov0 should exist in {well}"
        assert (well / "fov1").exists(), f"Field fov1 should exist in {well}"

    # Cleanup
    shutil.rmtree(output_path)


def test_hcs_metadata_validation(tmp_path: Path) -> None:
    """Test that HCS plate metadata is properly validated using yaozarrs."""
    from useq import GridRowsColumns, MDASequence, WellPlatePlan
    from yaozarrs import validate_ome_json

    from ome_writers import dims_from_useq, plate_from_useq

    # Create plate plan
    plate_plan = WellPlatePlan(
        plate="96-well",
        a1_center_xy=(0.0, 0.0),
        selected_wells=([0, 1], [0, 0]),  # A1, B1
        well_points_plan=GridRowsColumns(rows=1, columns=1),  # Single FOV
    )

    seq = MDASequence(
        stage_positions=plate_plan,
        channels=["DAPI"],
    )

    plate = plate_from_useq(seq)
    assert plate is not None

    dims = dims_from_useq(seq, image_width=32, image_height=32)

    output_path = tmp_path / "validation_plate.zarr"

    with create_stream(
        path=str(output_path),
        dimensions=dims,
        dtype=np.uint16,
        backend="acquire-zarr",
        plate=plate,
        overwrite=True,
    ) as stream:
        # 2 wells x 1 field x 1 channel = 2 frames
        for _ in range(2):
            frame = np.random.randint(0, 100, (32, 32), dtype=np.uint16)
            stream.append(frame)

    # Validate the Zarr store structure using yaozarrs
    import json

    # Check that plate metadata exists at root
    zarr_json_path = output_path / "zarr.json"
    assert zarr_json_path.exists(), "zarr.json should exist at root"

    with open(zarr_json_path) as f:
        root_meta = json.load(f)

    # Verify it's a group
    assert root_meta.get("node_type") == "group", "Root should be a group"

    # For plate structures, the OME metadata is at the plate level
    # Check that wells exist
    plate_name_sanitized = plate.name.replace(" ", "_")
    plate_dir = output_path / plate_name_sanitized
    assert plate_dir.exists(), f"Plate directory {plate_name_sanitized} should exist"

    # Validate plate-level OME-NGFF metadata using yaozarrs
    plate_zarr_json = plate_dir / "zarr.json"
    assert plate_zarr_json.exists(), "zarr.json should exist at plate level"

    with open(plate_zarr_json) as f:
        plate_meta = json.load(f)

    # The plate metadata should have OME metadata with plate information
    assert "ome" in plate_meta.get("attributes", {}), (
        "Plate metadata should have OME attributes"
    )
    assert "plate" in plate_meta["attributes"]["ome"], (
        "OME metadata should have plate information"
    )

    # Validate using yaozarrs
    validate_ome_json(json.dumps(plate_meta))

    # Verify wells exist for basic structure check
    for well_pos in plate.wells:
        row, col = well_pos.path.split("/")
        well_dir = plate_dir / row / col / "0"  # Single FOV

        assert well_dir.exists(), f"Well {well_pos.path}/0 should exist"

    # Cleanup
    shutil.rmtree(output_path)
