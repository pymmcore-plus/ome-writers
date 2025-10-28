"""Test that axis_order is properly respected in multi-position acquisitions.

These tests verify that frames are written to the correct array positions
when using different axis_order values in MDASequence. Each test creates
frames with unique statistical properties (mean, std) that encode the
exact indices (p, t, c), making it easy to verify correctness.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pytest

import ome_writers as omew

if TYPE_CHECKING:
    from .conftest import AvailableBackend

pytest.importorskip("useq")


def create_unique_frame(
    p: int, t: int, c: int, z: int, shape: tuple[int, int] = (32, 32)
) -> np.ndarray:
    """Create a frame with a unique constant value based on indices.

    The frame is filled with: value = p * 10000 + t * 1000 + c * 100 + z * 10

    This makes it easy to verify that frames are stored in the correct
    position by checking their mean value.
    """
    value = p * 10000 + t * 1000 + c * 100 + z * 10
    return np.full(shape, value, dtype=np.uint16)


def verify_frame_value(
    array: np.ndarray,
    expected_p: int,
    expected_t: int,
    expected_c: int,
    expected_z: int,
) -> tuple[bool, str]:
    """Verify that frame has the expected constant value.

    Returns (is_correct, message).
    """
    mean = float(np.mean(array))
    expected_value = (
        expected_p * 10000 + expected_t * 1000 + expected_c * 100 + expected_z * 10
    )

    # For constant-filled arrays, mean should be exactly the fill value
    # Allow small tolerance for floating point arithmetic
    is_correct = abs(mean - expected_value) < 0.1

    if is_correct:
        msg = (
            f"Correct (p={expected_p}, t={expected_t}, c={expected_c}, z={expected_z})"
        )
        return True, msg
    else:
        return (
            False,
            f"Value mismatch: expected {expected_value} for "
            f"(p={expected_p}, t={expected_t}, c={expected_c}, z={expected_z}), "
            f"got {mean:.0f}",
        )


@pytest.mark.parametrize("tiff_mmmap", [True, False])
@pytest.mark.parametrize("axis_order", ["tpzc", "ptzc", "cztp", "tpcz", "ptcz"])
def test_axis_order(
    axis_order: str, backend: AvailableBackend, tiff_mmmap: bool, tmp_path: Path
) -> None:
    """Test that different axis_order values work correctly.

    This test ensures that frames are written to the correct positions
    regardless of the acquisition order specified by axis_order.

    Parameters
    ----------
    axis_order : str
        The acquisition order (e.g., 'tpzc', 'ptzc', 'cztp').
    backend : AvailableBackend
        The backend to use for testing.
    tmp_path : Path
        Temporary directory for test output.
    """
    from useq import MDASequence

    # Create output path - for TIFF backend, we need to explicitly add .ome.tiff
    # because the backend strips and re-adds extensions
    if backend.file_ext.endswith("tiff"):
        output_path = tmp_path / f"test_{axis_order}.ome.tiff"
    else:
        if tiff_mmmap:
            pytest.skip("Memory-mapped option only applies to TIFF backend.")
        output_path = tmp_path / f"test_{axis_order}.{backend.file_ext}"

    # Create sequence with specified axis_order
    # 2 positions, 3 timepoints, 2 channels, 4 z-slices = 48 frames
    seq = MDASequence(
        axis_order=axis_order,
        stage_positions=[(0.0, 0.0), (1.0, 1.0)],
        time_plan={"interval": 0.1, "loops": 3},
        channels=["DAPI", "FITC"],
        z_plan={"range": 3.0, "step": 1.0},
    )

    dims = omew.dims_from_useq(seq, image_width=32, image_height=32)

    if backend.file_ext.endswith("tiff"):
        # For TIFF, enable memory mapping option
        stream = omew.TifffileStream(use_memmap=tiff_mmmap)
        stream.create(
            path=output_path,
            dimensions=dims,
            dtype=np.uint16,
            overwrite=True,
        )
    else:
        stream = omew.create_stream(
            path=output_path,
            dimensions=dims,
            dtype=np.uint16,
            backend=backend.name,
            overwrite=True,
        )

    # Write frames with unique statistics
    for event in seq:
        p = event.index.get("p", 0)
        t = event.index.get("t", 0)
        c = event.index.get("c", 0)
        z = event.index.get("z", 0)
        frame = create_unique_frame(p, t, c, z)
        stream.append(frame)

    stream.flush()

    # Verify data based on backend type
    if backend.file_ext.endswith("zarr"):
        pytest.importorskip("zarr", reason="zarr not installed")
        import zarr

        zg = zarr.open_group(output_path, mode="r")

        # Determine the expected array indexing based on axis_order
        # The dims are created in axis_order, so we need to know the non-position order
        non_pos_dims = [d.label for d in dims if d.label not in "pyx"]

        # Check all positions, timepoints, channels, and z-slices
        for p in range(2):
            pos_array = zg[str(p)]
            for t in range(3):
                for c in range(2):
                    for z in range(4):
                        # Build index dict for all dimensions
                        indices = {"t": t, "c": c, "z": z}
                        idx_tuple = tuple(indices[d] for d in non_pos_dims)
                        frame_data = pos_array[(*idx_tuple, slice(None), slice(None))]

                        is_correct, msg = verify_frame_value(frame_data, p, t, c, z)
                        assert is_correct, (
                            f"Position {p}, Time {t}, Channel {c}, Z {z}: {msg}"
                        )

    elif backend.file_ext.endswith("tiff"):
        pytest.importorskip("tifffile", reason="tifffile not installed")
        import tifffile

        # For multi-position TIFF, the backend strips .ome.tiff extension,
        # adds _p{idx:03d}, then adds .ome.tiff back
        # So test_tpc.ome.tiff becomes test_tpc_p000.ome.tiff
        base_path = str(output_path)
        for possible_ext in [".ome.tiff", ".ome.tif", ".tiff", ".tif"]:
            if base_path.endswith(possible_ext):
                ext = possible_ext
                base_name = base_path[: -len(possible_ext)]
                break
        else:
            ext = ""
            base_name = base_path

        # Determine the expected storage order from the dims
        # (TIFF can store in any order matching the dimension order)
        non_pos_dims = [d.label for d in dims if d.label not in "pyx"]

        for p in range(2):
            tiff_path = tmp_path / f"{Path(base_name).name}_p{p:03d}{ext}"

            with tifffile.TiffFile(tiff_path) as tif:
                data = tif.asarray()
                # TIFF data is stored in the order matching non_pos_dims
                for t in range(3):
                    for c in range(2):
                        for z in range(4):
                            # Build index dict for all dimensions
                            indices = {"t": t, "c": c, "z": z}
                            idx_tuple = tuple(indices[d] for d in non_pos_dims)
                            frame_data = data[(*idx_tuple, slice(None), slice(None))]
                            is_correct, msg = verify_frame_value(frame_data, p, t, c, z)
                            assert is_correct, (
                                f"Position {p}, Time {t}, Channel {c}, Z {z}: {msg}"
                            )


@pytest.mark.parametrize("axis_order", ["tpc", "ptc", "ctp"])
def test_axis_order_with_plate(axis_order: str, tmp_path: Path) -> None:
    """Test that axis_order works correctly with HCS plate acquisitions.

    This test verifies that frames are written to the correct positions
    in a plate structure when using different axis_order values.

    Parameters
    ----------
    axis_order : str
        The acquisition order (e.g., 'tpc', 'ptc', 'ctp').
    tmp_path : Path
        Temporary directory for test output.
    """
    pytest.importorskip("acquire_zarr", reason="acquire-zarr not installed")
    from useq import GridRowsColumns, MDASequence, WellPlatePlan

    # Create a simple plate plan: 2 wells, 2 fields per well
    plate_plan = WellPlatePlan(
        plate="96-well",
        a1_center_xy=(0.0, 0.0),
        selected_wells=([0, 1], [0, 0]),  # A1, B1
        well_points_plan=GridRowsColumns(rows=1, columns=2),  # 2 FOV per well
    )

    # Create sequence with specified axis_order
    # 2 wells x 2 fields x 2 timepoints x 2 channels = 16 frames
    seq = MDASequence(
        axis_order=axis_order,
        stage_positions=plate_plan,
        time_plan={"interval": 0.1, "loops": 2},
        channels=["DAPI", "FITC"],
    )

    output_path = tmp_path / f"plate_{axis_order}.ome.zarr"

    # Get plate and dimensions from sequence
    plate = omew.plate_from_useq(seq)
    assert plate is not None
    assert len(plate.wells) == 2
    assert plate.field_count == 2

    dims = omew.dims_from_useq(seq, image_width=32, image_height=32)

    # Create stream with plate
    stream = omew.create_stream(
        path=str(output_path),
        dimensions=dims,
        dtype=np.uint16,
        backend="acquire-zarr",
        plate=plate,
        overwrite=True,
    )

    # Write frames with unique values
    for event in seq:
        p = event.index.get("p", 0)
        t = event.index.get("t", 0)
        c = event.index.get("c", 0)
        z = event.index.get("z", 0)
        frame = create_unique_frame(p, t, c, z)
        stream.append(frame)

    stream.flush()

    # Verify data in plate structure
    import zarr

    # The plate structure should have wells A/01 and B/01, each with 2 fields
    plate_name_sanitized = plate.name.replace(" ", "_")
    plate_dir = output_path / plate_name_sanitized

    # Verify wells exist
    well_a01 = plate_dir / "A" / "01"
    well_b01 = plate_dir / "B" / "01"
    assert well_a01.exists(), "Well A/01 should exist"
    assert well_b01.exists(), "Well B/01 should exist"

    # Map global position index to well and field
    # p=0: Well A/01, field 0
    # p=1: Well A/01, field 1
    # p=2: Well B/01, field 0
    # p=3: Well B/01, field 1
    position_map = [
        (well_a01, 0),  # p=0
        (well_a01, 1),  # p=1
        (well_b01, 0),  # p=2
        (well_b01, 1),  # p=3
    ]

    # Determine the expected array indexing based on axis_order
    non_pos_dims = [d.label for d in dims if d.label not in "pyx"]

    # Check all positions, timepoints, and channels
    for p in range(4):
        well_path, field_idx = position_map[p]
        field_name = f"fov{field_idx}" if plate.field_count > 1 else "0"
        field_path = well_path / field_name

        assert field_path.exists(), f"Field {field_name} should exist at {well_path}"

        # Open the field array
        field_array = zarr.open_array(field_path, mode="r")

        for t in range(2):
            for c in range(2):
                # Build index tuple based on non-position dimensions
                indices = {"t": t, "c": c, "z": 0}
                idx_tuple = tuple(indices.get(d, 0) for d in non_pos_dims)

                frame_data = field_array[(*idx_tuple, slice(None), slice(None))]

                is_correct, msg = verify_frame_value(frame_data, p, t, c, 0)
                assert is_correct, (
                    f"Well {well_path.name}, Field {field_idx}, "
                    f"Time {t}, Channel {c}: {msg}"
                )
