"""Test that axis_order is properly respected in multi-position acquisitions.

These tests verify that frames are written to the correct array positions
when using different axis_order values in MDASequence. Each test creates
frames with unique statistical properties (mean, std) that encode the
exact indices (p, t, c), making it easy to verify correctness.
"""

from pathlib import Path

import numpy as np
import pytest

import ome_writers as omew


def create_unique_frame(
    p: int, t: int, c: int, shape: tuple[int, int] = (32, 32)
) -> np.ndarray:
    """Create a frame with a unique constant value based on indices.

    The frame is filled with: value = p * 10000 + t * 1000 + c * 100

    This makes it easy to verify that frames are stored in the correct
    position by checking their mean value.
    """
    value = p * 10000 + t * 1000 + c * 100
    return np.full(shape, value, dtype=np.uint16)


def verify_frame_value(
    array: np.ndarray, expected_p: int, expected_t: int, expected_c: int
) -> tuple[bool, str]:
    """Verify that frame has the expected constant value.

    Returns (is_correct, message).
    """
    mean = float(np.mean(array))
    expected_value = expected_p * 10000 + expected_t * 1000 + expected_c * 100

    # For constant-filled arrays, mean should be exactly the fill value
    # Allow small tolerance for floating point arithmetic
    is_correct = abs(mean - expected_value) < 0.1

    if is_correct:
        return True, f"Correct (p={expected_p}, t={expected_t}, c={expected_c})"
    else:
        return (
            False,
            f"Value mismatch: expected {expected_value} for "
            f"(p={expected_p}, t={expected_t}, c={expected_c}), got {mean:.0f}",
        )


def test_axis_order_tpc_without_plate(tmp_path: Path) -> None:
    """Test that axis_order='tpc' works correctly without HCS plates.

    This test ensures that when using axis_order='tpc' (time, position, channel),
    frames are written to the correct positions in the zarr array even though
    the acquisition order differs from the default position-first order.

    Acquisition order with tpc: P0T0C0, P1T0C0, P0T1C0, P1T1C0, P0T2C0, P1T2C0
    """
    pytest.importorskip("useq")
    from useq import MDASequence

    output_path = tmp_path / "test_tpc.zarr"

    # Create sequence with axis_order="tpc"
    # 2 positions, 3 timepoints, 2 channels = 12 frames
    seq = MDASequence(
        axis_order="tpc",
        stage_positions=[(0.0, 0.0), (1.0, 1.0)],
        time_plan={"interval": 0.1, "loops": 3},
        channels=["DAPI", "FITC"],
    )

    dims = omew.dims_from_useq(seq, image_width=32, image_height=32)
    stream = omew.create_stream(
        path=output_path,
        dimensions=dims,
        dtype=np.uint16,
        backend="acquire-zarr",
        overwrite=True,
    )

    # Write frames with unique statistics
    for event in seq:
        p = event.index.get("p", 0)
        t = event.index.get("t", 0)
        c = event.index.get("c", 0)
        frame = create_unique_frame(p, t, c)
        stream.append(frame)

    stream.flush()

    # Verify data
    import zarr

    zg = zarr.open_group(output_path, mode="r")

    # Check all positions, timepoints, and channels
    for p in range(2):
        pos_array = zg[str(p)]
        for t in range(3):
            for c in range(2):
                frame_data = pos_array[t, c, :, :]
                is_correct, msg = verify_frame_value(frame_data, p, t, c)
                assert is_correct, (
                    f"Position {p}, Time {t}, Channel {c}: {msg}"
                )


def test_axis_order_ptc_without_plate(tmp_path: Path) -> None:
    """Test that axis_order='ptc' works correctly without HCS plates.

    This test ensures that when using axis_order='ptc' (position, time, channel),
    frames are written to the correct positions. This is the more traditional
    acquisition order where all timepoints are acquired at a position before
    moving to the next position.

    Acquisition order with ptc: P0T0C0, P0T0C1, P0T1C0, P0T1C1, ..., P1T0C0, ...
    """
    pytest.importorskip("useq")
    from useq import MDASequence

    output_path = tmp_path / "test_ptc.zarr"

    # Create sequence with axis_order="ptc"
    # 2 positions, 3 timepoints, 2 channels = 12 frames
    seq = MDASequence(
        axis_order="ptc",
        stage_positions=[(0.0, 0.0), (1.0, 1.0)],
        time_plan={"interval": 0.1, "loops": 3},
        channels=["DAPI", "FITC"],
    )

    dims = omew.dims_from_useq(seq, image_width=32, image_height=32)
    stream = omew.create_stream(
        path=output_path,
        dimensions=dims,
        dtype=np.uint16,
        backend="acquire-zarr",
        overwrite=True,
    )

    # Write frames with unique statistics
    for event in seq:
        p = event.index.get("p", 0)
        t = event.index.get("t", 0)
        c = event.index.get("c", 0)
        frame = create_unique_frame(p, t, c)
        stream.append(frame)

    stream.flush()

    # Verify data
    import zarr

    zg = zarr.open_group(output_path, mode="r")

    # Check all positions, timepoints, and channels
    for p in range(2):
        pos_array = zg[str(p)]
        for t in range(3):
            for c in range(2):
                frame_data = pos_array[t, c, :, :]
                is_correct, msg = verify_frame_value(frame_data, p, t, c)
                assert is_correct, (
                    f"Position {p}, Time {p}, Channel {c}: {msg}"
                )


def test_axis_order_ctp_without_plate(tmp_path: Path) -> None:
    """Test that axis_order='ctp' works correctly without HCS plates.

    This tests a different ordering where channels are acquired first,
    then timepoints, then positions.

    Acquisition order with ctp: P0T0C0, P1T0C0, P0T1C0, P1T1C0, ..., P0T0C1, ...
    Dimension order: [c, t, p, y, x] (non-position: [c, t])
    """
    pytest.importorskip("useq")
    from useq import MDASequence

    output_path = tmp_path / "test_ctp.zarr"

    # Create sequence with axis_order="ctp"
    seq = MDASequence(
        axis_order="ctp",
        stage_positions=[(0.0, 0.0), (1.0, 1.0)],
        time_plan={"interval": 0.1, "loops": 3},
        channels=["DAPI", "FITC"],
    )

    dims = omew.dims_from_useq(seq, image_width=32, image_height=32)
    stream = omew.create_stream(
        path=output_path,
        dimensions=dims,
        dtype=np.uint16,
        backend="acquire-zarr",
        overwrite=True,
    )

    # Write frames with unique values
    for event in seq:
        p = event.index.get("p", 0)
        t = event.index.get("t", 0)
        c = event.index.get("c", 0)
        frame = create_unique_frame(p, t, c)
        stream.append(frame)

    stream.flush()

    # Verify data
    import zarr

    zg = zarr.open_group(output_path, mode="r")

    # For axis_order="ctp", dimensions are [c, t, p, y, x]
    # So non-position dims are [c, t], meaning array is indexed as [c, t, y, x]
    for p in range(2):
        pos_array = zg[str(p)]
        for c in range(2):
            for t in range(3):
                # Index as [c, t, :, :] not [t, c, :, :]
                frame_data = pos_array[c, t, :, :]
                is_correct, msg = verify_frame_value(frame_data, p, t, c)
                assert is_correct, (
                    f"Position {p}, Time {t}, Channel {c}: {msg}"
                )


def test_axis_order_ptc_with_hcs_plate(tmp_path: Path) -> None:
    """Test that axis_order='ptc' works correctly with HCS plates.

    This test verifies that when using HCS plates with multiple fields of view
    per well, the axis_order is properly respected and frames are written to
    the correct field of view locations.
    """
    pytest.importorskip("useq")
    from useq import GridRowsColumns, MDASequence, WellPlatePlan

    output_path = tmp_path / "test_ptc_plate.zarr"

    # Create a plate with 2 wells, 2 FOVs per well (4 positions total)
    plate_plan = WellPlatePlan(
        plate="96-well",
        a1_center_xy=(0.0, 0.0),
        selected_wells=([0, 1], [0, 0]),  # A1, B1
        well_points_plan=GridRowsColumns(rows=1, columns=2),  # 2 FOV per well
    )

    # axis_order="ptc": acquire all timepoints for each position
    seq = MDASequence(
        axis_order="ptc",
        stage_positions=plate_plan,
        time_plan={"interval": 0.1, "loops": 3},
        channels=["DAPI", "FITC"],
    )

    plate = omew.plate_from_useq(seq)
    dims = omew.dims_from_useq(seq, image_width=32, image_height=32)

    stream = omew.create_stream(
        path=output_path,
        dimensions=dims,
        dtype=np.uint16,
        backend="acquire-zarr",
        plate=plate,
        overwrite=True,
    )

    # Write frames with unique statistics
    for event in seq:
        p = event.index.get("p", 0)
        t = event.index.get("t", 0)
        c = event.index.get("c", 0)
        frame = create_unique_frame(p, t, c)
        stream.append(frame)

    stream.flush()

    # Verify data
    import zarr

    zg = zarr.open_group(output_path, mode="r")

    # Verify all FOVs in all wells
    # Well A/01: positions 0-1, Well B/01: positions 2-3
    well_fov_map = [
        ("96-well/A/01/fov0", 0),
        ("96-well/A/01/fov1", 1),
        ("96-well/B/01/fov0", 2),
        ("96-well/B/01/fov1", 3),
    ]

    for fov_path, p in well_fov_map:
        fov_array = zg[fov_path]
        for t in range(3):
            for c in range(2):
                frame_data = fov_array[t, c, :, :]
                is_correct, msg = verify_frame_value(frame_data, p, t, c)
                assert is_correct, f"{fov_path} (p={p}, t={t}, c={c}): {msg}"


def test_axis_order_tpc_with_hcs_plate(tmp_path: Path) -> None:
    """Test that axis_order='tpc' works correctly with HCS plates.

    This test uses a different axis order with HCS plates to ensure
    the fix works correctly in all scenarios.
    """
    pytest.importorskip("useq")
    from useq import GridRowsColumns, MDASequence, WellPlatePlan

    output_path = tmp_path / "test_tpc_plate.zarr"

    # Create a plate with 3 wells, 2 FOVs per well (6 positions total)
    plate_plan = WellPlatePlan(
        plate="96-well",
        a1_center_xy=(0.0, 0.0),
        selected_wells=([0, 0, 1], [0, 1, 0]),  # A1, A2, B1
        well_points_plan=GridRowsColumns(rows=1, columns=2),
    )

    # axis_order="tpc": acquire all positions at each timepoint
    seq = MDASequence(
        axis_order="tpc",
        stage_positions=plate_plan,
        time_plan={"interval": 0.1, "loops": 2},
        channels=["DAPI"],
    )

    plate = omew.plate_from_useq(seq)
    dims = omew.dims_from_useq(seq, image_width=32, image_height=32)

    stream = omew.create_stream(
        path=output_path,
        dimensions=dims,
        dtype=np.uint16,
        backend="acquire-zarr",
        plate=plate,
        overwrite=True,
    )

    # Write frames with unique statistics
    for event in seq:
        p = event.index.get("p", 0)
        t = event.index.get("t", 0)
        c = event.index.get("c", 0)
        frame = create_unique_frame(p, t, c)
        stream.append(frame)

    stream.flush()

    # Verify data
    import zarr

    zg = zarr.open_group(output_path, mode="r")

    # Verify all FOVs in all wells
    well_fov_map = [
        ("96-well/A/01/fov0", 0),
        ("96-well/A/01/fov1", 1),
        ("96-well/A/02/fov0", 2),
        ("96-well/A/02/fov1", 3),
        ("96-well/B/01/fov0", 4),
        ("96-well/B/01/fov1", 5),
    ]

    for fov_path, p in well_fov_map:
        fov_array = zg[fov_path]
        for t in range(2):
            for c in range(1):
                frame_data = fov_array[t, c, :, :]
                is_correct, msg = verify_frame_value(frame_data, p, t, c)
                assert is_correct, f"{fov_path} (p={p}, t={t}, c={c}): {msg}"

