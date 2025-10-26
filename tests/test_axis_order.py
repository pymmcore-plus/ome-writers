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

pytest.importorskip("useq")
pytest.importorskip("zarr")


def create_unique_frame(
    p: int, t: int, c: int, shape: tuple[int, int] = (32, 32)
) -> np.ndarray:
    """Create a frame with a unique constant value based on indices.

    The frame is filled with: value = p * 1000 + t * 100 + c * 10

    This makes it easy to verify that frames are stored in the correct
    position by checking their mean value.
    """
    value = p * 1000 + t * 100 + c * 10
    return np.full(shape, value, dtype=np.uint16)


def verify_frame_value(
    array: np.ndarray, expected_p: int, expected_t: int, expected_c: int
) -> tuple[bool, str]:
    """Verify that frame has the expected constant value.

    Returns (is_correct, message).
    """
    mean = float(np.mean(array))
    expected_value = expected_p * 1000 + expected_t * 100 + expected_c * 10

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


def test_axis_order_tpc(tmp_path: Path) -> None:
    """Test that axis_order='tpc' works correctly.

    This test ensures that when using axis_order='tpc' (time, position, channel),
    frames are written to the correct positions in the zarr array even though
    the acquisition order differs from the default position-first order.

    Acquisition order with tpc: P0T0C0, P1T0C0, P0T1C0, P1T1C0, P0T2C0, P1T2C0
    """
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

    import zarr

    zg = zarr.open_group(output_path, mode="r")

    # Check all positions, timepoints, and channels
    for p in range(2):
        pos_array = zg[str(p)]
        for t in range(3):
            for c in range(2):
                frame_data = pos_array[t, c, :, :]
                is_correct, msg = verify_frame_value(frame_data, p, t, c)
                assert is_correct, f"Position {p}, Time {t}, Channel {c}: {msg}"


def test_axis_order_ptc(tmp_path: Path) -> None:
    """Test that axis_order='ptc' works correctly.

    This test ensures that when using axis_order='ptc' (position, time, channel),
    frames are written to the correct positions. This is the more traditional
    acquisition order where all timepoints are acquired at a position before
    moving to the next position.

    Acquisition order with ptc: P0T0C0, P0T0C1, P0T1C0, P0T1C1, ..., P1T0C0, ...
    """
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
                assert is_correct, f"Position {p}, Time {p}, Channel {c}: {msg}"


def test_axis_order_ctp(tmp_path: Path) -> None:
    """Test that axis_order='ctp' works correctly.

    This tests a different ordering where channels are acquired first,
    then timepoints, then positions.

    Acquisition order with ctp: P0T0C0, P1T0C0, P0T1C0, P1T1C0, ..., P0T0C1, ...
    Dimension order: [c, t, p, y, x] (non-position: [c, t])
    """
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
                assert is_correct, f"Position {p}, Time {t}, Channel {c}: {msg}"


def test_axis_order_tensorstore(tmp_path: Path) -> None:
    """Test that axis_order works with tensorstore backend."""
    pytest.importorskip("tensorstore")
    from useq import MDASequence

    output_path = tmp_path / "test_tensorstore_tpc.zarr"

    # Create sequence with axis_order="tpc"
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
        backend="tensorstore",
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

    import zarr

    zg = zarr.open_group(output_path, mode="r")

    # Check all positions, timepoints, and channels
    for p in range(2):
        pos_array = zg[str(p)]
        for t in range(3):
            for c in range(2):
                frame_data = pos_array[t, c, :, :]
                is_correct, msg = verify_frame_value(frame_data, p, t, c)
                assert is_correct, f"Position {p}, Time {t}, Channel {c}: {msg}"


def test_axis_order_tiff(tmp_path: Path) -> None:
    """Test that axis_order works with tiff backend."""
    pytest.importorskip("tifffile")
    from useq import MDASequence

    output_path = tmp_path / "test_tiff_tpc.ome.tiff"

    # Create sequence with axis_order="tpc"
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
        backend="tiff",
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

    import tifffile

    # For multi-position TIFF, separate files are created with _p{idx:03d} suffix
    for p in range(2):
        tiff_path = tmp_path / f"test_tiff_tpc_p{p:03d}.ome.tiff"

        with tifffile.TiffFile(tiff_path) as tif:
            data = tif.asarray()
            # TIFF data shape is [t, c, y, x]
            for t in range(3):
                for c in range(2):
                    frame_data = data[t, c, :, :]
                    is_correct, msg = verify_frame_value(frame_data, p, t, c)
                    assert is_correct, f"Position {p}, Time {t}, Channel {c}: {msg}"
