"""Test that axis_order is properly respected in multi-position acquisitions.

These tests verify that frames are written to the correct array positions
when using different axis_order values in MDASequence. Each test creates
frames with unique statistical properties (mean, std) that encode the
exact indices (p, t, c), making it easy to verify correctness.
"""

from __future__ import annotations

from contextlib import suppress
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


@pytest.mark.parametrize("multi_position", [True, False])
@pytest.mark.parametrize("axis_order", ["ptzc", "ptcz", "tpzc", "cztp", "tpcz"])
def test_axis_order(
    axis_order: str, multi_position: bool, backend: AvailableBackend, tmp_path: Path
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
        output_path = tmp_path / f"test_{axis_order}.{backend.file_ext}"

    # Create sequence with specified axis_order
    # 2 positions, 3 timepoints, 2 channels, 4 z-slices = 48 frames
    seq = MDASequence(
        axis_order=axis_order,
        stage_positions=[(0.0, 0.0), (1.0, 1.0)] if multi_position else [],
        time_plan={"interval": 0.1, "loops": 3},
        channels=["DAPI", "FITC"],
        z_plan={"range": 3.0, "step": 1.0},
    )

    dims = omew.dims_from_useq(seq, image_width=32, image_height=32)

    stream = omew.create_stream(
        path=str(output_path),
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

    range_p = range(len(seq.stage_positions)) if multi_position else range(1)
    range_t = range(seq.time_plan.loops)
    range_c = range(len(seq.channels))
    range_z = range(seq.z_plan.num_positions())

    # Verify data based on backend type
    if backend.name in ("acquire-zarr", "tensorstore"):
        pytest.importorskip("zarr", reason="zarr not installed")
        import zarr

        zg = zarr.open_group(output_path, mode="r")

        # we are only validating tensorstore because acquire-zarr allows to save
        # data in acquisition order and thus the validation will fail in some cases
        if backend.name == "tensorstore":
            with suppress(ImportError):
                from yaozarrs import validate_zarr_store

                validate_zarr_store(output_path)

        # Check all positions, timepoints, channels, and z-slices
        # Get the dimension order from dims (which reflects storage order)
        non_pos_dims = [d.label for d in dims if d.label not in "pyx"]
        for p in range_p:
            pos_array = zg[str(p)]
            for t in range_t:
                for c in range_c:
                    for z in range_z:
                        idx_tuple = (t, c, z)

                        frame_data = pos_array[(*idx_tuple, slice(None), slice(None))]

                        is_correct, msg = verify_frame_value(frame_data, p, t, c, z)
                        assert is_correct, (
                            f"Position {p}, Time {t}, Channel {c}, Z {z}: {msg}"
                        )

    elif backend.name == "tiff":
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

        # TIFF data is stored in acquisition order (axis_order)
        # Get the dimension order from dims (which reflects acquisition order)
        non_pos_dims = [d.label for d in dims if d.label not in "pyx"]

        for p in range_p:
            if multi_position:
                tiff_path = tmp_path / f"{Path(base_name).name}_p{p:03d}{ext}"
            else:
                tiff_path = output_path

            with tifffile.TiffFile(tiff_path) as tif:
                data = tif.asarray()
                # TIFF data is stored in acquisition order
                for t in range_t:
                    for c in range_c:
                        for z in range_z:
                            # Build index based on acquisition order
                            indices = {"t": t, "c": c, "z": z}
                            idx_tuple = tuple(indices[d] for d in non_pos_dims)
                            frame_data = data[(*idx_tuple, slice(None), slice(None))]
                            is_correct, msg = verify_frame_value(frame_data, p, t, c, z)
                            assert is_correct, (
                                f"Position {p}, Time {t}, Channel {c}, Z {z}: {msg}"
                            )
                            with suppress(ImportError):
                                from ome_types import validate_xml

                                assert tif.ome_metadata is not None
                                validate_xml(tif.ome_metadata)
