"""Test that axis_order is properly respected in multi-position acquisitions."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pytest

import ome_writers as omew

from .conftest import validate_path

if TYPE_CHECKING:
    from .conftest import AvailableBackend

try:
    import useq
except ImportError:
    pytest.skip("useq not installed", allow_module_level=True)


def _encode_ptcz(p: int, t: int, c: int, z: int) -> int:
    """Encode p, t, c, z indices into a unique integer value."""
    return p * 10000 + t * 1000 + c * 100 + z * 10


def create_unique_frame(
    p: int, t: int, c: int, z: int, shape: tuple[int, int] = (32, 32)
) -> np.ndarray:
    """Create frame with unique value: p*10000 + t*1000 + c*100 + z*10."""
    return np.full(shape, _encode_ptcz(p, t, c, z), dtype=np.uint16)


def assert_frame_value(frame: np.ndarray, p: int, t: int, c: int, z: int) -> None:
    """Assert frame has expected value based on indices."""
    expected = _encode_ptcz(p, t, c, z)
    actual = float(np.mean(frame))
    assert abs(actual - expected) < 0.1, (
        f"Value mismatch at (p={p}, t={t}, c={c}, z={z}): "
        f"expected {expected}, got {actual:.0f}"
    )


@pytest.mark.parametrize("multi_position", [True, False])
@pytest.mark.parametrize(
    "axis_order",
    ["tczp", "ptzc", "ptcz", "tpzc", "cztp", "tpcz"],
)
def test_axis_order(
    axis_order: str, multi_position: bool, backend: AvailableBackend, tmp_path: Path
) -> None:
    """Test that different axis_order values work correctly."""

    ext = ".ome.tiff" if backend.file_ext.endswith("tiff") else f".{backend.file_ext}"
    output_path = tmp_path / f"test_{axis_order}{ext}"

    seq = useq.MDASequence(
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

    for event in seq:
        p, t, c, z = (event.index.get(k, 0) for k in "ptcz")
        stream.append(create_unique_frame(p, t, c, z))
    stream.flush()

    range_p = range(len(seq.stage_positions)) if multi_position else range(1)
    range_t = range(seq.time_plan.loops)
    range_c = range(len(seq.channels))
    range_z = range(seq.z_plan.num_positions())
    non_pos_dims = [d.label for d in dims if d.label not in "pyx"]

    if backend.name in ("acquire-zarr", "tensorstore", "zarr"):
        zarr = pytest.importorskip("zarr")

        # acquire-zarr and tensorstore create zarr v3 stores which cannot be read
        # by zarr v2. Skip the test if zarr v2 is used with these backends.
        if backend.name in ("acquire-zarr", "tensorstore"):
            from packaging.version import Version

            if Version(zarr.__version__) < Version("3.0.0"):
                return

        zg = zarr.open(output_path, mode="r")

        for p in range_p:
            # tensorstore and zarr use bioformats2raw layout: scale/array (0) for single
            # or position/scale/array (p/0) for multi-position
            # acquire-zarr uses direct layout: array (0) for single position
            # or position/array (p) for multi-position
            if backend.name in ("tensorstore", "zarr"):
                if multi_position:
                    pos_group = zg[str(p)]
                    pos_array = pos_group["0"]  # Scale level 0
                else:
                    pos_array = zg["0"]  # Scale level 0
                # tensorstore/zarr always use canonical NGFF order (tczyx)
                # so we always index as [t, c, z, y, x] regardless of acquisition order
                for t in range_t:
                    for c in range_c:
                        for z in range_z:
                            frame = pos_array[t, c, z, :, :]
                            assert_frame_value(frame, p, t, c, z)
            else:
                # acquire-zarr uses acquisition order
                pos_array = zg[str(p)]
                for t in range_t:
                    for c in range_c:
                        for z in range_z:
                            idx = tuple(
                                {"t": t, "c": c, "z": z}[d] for d in non_pos_dims
                            )
                            frame = pos_array[(*idx, slice(None), slice(None))]
                            assert_frame_value(frame, p, t, c, z)
        validate_path(output_path)

    elif backend.name == "tiff":
        tifffile = pytest.importorskip("tifffile")
        base_name = str(output_path).removesuffix(ext)

        for p in range_p:
            if multi_position:
                tiff_path = tmp_path / f"{Path(base_name).name}_p{p:03d}{ext}"
            else:
                tiff_path = output_path

            with tifffile.TiffFile(tiff_path) as tif:
                data = tif.asarray()
                for t in range_t:
                    for c in range_c:
                        for z in range_z:
                            indices = {"t": t, "c": c, "z": z}
                            idx = tuple(indices[d] for d in non_pos_dims)
                            frame = data[(*idx, slice(None), slice(None))]
                            assert_frame_value(frame, p, t, c, z)

                # Validate OME-XML once per file, not per frame
                try:
                    from ome_types import validate_xml

                    assert tif.ome_metadata is not None
                    validate_xml(tif.ome_metadata)
                except ImportError:
                    pass
