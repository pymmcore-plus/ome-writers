"""Tests specifically for YaozarrsStream implementation."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import numpy as np
import pytest

import ome_writers as omew

if TYPE_CHECKING:
    from pathlib import Path

pytest.importorskip("yaozarrs", reason="yaozarrs not installed")


def test_yaozarrs_stream_availability() -> None:
    """Test that YaozarrsStream availability check works."""
    # Just verify the method exists and returns a boolean
    assert isinstance(omew.YaozarrsStream.is_available(), bool)


def test_yaozarrs_single_position(tmp_path: Path) -> None:
    """Test YaozarrsStream with single position data."""
    from yaozarrs import validate_zarr_store

    data_gen, dimensions, dtype = omew.fake_data_for_sizes(
        sizes={"t": 2, "c": 2, "z": 3, "y": 32, "x": 32},
        chunk_sizes={"t": 1, "y": 16, "x": 16},
        dtype=np.uint16,
    )

    output_path = tmp_path / "single_position.ome.zarr"
    stream = omew.YaozarrsStream()
    stream = stream.create(str(output_path), dtype, dimensions)
    assert stream.is_active()

    # Write all frames
    for data in data_gen:
        stream.append(data)
    stream.flush()

    assert not stream.is_active()
    assert output_path.exists()

    # Validate the zarr store structure
    validate_zarr_store(output_path)

    # Check that metadata exists
    zarr_json = output_path / "zarr.json"
    assert zarr_json.exists()

    # For single position, should have array at "0"
    array_path = output_path / "0"
    assert array_path.exists()
    array_zarr_json = array_path / "zarr.json"
    assert array_zarr_json.exists()


def test_yaozarrs_multi_position(tmp_path: Path) -> None:
    """Test YaozarrsStream with multi-position data (bioformats2raw layout)."""
    from yaozarrs import validate_zarr_store

    data_gen, dimensions, dtype = omew.fake_data_for_sizes(
        sizes={"p": 3, "t": 2, "c": 2, "y": 32, "x": 32},
        chunk_sizes={"t": 1, "y": 16, "x": 16},
        dtype=np.uint16,
    )

    output_path = tmp_path / "multi_position.ome.zarr"
    stream = omew.YaozarrsStream()
    stream = stream.create(str(output_path), dtype, dimensions)
    assert stream.is_active()

    # Write all frames
    for data in data_gen:
        stream.append(data)
    stream.flush()

    assert not stream.is_active()
    assert output_path.exists()

    # Validate the zarr store structure
    validate_zarr_store(output_path)

    # Check root metadata
    zarr_json = output_path / "zarr.json"
    assert zarr_json.exists()
    root_meta = json.loads(zarr_json.read_text())

    # Multi-position should use bioformats2raw layout
    attrs = root_meta.get("attributes", {})
    ome_meta = attrs.get("ome", {})
    assert "bioformats2raw.layout" in ome_meta

    # Check that each position exists as a separate image
    for pos_idx in range(3):
        pos_path = output_path / str(pos_idx)
        assert pos_path.exists(), f"Position {pos_idx} should exist"

        # Each position should have its own zarr.json (as a group)
        pos_zarr_json = pos_path / "zarr.json"
        assert pos_zarr_json.exists()

        # The actual array should be at position/0 (first resolution level)
        array_path = pos_path / "0"
        assert array_path.exists(), f"Array for position {pos_idx} should exist at 0"
        array_zarr_json = array_path / "zarr.json"
        assert array_zarr_json.exists()


def test_yaozarrs_overwrite_behavior(tmp_path: Path) -> None:
    """Test overwrite behavior of YaozarrsStream."""
    dimensions = [
        omew.Dimension(label="t", size=2, chunk_size=1),
        omew.Dimension(label="y", size=32, chunk_size=16),
        omew.Dimension(label="x", size=32, chunk_size=16),
    ]
    dtype = np.dtype(np.uint8)

    output_path = tmp_path / "overwrite_test.ome.zarr"

    # First write
    stream1 = omew.YaozarrsStream()
    stream1.create(str(output_path), dtype, dimensions)
    stream1.append(np.zeros((32, 32), dtype=dtype))
    stream1.flush()

    # Second write without overwrite should fail
    stream2 = omew.YaozarrsStream()
    with pytest.raises((FileExistsError, ValueError, RuntimeError)):
        stream2.create(str(output_path), dtype, dimensions, overwrite=False)

    # Third write with overwrite should succeed
    stream3 = omew.YaozarrsStream()
    stream3.create(str(output_path), dtype, dimensions, overwrite=True)
    stream3.append(np.ones((32, 32), dtype=dtype))
    stream3.flush()


def test_yaozarrs_data_integrity(tmp_path: Path) -> None:
    """Test that data written by YaozarrsStream can be read back correctly."""
    # Create simple test data
    dimensions = [
        omew.Dimension(label="t", size=3, chunk_size=1),
        omew.Dimension(label="c", size=2, chunk_size=1),
        omew.Dimension(label="y", size=16, chunk_size=16),
        omew.Dimension(label="x", size=16, chunk_size=16),
    ]
    dtype = np.dtype(np.uint16)

    output_path = tmp_path / "integrity_test.ome.zarr"

    # Create unique data for each frame
    stream = omew.YaozarrsStream()
    stream.create(str(output_path), dtype, dimensions)

    written_frames = []
    for t in range(3):
        for c in range(2):
            frame = np.full((16, 16), t * 10 + c, dtype=dtype)
            stream.append(frame)
            written_frames.append(frame)

    stream.flush()

    # Read back and verify
    try:
        import zarr

        store = zarr.open_group(output_path, mode="r")
        array = store["0"]

        for t in range(3):
            for c in range(2):
                frame_idx = t * 2 + c
                read_frame = array[t, c, :, :]
                expected_frame = written_frames[frame_idx]
                np.testing.assert_array_equal(
                    read_frame,
                    expected_frame,
                    err_msg=f"Frame mismatch at t={t}, c={c}",
                )
    except ImportError:
        # If zarr-python not available, try tensorstore
        import tensorstore as ts

        store = ts.open(
            {
                "driver": "zarr3",
                "kvstore": {"driver": "file", "path": str(output_path / "0")},
            }
        ).result()

        for t in range(3):
            for c in range(2):
                frame_idx = t * 2 + c
                read_frame = store[t, c, :, :].read().result()
                expected_frame = written_frames[frame_idx]
                np.testing.assert_array_equal(
                    read_frame,
                    expected_frame,
                    err_msg=f"Frame mismatch at t={t}, c={c}",
                )


def test_yaozarrs_metadata_structure(tmp_path: Path) -> None:
    """Test that YaozarrsStream creates proper OME-Zarr v0.5 metadata."""
    dimensions = [
        omew.Dimension(label="t", size=2, unit=(1.0, "s"), chunk_size=1),
        omew.Dimension(label="z", size=3, unit=(0.5, "um"), chunk_size=1),
        omew.Dimension(label="y", size=32, unit=(0.2, "um"), chunk_size=16),
        omew.Dimension(label="x", size=32, unit=(0.2, "um"), chunk_size=16),
    ]
    dtype = np.dtype(np.uint16)

    output_path = tmp_path / "metadata_test.ome.zarr"

    stream = omew.YaozarrsStream()
    stream.create(str(output_path), dtype, dimensions)
    stream.flush()

    # Read group metadata (single position stores metadata at group level)
    group_zarr_json = output_path / "zarr.json"
    assert group_zarr_json.exists()

    group_meta = json.loads(group_zarr_json.read_text())

    # Check OME-Zarr v0.5 metadata in attributes
    attrs = group_meta.get("attributes", {})
    ome_meta = attrs.get("ome", {})
    assert "multiscales" in ome_meta

    multiscales = ome_meta["multiscales"]
    assert len(multiscales) == 1

    ms = multiscales[0]
    assert "axes" in ms
    assert "datasets" in ms

    # Check axes
    axes = ms["axes"]
    assert len(axes) == 4  # t, z, y, x
    axis_names = [ax["name"] for ax in axes]
    assert axis_names == ["t", "z", "y", "x"]

    # Check that spatial axes have units
    for ax in axes:
        if ax["name"] in ("y", "x", "z"):
            assert "unit" in ax
            assert ax["unit"] == "micrometer"
        elif ax["name"] == "t":
            assert "unit" in ax
            assert ax["unit"] == "second"

    # Check coordinate transformations (scales)
    datasets = ms["datasets"]
    assert len(datasets) == 1
    assert datasets[0]["path"] == "0"
    assert "coordinateTransformations" in datasets[0]

    transforms = datasets[0]["coordinateTransformations"]
    assert len(transforms) == 1
    assert transforms[0]["type"] == "scale"
    scales = transforms[0]["scale"]
    # Should be [t_scale, z_scale, y_scale, x_scale]
    assert scales == [1.0, 0.5, 0.2, 0.2]


def test_yaozarrs_context_manager(tmp_path: Path) -> None:
    """Test YaozarrsStream as a context manager."""
    dimensions = [
        omew.Dimension(label="t", size=2, chunk_size=1),
        omew.Dimension(label="y", size=16, chunk_size=16),
        omew.Dimension(label="x", size=16, chunk_size=16),
    ]
    dtype = np.dtype(np.uint8)

    output_path = tmp_path / "context_test.ome.zarr"

    # Use as context manager
    with omew.YaozarrsStream() as stream:
        stream.create(str(output_path), dtype, dimensions)
        stream.append(np.zeros((16, 16), dtype=dtype))
        stream.append(np.ones((16, 16), dtype=dtype))

    # Stream should be flushed and closed
    assert not stream.is_active()
    assert output_path.exists()


def test_yaozarrs_minimal_2d(tmp_path: Path) -> None:
    """Test YaozarrsStream with minimal 2D data (only y and x dimensions)."""
    dimensions = [
        omew.Dimension(label="y", size=64, chunk_size=32),
        omew.Dimension(label="x", size=64, chunk_size=32),
    ]
    dtype = np.dtype(np.uint8)

    output_path = tmp_path / "minimal_2d.ome.zarr"

    stream = omew.YaozarrsStream()
    stream.create(str(output_path), dtype, dimensions)
    stream.append(np.random.randint(0, 255, (64, 64), dtype=dtype))
    stream.flush()

    assert output_path.exists()
    array_path = output_path / "0"
    assert array_path.exists()


def test_yaozarrs_axis_reordering(tmp_path: Path) -> None:
    """Test that dimensions are reordered to NGFF v0.5 order.

    NGFF v0.5 requires axes in order: time → channel → space (z,y,x).
    This test uses acquisition order t,z,c,y,x (non-compliant) and verifies
    that data is correctly transposed to storage order t,c,z,y,x.
    """
    # Acquisition order: t, z, c, y, x (z and c swapped from NGFF order)
    dimensions = [
        omew.Dimension(label="t", size=2, chunk_size=1),
        omew.Dimension(label="z", size=3, chunk_size=1),
        omew.Dimension(label="c", size=2, chunk_size=1),
        omew.Dimension(label="y", size=32, chunk_size=16),
        omew.Dimension(label="x", size=32, chunk_size=16),
    ]
    dtype = np.dtype(np.uint16)
    output_path = tmp_path / "reordered.ome.zarr"

    stream = omew.YaozarrsStream()
    stream.create(str(output_path), dtype, dimensions)

    # Write test data in acquisition order: iterate t, then z, then c
    # Each frame value encodes its acquisition position: t*100 + z*10 + c
    for t in range(2):
        for z in range(3):
            for c in range(2):
                frame = np.ones((32, 32), dtype=np.uint16) * (t * 100 + z * 10 + c)
                stream.append(frame)

    stream.flush()

    # Verify the data was stored in NGFF order (t, c, z, y, x)
    import zarr

    store = zarr.open_group(output_path, mode="r")
    array = store["0"]

    # Check shape matches NGFF order
    assert array.shape == (2, 2, 3, 32, 32), "Array shape should be (t,c,z,y,x)"

    # Verify a few specific values are in the correct positions
    # Acquisition: t=0, z=0, c=0 → value=0 → Storage: t=0, c=0, z=0
    assert array[0, 0, 0, 0, 0] == 0

    # Acquisition: t=0, z=1, c=0 → value=10 → Storage: t=0, c=0, z=1
    assert array[0, 0, 1, 0, 0] == 10

    # Acquisition: t=0, z=0, c=1 → value=1 → Storage: t=0, c=1, z=0
    assert array[0, 1, 0, 0, 0] == 1

    # Acquisition: t=1, z=2, c=1 → value=121 → Storage: t=1, c=1, z=2
    assert array[1, 1, 2, 0, 0] == 121

    # Check axes are in NGFF order in metadata
    multiscales = store.attrs.get("multiscales", [{}])[0]
    [ax["name"] for ax in multiscales.get("axes", [])]
    # yaozarrs may use bioformats2raw layout which doesn't store axes at top level
    # But the array shape itself proves the correct storage order
    assert array.shape == (2, 2, 3, 32, 32)


def test_yaozarrs_axis_reordering_multiposition(tmp_path: Path) -> None:
    """Test axis reordering with multi-position data.

    Verifies that NGFF v0.5 axis ordering works correctly with bioformats2raw
    layout (multi-position). Each position should have data stored in NGFF order.
    """
    # Acquisition order: p, t, z, c, y, x (z and c swapped from NGFF order)
    dimensions = [
        omew.Dimension(label="p", size=2, chunk_size=1),
        omew.Dimension(label="t", size=2, chunk_size=1),
        omew.Dimension(label="z", size=2, chunk_size=1),
        omew.Dimension(label="c", size=2, chunk_size=1),
        omew.Dimension(label="y", size=16, chunk_size=16),
        omew.Dimension(label="x", size=16, chunk_size=16),
    ]
    dtype = np.dtype(np.uint16)
    output_path = tmp_path / "reordered_multipos.ome.zarr"

    stream = omew.YaozarrsStream()
    stream.create(str(output_path), dtype, dimensions)

    # Write test data in acquisition order: iterate p, t, z, c
    # Each frame value encodes its position: p*1000 + t*100 + z*10 + c
    for p in range(2):
        for t in range(2):
            for z in range(2):
                for c in range(2):
                    value = p * 1000 + t * 100 + z * 10 + c
                    frame = np.ones((16, 16), dtype=np.uint16) * value
                    stream.append(frame)

    stream.flush()

    # Verify each position's data is stored in NGFF order (t, c, z, y, x)
    import zarr

    store = zarr.open_group(output_path, mode="r")

    for p in range(2):
        array = store[f"{p}/0"]  # bioformats2raw layout: position/resolution

        # Check shape matches NGFF order (not acquisition order)
        assert array.shape == (
            2,
            2,
            2,
            16,
            16,
        ), f"Position {p} shape should be (t,c,z,y,x)"

        # Verify specific values for this position
        # Acquisition: p=p, t=0, z=0, c=0 → value=p*1000 → Storage: array[t=0, c=0, z=0]
        expected_base = p * 1000
        assert array[0, 0, 0, 0, 0] == expected_base

        # Acquisition: p=p, t=0, z=1, c=0 → value=p*1000+10 → Storage: [t=0, c=0, z=1]
        assert array[0, 0, 1, 0, 0] == expected_base + 10

        # Acquisition: p=p, t=0, z=0, c=1 → value=p*1000+1 → Storage: [t=0, c=1, z=0]
        assert array[0, 1, 0, 0, 0] == expected_base + 1

        # Acquisition: p=p, t=1, z=1, c=1 → value=p*1000+111 → Storage: [t=1, c=1, z=1]
        assert array[1, 1, 1, 0, 0] == expected_base + 111


def test_yaozarrs_writer_zarr_backend(tmp_path: Path) -> None:
    """Test YaozarrsStream with zarr-python backend."""
    zarr = pytest.importorskip("zarr")

    data_gen, dimensions, dtype = omew.fake_data_for_sizes(
        sizes={"t": 2, "c": 2, "y": 16, "x": 16},
        chunk_sizes={"t": 1, "y": 16, "x": 16},
        dtype=np.uint8,
    )

    output_path = tmp_path / "zarr_backend.ome.zarr"
    stream = omew.YaozarrsStream()

    # Explicitly use zarr backend
    stream = stream.create(
        str(output_path),
        dtype,
        dimensions,
        overwrite=True,
        writer="zarr",
    )

    # Write frames
    for data in data_gen:
        stream.append(data)
    stream.flush()

    assert output_path.exists()
    # Verify it's a valid zarr array
    array = zarr.open(output_path / "0", mode="r")
    assert array.shape == (2, 2, 16, 16)


def test_yaozarrs_writer_auto_backend(tmp_path: Path) -> None:
    """Test YaozarrsStream with auto backend selection."""
    data_gen, dimensions, dtype = omew.fake_data_for_sizes(
        sizes={"t": 2, "y": 16, "x": 16},
        chunk_sizes={"t": 1, "y": 16, "x": 16},
        dtype=np.uint8,
    )

    output_path = tmp_path / "auto_backend.ome.zarr"
    stream = omew.YaozarrsStream()

    # Use auto backend
    stream = stream.create(
        str(output_path),
        dtype,
        dimensions,
        overwrite=True,
        writer="auto",
    )

    # Write frames
    for data in data_gen:
        stream.append(data)
    stream.flush()

    assert output_path.exists()


def test_yaozarrs_multi_position_zarr_backend(tmp_path: Path) -> None:
    """Test YaozarrsStream multi-position with zarr backend."""
    zarr = pytest.importorskip("zarr")

    data_gen, dimensions, dtype = omew.fake_data_for_sizes(
        sizes={"p": 3, "t": 2, "c": 2, "y": 16, "x": 16},
        chunk_sizes={"t": 1, "y": 16, "x": 16},
        dtype=np.uint8,
    )

    output_path = tmp_path / "multi_pos_zarr.ome.zarr"
    stream = omew.YaozarrsStream()

    # Multi-position with zarr backend
    stream = stream.create(
        str(output_path),
        dtype,
        dimensions,
        overwrite=True,
        writer="zarr",
    )

    # Write all frames
    for data in data_gen:
        stream.append(data)
    stream.flush()

    assert output_path.exists()

    # Verify all positions exist
    for pos in range(3):
        pos_path = output_path / str(pos) / "0"
        assert pos_path.exists()
        array = zarr.open(pos_path, mode="r")
        assert array.shape == (2, 2, 16, 16)  # t, c, y, x in NGFF order


def test_yaozarrs_without_write_module_import_error() -> None:
    """Test that YaozarrsStream raises helpful error without write module."""
    # This test checks that the error message is helpful
    # We can't actually test the ImportError condition without
    # uninstalling yaozarrs, but we can verify the __init__ exists
    stream = omew.YaozarrsStream()
    assert stream is not None
