"""Tests for ome-writers library."""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pytest

import ome_writers as omew

from .conftest import validate_path

if TYPE_CHECKING:
    from .conftest import AvailableBackend


def test_minimal_2d_dimensions(backend: AvailableBackend, tmp_path: Path) -> None:
    """Test with minimal 2D dimensions (just x and y)."""
    data_gen, dimensions, dtype = omew.fake_data_for_sizes(
        sizes={"t": 1, "y": 32, "x": 32},
        chunk_sizes={"t": 1, "y": 16, "x": 16},
        dtype=np.uint8,
    )

    # Create the stream
    stream = backend.cls()

    # Set output path
    output_path = tmp_path / f"test_2d_{backend.name.lower()}.{backend.file_ext}"

    stream = stream.create(str(output_path), dtype, dimensions)
    assert stream.is_active()

    # Get the data from the generator
    for data in data_gen:
        stream.append(data)
    stream.flush()

    assert not stream.is_active()
    validate_path(output_path)


def test_stream_error_handling(backend: AvailableBackend) -> None:
    """Test error handling in streams."""
    empty_stream = backend.cls()

    expected_message = "Stream is closed or uninitialized"
    test_frame = np.zeros((64, 64), dtype=np.uint16)
    with pytest.raises(RuntimeError, match=expected_message):
        empty_stream.append(test_frame)


def test_dimension_info_properties() -> None:
    """Test DimensionInfo properties."""
    # Test spatial dimension
    x_dim = omew.Dimension(label="x", size=100, unit=(0.5, "um"), chunk_size=50)
    assert x_dim.ome_dim_type == "space"
    assert x_dim.ome_unit == "micrometer"
    assert x_dim.ome_scale == 0.5

    # Test time dimension
    t_dim = omew.Dimension(label="t", size=10, unit=(2.0, "s"), chunk_size=1)
    assert t_dim.ome_dim_type == "time"
    assert t_dim.ome_unit == "second"
    assert t_dim.ome_scale == 2.0

    # Test channel dimension
    c_dim = omew.Dimension(label="c", size=3, chunk_size=1)
    assert c_dim.ome_dim_type == "channel"
    assert c_dim.ome_unit == "unknown"
    assert c_dim.ome_scale == 1.0

    # Test custom dimension
    p_dim = omew.Dimension(label="p", size=5, chunk_size=1)
    assert p_dim.ome_dim_type == "other"


def test_create_stream_factory_function(
    backend: AvailableBackend, tmp_path: Path
) -> None:
    """Test the create_stream factory function."""
    data_gen, dimensions, dtype = omew.fake_data_for_sizes(
        sizes={"t": 3, "c": 2, "z": 2, "y": 64, "x": 64},
        chunk_sizes={"y": 32, "x": 32},
    )

    output_path = tmp_path / f"factory_test.{backend.file_ext}"
    stream = omew.create_stream(
        str(output_path), dtype, dimensions, backend=backend.name
    )
    assert isinstance(stream, omew.OMEStream)
    assert stream.is_active()

    # Write a test frame
    for data in data_gen:
        stream.append(data)
        break  # Just write one frame for this test

    stream.flush()
    assert not stream.is_active()
    validate_path(output_path)


@pytest.mark.parametrize(
    "dtype", [np.dtype(np.uint8), np.dtype(np.uint16)], ids=["uint8", "uint16"]
)
@pytest.mark.parametrize(
    "sizes",
    [
        {"t": 3, "c": 2, "z": 2, "y": 64, "x": 64},  # TCZYX - canonical order
        {"c": 2, "t": 3, "z": 2, "y": 64, "x": 64},  # CTZYX - non-canonical
        {"t": 3, "z": 2, "c": 2, "y": 64, "x": 64},  # TZCYX - non-canonical
        {"z": 2, "y": 64, "x": 64},  # ZYX - only spatial (canonical)
        {"t": 2, "y": 64, "x": 64},  # TYX - time only (canonical)
    ],
    ids=["tczyx", "ctzyx", "tzcyx", "zyx", "tyx"],
)
def test_data_integrity_roundtrip(
    backend: AvailableBackend,
    tmp_path: Path,
    dtype: np.dtype,
    sizes: dict[str, int],
) -> None:
    """Test data integrity roundtrip with different data types and dimension orders."""
    data_gen, dimensions, dtype = omew.fake_data_for_sizes(
        sizes=sizes,
        chunk_sizes={"y": 32, "x": 32},
    )

    # Convert generator to list to use data multiple times
    original_frames = list(data_gen)
    dim_order = "".join([d.label for d in dimensions if d.label not in "yx"])
    filename = f"{backend.name.lower()}_{dtype.name}_{dim_order}{backend.file_ext}"
    output_path = tmp_path / filename

    # Write data using our stream
    stream = backend.cls()
    stream = stream.create(str(output_path), dtype, dimensions)
    assert stream.is_active()
    for frame in original_frames:
        stream.append(frame)

    stream.flush()
    assert not stream.is_active()

    # Read data back and verify it matches
    validate_path(output_path)
    disk_data = backend.read_data(output_path)

    # Reconstruct original data array from frames
    shape = tuple(d.size for d in dimensions)
    original_data = np.array(original_frames).reshape(shape)

    # Verify the data matches exactly
    np.testing.assert_array_equal(
        original_data,
        disk_data,
        err_msg=f"Data mismatch in {backend.name} roundtrip test with {dtype}",
    )

    # Test 2: Try to create again without overwrite (should fail)
    with pytest.raises(FileExistsError, match=r".*already exists"):
        stream = backend.cls()
        stream = stream.create(str(output_path), dtype, dimensions, overwrite=False)

    # Test 3: Create again with overwrite=True (should succeed)
    stream = backend.cls()
    stream = stream.create(str(output_path), dtype, dimensions, overwrite=True)
    assert isinstance(stream, omew.OMEStream)
    assert stream.is_active()

    for frame in original_frames:
        stream.append(frame)
    stream.flush()
    assert not stream.is_active()
    validate_path(output_path)


def test_multiposition_acquisition(backend: AvailableBackend, tmp_path: Path) -> None:
    """Test multi-position acquisition support with position dimension."""
    stream_cls = backend.cls
    data_gen, dimensions, dtype = omew.fake_data_for_sizes(
        sizes={"t": 3, "c": 2, "z": 2, "y": 32, "x": 32, "p": 3},
        chunk_sizes={"y": 16, "x": 16},
    )

    # Create the stream
    stream = stream_cls()
    output_path = (
        tmp_path / f"test_multipos_{stream_cls.__name__.lower()}.{backend.file_ext}"
    )

    stream = stream.create(str(output_path), dtype, dimensions)
    assert stream.is_active()

    # Write frames for all positions and time/channel combinations
    # Total frames = 3 positions x 2 time x 2 channels = 12 frames
    for frame in data_gen:
        stream.append(frame)

    stream.flush()
    assert not stream.is_active()

    if backend.file_ext.endswith("zarr"):
        validate_path(output_path)

        # Verify zarr structure (duplicate of above, but ensures positions are separate)
        assert (output_path / "0").exists()
        assert (output_path / "1").exists()
        assert (output_path / "2").exists()
        assert (output_path / "zarr.json").exists()

        # Verify that each position has correct metadata
        with open(output_path / "zarr.json") as f:
            group_meta = json.load(f)

        ome_attrs = group_meta["attributes"]["ome"]
        multiscales = ome_attrs["multiscales"]
        assert ome_attrs["version"] == "0.5"
        assert isinstance(multiscales, list)
        assert len(multiscales) == 1
        assert len(multiscales[0]["datasets"]) == 3

        axes_names = {ax["name"] for ax in multiscales[0]["axes"]}
        assert all(x in axes_names for x in ["t", "c", "y", "x"])

    elif (ext := backend.file_ext).endswith("tiff"):
        # For TIFF, separate files are created for each position
        base_path = Path(str(output_path).replace(ext, ""))

        # Verify that each TIFF file has the correct metadata and shape
        for pos_idx in range(3):
            pos_file = base_path.with_name(f"{base_path.name}_p{pos_idx:03d}{ext}")
            validate_path(pos_file)

            # Read the file to verify it has correct shape
            data = backend.read_data(pos_file)
            # Shape should be (t, z, c, y, x) = (3, 2, 2, 32, 32)
            expected_shape = (3, 2, 2, 32, 32)
            assert data.shape == expected_shape


@pytest.mark.parametrize(
    "sizes",
    [
        {"t": 2, "c": 3, "z": 4, "y": 32, "x": 32},  # TCZYX - canonical
        {"t": 2, "z": 4, "c": 3, "y": 32, "x": 32},  # TZCYX - non-canonical
        {"c": 3, "t": 2, "z": 4, "y": 32, "x": 32},  # CTZYX - non-canonical
    ],
)
def test_zarr_metadata_correctness(
    zarr_backend: AvailableBackend, sizes: dict[str, int], tmp_path: Path
) -> None:
    """Test that zarr metadata preserves acquisition order."""
    output_path = tmp_path / "test_metadata.zarr"
    data_gen, dims, dtype = omew.fake_data_for_sizes(sizes)

    stream = omew.create_stream(
        path=str(output_path),
        dimensions=dims,
        dtype=dtype,
        backend=zarr_backend.name,
        overwrite=True,
    )
    for frame in data_gen:
        stream.append(frame)
    stream.flush()

    # Check group metadata
    group_meta = json.loads((output_path / "zarr.json").read_text())
    axes = group_meta["attributes"]["ome"]["multiscales"][0]["axes"]

    # Verify axes are in acquisition order
    axis_info = {
        "t": {"type": "time", "unit": "second"},
        "c": {"type": "channel", "unit": "unknown"},
        "z": {"type": "space", "unit": "micrometer"},
        "y": {"type": "space", "unit": "micrometer"},
        "x": {"type": "space", "unit": "micrometer"},
    }
    expected_axes = [{"name": d.label, **axis_info[d.label]} for d in dims]
    assert axes == expected_axes

    # Check array metadata
    array_meta = json.loads((output_path / "0" / "zarr.json").read_text())
    assert array_meta["dimension_names"] == [d.label for d in dims]
    assert array_meta["shape"] == [d.size for d in dims]

    validate_path(output_path)
