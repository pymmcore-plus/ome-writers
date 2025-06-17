"""Tests for ome-writers library."""

from __future__ import annotations

import importlib
import importlib.util
import json
from importlib.metadata import version
from pathlib import Path
from typing import TYPE_CHECKING, Literal

import numpy as np
import pytest
import tensorstore as ts
import tifffile

from ome_writers import (
    AcquireZarrStream,
    Dimension,
    OMEStream,
    TensorStoreZarrStream,
    TifffileStream,
    create_stream,
    fake_data_for_sizes,
)
from ome_writers._auto import BACKENDS


def read_file_data(output_path: Path) -> np.ndarray:
    """Read data from a zarr or tiff file and return as numpy array."""
    file_ext = output_path.suffix.lstrip(".")

    if file_ext == "zarr":
        spec = {
            "driver": "zarr3",
            "kvstore": {"driver": "file", "path": str(output_path / "0")},
        }
        store = ts.open(spec).result()
        return store.read().result()  # type: ignore

    elif file_ext == "tiff":
        return tifffile.imread(str(output_path))

    else:
        pytest.fail(f"Unknown file extension: {file_ext}")


if TYPE_CHECKING:
    from pathlib import Path


# Test configurations for each backend
backends_to_test: list[tuple[type[OMEStream], str]] = []
if importlib.util.find_spec("tensorstore") is not None:
    backends_to_test.append((TensorStoreZarrStream, "zarr"))
if importlib.util.find_spec("acquire_zarr") is not None:
    backends_to_test.append((AcquireZarrStream, "zarr"))
    AQ_VERSION = tuple(int(x) for x in version("acquire-zarr").split(".")[:3])

if importlib.util.find_spec("tifffile") is not None:
    backends_to_test.append((TifffileStream, "tiff"))


@pytest.mark.parametrize("stream_cls,file_ext", backends_to_test)
def test_minimal_2d_dimensions(
    stream_cls: type[OMEStream], file_ext: str, tmp_path: Path
) -> None:
    """Test with minimal 2D dimensions (just x and y)."""
    data_gen, dimensions, dtype = fake_data_for_sizes(
        sizes={"t": 1, "y": 32, "x": 32},
        chunk_sizes={"t": 1, "y": 16, "x": 16},
        dtype=np.uint8,
    )

    # Create the stream
    stream = stream_cls()

    # Set output path
    output_path = tmp_path / f"test_2d_{stream_cls.__name__.lower()}.{file_ext}"

    stream = stream.create(str(output_path), dtype, dimensions)
    assert stream.is_active()

    # Get the data from the generator
    for data in data_gen:
        stream.append(data)
    stream.flush()

    assert not stream.is_active()
    assert output_path.exists()


@pytest.mark.parametrize("stream_cls,file_ext", backends_to_test)
def test_stream_error_handling(
    stream_cls: type[OMEStream], file_ext: str, tmp_path: Path
) -> None:
    """Test error handling in streams."""
    empty_stream = stream_cls()

    expected_message = "Stream is closed or uninitialized"
    test_frame = np.zeros((64, 64), dtype=np.uint16)
    with pytest.raises(RuntimeError, match=expected_message):
        empty_stream.append(test_frame)


def test_dimension_info_properties() -> None:
    """Test DimensionInfo properties."""
    # Test spatial dimension
    x_dim = Dimension(label="x", size=100, unit=(0.5, "um"), chunk_size=50)
    assert x_dim.ome_dim_type == "space"
    assert x_dim.ome_unit == "micrometer"
    assert x_dim.ome_scale == 0.5

    # Test time dimension
    t_dim = Dimension(label="t", size=10, unit=(2.0, "s"), chunk_size=1)
    assert t_dim.ome_dim_type == "time"
    assert t_dim.ome_unit == "second"
    assert t_dim.ome_scale == 2.0

    # Test channel dimension
    c_dim = Dimension(label="c", size=3, chunk_size=1)
    assert c_dim.ome_dim_type == "channel"
    assert c_dim.ome_unit == "unknown"
    assert c_dim.ome_scale == 1.0

    # Test custom dimension
    p_dim = Dimension(label="p", size=5, chunk_size=1)
    assert p_dim.ome_dim_type == "other"


backends_reverse = {v: k for k, v in BACKENDS.items()}
backend_names = [(backends_reverse[cls], ext) for cls, ext in backends_to_test]
backend_names += [("auto", "zarr"), ("auto", "tiff")]


@pytest.mark.parametrize("backend_name,file_ext", backend_names)
def test_create_stream_factory_function(
    backend_name: Literal["acquire-zarr", "tensorstore", "tiff", "auto"],
    file_ext: str,
    tmp_path: Path,
) -> None:
    """Test the create_stream factory function."""
    data_gen, dimensions, dtype = fake_data_for_sizes(
        sizes={"t": 3, "z": 2, "c": 2, "y": 64, "x": 64},
        chunk_sizes={"y": 32, "x": 32},
    )

    output_path = tmp_path / f"factory_test.{file_ext}"

    stream = create_stream(str(output_path), dtype, dimensions, backend=backend_name)
    assert isinstance(stream, OMEStream)
    assert stream.is_active()

    # Write a test frame
    for data in data_gen:
        stream.append(data)
        break  # Just write one frame for this test

    stream.flush()
    assert not stream.is_active()
    assert output_path.exists()


@pytest.mark.parametrize(
    "dtype", [np.dtype(np.uint8), np.dtype(np.uint16)], ids=["uint8", "uint16"]
)
@pytest.mark.parametrize("stream_cls,file_ext", backends_to_test)
def test_data_integrity_roundtrip(
    stream_cls: type[OMEStream],
    file_ext: str,
    tmp_path: Path,
    dtype: np.dtype,
) -> None:
    """Test data integrity roundtrip with different data types."""
    data_gen, dimensions, dtype = fake_data_for_sizes(
        sizes={"t": 3, "z": 2, "c": 2, "y": 64, "x": 64},
        chunk_sizes={"y": 32, "x": 32},
    )

    # Convert generator to list to use data multiple times
    original_frames = list(data_gen)

    output_path = tmp_path / f"{stream_cls.__name__.lower()}_{dtype.name}.{file_ext}"

    # Write data using our stream
    stream = stream_cls()
    stream = stream.create(str(output_path), dtype, dimensions)
    assert stream.is_active()
    for frame in original_frames:
        stream.append(frame)

    stream.flush()
    assert not stream.is_active()

    # Read data back and verify it matches
    assert output_path.exists()
    read_data = read_file_data(output_path)

    # Reconstruct original data array from frames
    shape = tuple(d.size for d in dimensions)
    original_data = np.array(original_frames).reshape(shape)

    # Verify the data matches exactly
    np.testing.assert_array_equal(
        original_data,
        read_data,
        err_msg=f"Data mismatch in {stream_cls.__name__} roundtrip test with {dtype}",
    )

    # Test 2: Try to create again without overwrite (should fail)
    with pytest.raises(FileExistsError, match=".*already exists"):
        stream = stream_cls()
        stream = stream.create(str(output_path), dtype, dimensions, overwrite=False)

    # Test 3: Create again with overwrite=True (should succeed)
    stream = stream_cls()
    stream = stream.create(str(output_path), dtype, dimensions, overwrite=True)
    assert isinstance(stream, OMEStream)
    assert stream.is_active()

    for frame in original_frames:
        stream.append(frame)
    stream.flush()
    assert not stream.is_active()
    assert output_path.exists()


@pytest.mark.parametrize("stream_cls,file_ext", backends_to_test)
def test_multiposition_acquisition(
    stream_cls: type[OMEStream], file_ext: str, tmp_path: Path
) -> None:
    """Test multi-position acquisition support with position dimension."""

    if stream_cls == AcquireZarrStream:
        if AQ_VERSION <= (0, 4, 0):
            pytest.xfail("AcquireZarrStream does not support multi-position in 0.4.0")

    data_gen, dimensions, dtype = fake_data_for_sizes(
        sizes={"t": 3, "z": 2, "c": 2, "y": 32, "x": 32, "p": 3},
        chunk_sizes={"y": 16, "x": 16},
    )

    # Create the stream
    stream = stream_cls()
    output_path = tmp_path / f"test_multipos_{stream_cls.__name__.lower()}.{file_ext}"

    stream = stream.create(str(output_path), dtype, dimensions)
    assert stream.is_active()

    # Write frames for all positions and time/channel combinations
    # Total frames = 3 positions x 2 time x 2 channels = 12 frames
    for frame in data_gen:
        stream.append(frame)

    stream.flush()
    assert not stream.is_active()

    if file_ext == "zarr":
        assert output_path.exists()
        # Verify zarr structure
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

    elif file_ext == "tiff":
        # For TIFF, separate files are created for each position
        base_path = output_path.with_suffix("")
        assert (base_path.with_name(f"{base_path.name}_p000.tiff")).exists()
        assert (base_path.with_name(f"{base_path.name}_p001.tiff")).exists()
        assert (base_path.with_name(f"{base_path.name}_p002.tiff")).exists()

        # Verify that each TIFF file has the correct metadata and shape
        for pos_idx in range(3):
            pos_file = base_path.with_name(f"{base_path.name}_p{pos_idx:03d}.tiff")
            assert pos_file.exists()

            # Read the file to verify it has correct shape
            with tifffile.TiffFile(str(pos_file)) as tif:
                # Shape should be (t, z, c, y, x) = (3, 2, 2, 32, 32)
                expected_shape = (3, 2, 2, 32, 32)
                assert tif.asarray().shape == expected_shape
