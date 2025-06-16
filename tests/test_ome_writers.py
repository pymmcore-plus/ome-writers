"""Tests for ome-writers library."""

from __future__ import annotations

import importlib
import importlib.util
from itertools import product
from pathlib import Path
from typing import TYPE_CHECKING, Literal

import numpy as np
import pytest

from ome_writers import (
    AcquireZarrStream,
    DimensionInfo,
    OMEStream,
    TensorStoreZarrStream,
    TiffStream,
)
from ome_writers._auto import BACKENDS


def read_file_data(output_path: Path) -> np.ndarray:
    """Read data from a zarr or tiff file and return as numpy array."""
    file_ext = output_path.suffix.lstrip(".")

    if file_ext == "zarr":
        import zarr

        zarr_group = zarr.open(str(output_path), mode="r")
        return zarr_group["0"][:]  # type: ignore[no-any-return]

    elif file_ext == "tiff":
        import tifffile

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
if importlib.util.find_spec("tifffile") is not None:
    backends_to_test.append((TiffStream, "tiff"))


@pytest.fixture
def sample_dimensions() -> list[DimensionInfo]:
    """Create sample dimensions for testing."""
    return [
        DimensionInfo(label="t", size=3, unit=(1.0, "s"), chunk_size=1),
        DimensionInfo(label="z", size=2, unit=(0.5, "um"), chunk_size=1),
        DimensionInfo(label="c", size=2, chunk_size=1),
        DimensionInfo(label="y", size=64, unit=(0.1, "um"), chunk_size=32),
        DimensionInfo(label="x", size=64, unit=(0.1, "um"), chunk_size=32),
    ]


@pytest.mark.parametrize("stream_cls,file_ext", backends_to_test)
def test_minimal_2d_dimensions(
    stream_cls: type[OMEStream], file_ext: str, tmp_path: Path
) -> None:
    """Test with minimal 2D dimensions (just x and y)."""
    dimensions = [
        DimensionInfo(label="t", size=1, unit=(2.0, "s"), chunk_size=1),
        DimensionInfo(label="y", size=32, unit=(1.0, "um"), chunk_size=16),
        DimensionInfo(label="x", size=32, unit=(1.0, "um"), chunk_size=16),
    ]

    # Create the stream
    stream = stream_cls()

    # Set output path
    output_path = tmp_path / f"test_2d_{stream_cls.__name__.lower()}.{file_ext}"

    data = np.random.randint(0, 255, size=(32, 32), dtype=np.uint8)

    stream = stream.create(str(output_path), data.dtype, dimensions)
    assert stream.is_active()

    stream.append(data)
    stream.flush()

    assert not stream.is_active()
    assert output_path.exists()


@pytest.mark.parametrize("stream_cls,file_ext", backends_to_test)
def test_stream_error_handling(
    stream_cls: type[OMEStream], file_ext: str, tmp_path: Path
) -> None:
    """Test error handling in streams."""
    # Test appending to uninitialized stream
    empty_stream = stream_cls()

    # Determine expected error message based on stream type
    if stream_cls.__name__ in ("TensorStoreZarrStream", "AcquireZarrStream"):
        expected_message = "Stream is closed or uninitialized"
    else:  # TiffStreamWriter
        expected_message = "Stream is not active"

    test_frame = np.zeros((64, 64), dtype=np.uint16)

    with pytest.raises(RuntimeError, match=expected_message):
        empty_stream.append(test_frame)


def test_dimension_info_properties() -> None:
    """Test DimensionInfo properties."""
    # Test spatial dimension
    x_dim = DimensionInfo(label="x", size=100, unit=(0.5, "um"), chunk_size=50)
    assert x_dim.ome_dim_type == "space"
    assert x_dim.ome_unit == "micrometer"
    assert x_dim.ome_scale == 0.5

    # Test time dimension
    t_dim = DimensionInfo(label="t", size=10, unit=(2.0, "s"), chunk_size=1)
    assert t_dim.ome_dim_type == "time"
    assert t_dim.ome_unit == "second"
    assert t_dim.ome_scale == 2.0

    # Test channel dimension
    c_dim = DimensionInfo(label="c", size=3, chunk_size=1)
    assert c_dim.ome_dim_type == "channel"
    assert c_dim.ome_unit == "unknown"
    assert c_dim.ome_scale == 1.0

    # Test custom dimension
    p_dim = DimensionInfo(label="p", size=5, chunk_size=1)
    assert p_dim.ome_dim_type == "other"


backends_reverse = {v: k for k, v in BACKENDS.items()}
backend_names = [(backends_reverse[cls], ext) for cls, ext in backends_to_test]
backend_names += [("auto", "zarr"), ("auto", "tiff")]


@pytest.mark.parametrize("backend_name,file_ext", backend_names)
def test_create_stream_factory_function(
    backend_name: Literal["acquire-zarr", "tensorstore", "tiff", "auto"],
    file_ext: str,
    tmp_path: Path,
    sample_dimensions: list[DimensionInfo],
) -> None:
    """Test the create_stream factory function."""
    from ome_writers import create_stream

    output_path = tmp_path / f"factory_test.{file_ext}"

    stream = create_stream(
        str(output_path), np.dtype(np.uint16), sample_dimensions, backend=backend_name
    )
    assert isinstance(stream, OMEStream)
    assert stream.is_active()

    # Write a test frame
    test_frame = np.random.randint(0, 65536, size=(64, 64), dtype=np.uint16)
    stream.append(test_frame)

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
    sample_dimensions: list[DimensionInfo],
) -> None:
    """Test data integrity roundtrip with different data types."""
    # Create deterministic random data for reproducible tests
    np.random.seed(123)  # Different seed from other test
    shape = tuple(d.size for d in sample_dimensions)
    max_val = np.iinfo(dtype).max
    original_data = np.random.randint(0, max_val + 1, size=shape, dtype=dtype)

    output_path = (
        tmp_path
        / f"roundtrip_dtype_{stream_cls.__name__.lower()}_{dtype.name}.{file_ext}"
    )

    # Write data using our stream
    stream = stream_cls()
    stream = stream.create(str(output_path), dtype, sample_dimensions)
    assert stream.is_active()

    # Write all frames in the expected order
    nt, nz, nc = shape[:3]
    for t, z, c in product(range(nt), range(nz), range(nc)):
        frame = original_data[t, z, c]
        stream.append(frame)

    stream.flush()
    assert not stream.is_active()
    assert output_path.exists()

    # Read data back and verify it matches
    read_data = read_file_data(output_path)

    # Verify the data matches exactly
    np.testing.assert_array_equal(
        original_data,
        read_data,
        err_msg=f"Data mismatch in {stream_cls.__name__} roundtrip test with {dtype}",
    )

    # Also verify shape and dtype
    assert read_data.shape == original_data.shape, (
        f"Shape mismatch: expected {original_data.shape}, got {read_data.shape}"
    )
    assert read_data.dtype == original_data.dtype, (
        f"Dtype mismatch: expected {original_data.dtype}, got {read_data.dtype}"
    )


# Skip entire test class if no backends are available
if not backends_to_test:
    pytest.skip("No OME writer backends available", allow_module_level=True)
