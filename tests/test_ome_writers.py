"""Tests for ome-writers library."""

from __future__ import annotations

import importlib
import importlib.util
from itertools import product
from typing import TYPE_CHECKING

import numpy as np
import pytest

from ome_writers import (
    AcquireZarrStream,
    DimensionInfo,
    TensorStoreZarrStream,
    TiffStream,
)

if TYPE_CHECKING:
    from pathlib import Path

    from ome_writers import OMEStream

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


@pytest.fixture
def sample_data(sample_dimensions: list[DimensionInfo]) -> np.ndarray:
    """Create sample test data."""
    shape = tuple(d.size for d in sample_dimensions)
    return np.random.randint(0, 65536, size=shape, dtype=np.uint16)


@pytest.mark.parametrize("stream_cls,file_ext", backends_to_test)
def test_basic_stream_lifecycle(
    stream_cls: type[OMEStream],
    file_ext: str,
    tmp_path: Path,
    sample_dimensions: list[DimensionInfo],
    sample_data: np.ndarray,
) -> None:
    """Test basic stream creation, writing, and closing."""
    # Import and create the appropriate stream
    stream = stream_cls()

    # Set output path
    output_path = tmp_path / f"test_output_{stream_cls.__name__.lower()}.{file_ext}"

    # Create stream
    stream = stream.create(str(output_path), sample_data.dtype, sample_dimensions)

    # Check that stream is active
    assert stream.is_active()

    # Write all frames
    nt, nz, nc, ny, nx = sample_data.shape
    for t, z, c in product(range(nt), range(nz), range(nc)):
        frame = sample_data[t, z, c]
        stream.append(frame)

    # Flush the stream
    stream.flush()

    # Check that stream is no longer active after flush
    assert not stream.is_active()

    # Verify file was created
    assert output_path.exists()


@pytest.mark.parametrize("stream_cls,file_ext", backends_to_test)
def test_stream_with_different_dtypes(
    stream_cls: type[OMEStream],
    file_ext: str,
    tmp_path: Path,
    sample_dimensions: list[DimensionInfo],
) -> None:
    """Test stream creation with different data types."""
    dtypes_to_test = [np.uint8, np.uint16, np.float32]

    for dtype_class in dtypes_to_test:
        # Create the stream
        stream = stream_cls()

        # Set output path
        output_path = (
            tmp_path
            / f"test_{stream_cls.__name__.lower()}_{dtype_class.__name__}.{file_ext}"
        )

        # Create test data with the specific dtype
        shape = tuple(d.size for d in sample_dimensions)
        if dtype_class == np.float32:
            data = np.random.random(shape).astype(dtype_class)
        elif dtype_class == np.uint8:
            data = np.random.randint(0, 255, size=shape, dtype=np.uint8)
        elif dtype_class == np.uint16:
            data = np.random.randint(0, 65535, size=shape, dtype=np.uint16)
        else:
            # Fallback for any other types
            data = np.random.randint(0, 100, size=shape).astype(dtype_class)

        # Create and use stream
        dtype_instance = np.dtype(dtype_class)
        stream = stream.create(str(output_path), dtype_instance, sample_dimensions)
        assert stream.is_active()

        # Write a few frames
        nt, nz, nc = shape[:3]
        for t in range(min(2, nt)):
            for z in range(min(2, nz)):
                for c in range(min(2, nc)):
                    frame = data[t, z, c]
                    stream.append(frame)

        stream.flush()
        assert not stream.is_active()
        assert output_path.exists()


@pytest.mark.parametrize("stream_cls,file_ext", backends_to_test)
def test_minimal_2d_dimensions(
    stream_cls: type[OMEStream], file_ext: str, tmp_path: Path
) -> None:
    """Test with minimal 2D dimensions (just x and y)."""
    dimensions = [
        DimensionInfo(label="y", size=32, unit=(1.0, "um"), chunk_size=16),
        DimensionInfo(label="x", size=32, unit=(1.0, "um"), chunk_size=16),
    ]

    # Skip acquire-zarr for 2D as it requires at least 3 dimensions
    if stream_cls.__name__ == "AcquireZarrStream":
        pytest.skip("acquire-zarr requires at least 3 dimensions")

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


@pytest.mark.skipif(
    not any(cls.__name__ == "AcquireZarrStream" for cls, _ in backends_to_test),
    reason="acquire-zarr not available",
)
def test_create_stream_factory_function(
    tmp_path: Path, sample_dimensions: list[DimensionInfo]
) -> None:
    """Test the create_stream factory function."""
    from ome_writers import create_stream

    output_path = tmp_path / "factory_test.zarr"

    # The default create_stream uses AcquireZarrStream
    stream = create_stream(str(output_path), np.dtype(np.uint16), sample_dimensions)
    assert stream.is_active()

    # Write a test frame
    test_frame = np.random.randint(0, 65536, size=(64, 64), dtype=np.uint16)
    stream.append(test_frame)

    stream.flush()
    assert not stream.is_active()
    assert output_path.exists()


# Skip entire test class if no backends are available
if not backends_to_test:
    pytest.skip("No OME writer backends available", allow_module_level=True)
