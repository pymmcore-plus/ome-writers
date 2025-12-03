"""Tests for the memmap-based TifffileStream implementation.

This module tests the memmap-based TIFF writer to ensure it follows the same
patterns as the thread-based TIFF writer while providing memory-efficient writing.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pytest

import ome_writers as omew

if TYPE_CHECKING:
    import ome_types.model as ome

# Check if dependencies are available before importing
try:
    import ome_types.model  # noqa: F401
    import tifffile  # noqa: F401

    DEPS_AVAILABLE = True
except ImportError:
    DEPS_AVAILABLE = False

if not DEPS_AVAILABLE:
    pytest.skip("tifffile or ome-types not available", allow_module_level=True)

# Import the memmap-based stream
from ome_writers.backends._tifffile_memmap import TifffileStream as MemmapStream


def _read_tiff(output_path: Path) -> np.ndarray:
    """Helper to read TIFF files."""
    import tifffile

    return tifffile.imread(str(output_path))


def test_memmap_stream_is_available() -> None:
    """Test that the memmap stream is available when dependencies are installed."""
    assert MemmapStream.is_available()


def test_memmap_basic_write(tmp_path: Path) -> None:
    """Test basic write functionality with memmap stream."""
    import ome_types

    data_gen, dimensions, dtype = omew.fake_data_for_sizes(
        sizes={"t": 3, "c": 2, "z": 2, "y": 64, "x": 64},
        chunk_sizes={"y": 32, "x": 32},
    )

    output_path = tmp_path / "test_memmap.ome.tiff"
    stream = MemmapStream()

    stream = stream.create(str(output_path), dtype, dimensions)
    assert stream.is_active()

    # Write all frames
    original_frames = list(data_gen)
    for frame in original_frames:
        stream.append(frame)

    stream.flush()
    assert not stream.is_active()
    assert output_path.exists()

    # Verify data integrity
    disk_data = _read_tiff(output_path)
    shape = tuple(d.size for d in dimensions)
    original_data = np.array(original_frames).reshape(shape)
    np.testing.assert_array_equal(original_data, disk_data)

    # Validate OME-XML metadata
    import tifffile

    with tifffile.TiffFile(str(output_path)) as tif:
        ome_xml = tif.pages[0].description
        ome_metadata = ome_types.from_xml(ome_xml)
        assert len(ome_metadata.images) == 1
        assert ome_metadata.images[0].pixels.size_x == 64
        assert ome_metadata.images[0].pixels.size_y == 64


def test_memmap_minimal_2d(tmp_path: Path) -> None:
    """Test with minimal 2D dimensions."""
    import ome_types

    data_gen, dimensions, dtype = omew.fake_data_for_sizes(
        sizes={"t": 1, "y": 32, "x": 32},
        chunk_sizes={"t": 1, "y": 16, "x": 16},
        dtype=np.uint8,
    )

    output_path = tmp_path / "test_2d_memmap.ome.tiff"
    stream = MemmapStream()

    stream = stream.create(str(output_path), dtype, dimensions)
    assert stream.is_active()

    for data in data_gen:
        stream.append(data)

    stream.flush()
    assert not stream.is_active()
    assert output_path.exists()

    # Validate OME-XML metadata
    import tifffile

    with tifffile.TiffFile(str(output_path)) as tif:
        ome_xml = tif.pages[0].description
        ome_metadata = ome_types.from_xml(ome_xml)
        assert len(ome_metadata.images) == 1
        assert ome_metadata.images[0].pixels.type.value == "uint8"


def test_memmap_custom_flush_interval(tmp_path: Path) -> None:
    """Test that custom flush interval works."""
    data_gen, dimensions, dtype = omew.fake_data_for_sizes(
        sizes={"t": 10, "y": 32, "x": 32},
        chunk_sizes={"y": 16, "x": 16},
    )

    output_path = tmp_path / "test_flush_interval.ome.tiff"
    # Create stream with custom flush interval
    stream = MemmapStream(flush_interval=5)

    stream = stream.create(str(output_path), dtype, dimensions)
    assert stream.is_active()

    for frame in data_gen:
        stream.append(frame)

    stream.flush()
    assert not stream.is_active()
    assert output_path.exists()


def test_memmap_multiposition(tmp_path: Path) -> None:
    """Test multi-position acquisition with memmap stream."""
    import ome_types

    data_gen, dimensions, dtype = omew.fake_data_for_sizes(
        sizes={"t": 3, "c": 2, "z": 2, "y": 32, "x": 32, "p": 3},
        chunk_sizes={"y": 16, "x": 16},
    )

    output_path = tmp_path / "test_multipos_memmap.ome.tiff"
    stream = MemmapStream()

    stream = stream.create(str(output_path), dtype, dimensions)
    assert stream.is_active()

    # Write all frames for all positions
    for frame in data_gen:
        stream.append(frame)

    stream.flush()
    assert not stream.is_active()

    # Verify that separate TIFF files were created for each position
    base_path = Path(str(output_path).replace(".ome.tiff", ""))
    for pos_idx in range(3):
        pos_file = base_path.with_name(f"{base_path.name}_p{pos_idx:03d}.ome.tiff")
        assert pos_file.exists(), f"Position file {pos_file} not found"

        # Verify shape
        data = _read_tiff(pos_file)
        expected_shape = (3, 2, 2, 32, 32)  # (t, c, z, y, x)
        assert data.shape == expected_shape

        # Validate OME-XML metadata for each position
        import tifffile

        with tifffile.TiffFile(str(pos_file)) as tif:
            ome_xml = tif.pages[0].description
            ome_metadata = ome_types.from_xml(ome_xml)
            assert len(ome_metadata.images) == 1
            assert ome_metadata.images[0].pixels.size_t == 3
            assert ome_metadata.images[0].pixels.size_c == 2
            assert ome_metadata.images[0].pixels.size_z == 2


def test_memmap_overwrite_behavior(tmp_path: Path) -> None:
    """Test overwrite behavior."""
    data_gen, dimensions, dtype = omew.fake_data_for_sizes(
        sizes={"t": 2, "y": 32, "x": 32},
        chunk_sizes={"y": 16, "x": 16},
    )

    output_path = tmp_path / "test_overwrite.ome.tiff"

    # First write
    stream = MemmapStream()
    stream = stream.create(str(output_path), dtype, dimensions)
    for frame in data_gen:
        stream.append(frame)
    stream.flush()
    assert output_path.exists()

    # Try to create again without overwrite (should fail)
    data_gen, dimensions, dtype = omew.fake_data_for_sizes(
        sizes={"t": 2, "y": 32, "x": 32},
        chunk_sizes={"y": 16, "x": 16},
    )
    stream = MemmapStream()
    with pytest.raises(FileExistsError, match=r".*already exists"):
        stream.create(str(output_path), dtype, dimensions, overwrite=False)

    # Create again with overwrite=True (should succeed)
    data_gen, dimensions, dtype = omew.fake_data_for_sizes(
        sizes={"t": 2, "y": 32, "x": 32},
        chunk_sizes={"y": 16, "x": 16},
    )
    stream = MemmapStream()
    stream = stream.create(str(output_path), dtype, dimensions, overwrite=True)
    assert stream.is_active()
    for frame in data_gen:
        stream.append(frame)
    stream.flush()
    assert not stream.is_active()


def test_memmap_error_handling() -> None:
    """Test error handling in memmap stream."""
    stream = MemmapStream()

    expected_message = "Stream is closed or uninitialized"
    test_frame = np.zeros((64, 64), dtype=np.uint16)
    with pytest.raises(RuntimeError, match=expected_message):
        stream.append(test_frame)


def test_memmap_context_manager(tmp_path: Path) -> None:
    """Test context manager support."""
    data_gen, dimensions, dtype = omew.fake_data_for_sizes(
        sizes={"t": 2, "y": 32, "x": 32},
        chunk_sizes={"y": 16, "x": 16},
    )

    output_path = tmp_path / "test_context.ome.tiff"
    stream = MemmapStream()

    with stream.create(str(output_path), dtype, dimensions) as s:
        assert s.is_active()
        for frame in data_gen:
            s.append(frame)

    # After context exit, stream should be flushed
    assert not stream.is_active()
    assert output_path.exists()


def test_memmap_update_ome_metadata(tmp_path: Path) -> None:
    """Test updating OME metadata after flush."""
    import ome_types

    data_gen, dimensions, dtype = omew.fake_data_for_sizes(
        sizes={"t": 2, "c": 2, "y": 32, "x": 32},
        chunk_sizes={"y": 16, "x": 16},
    )

    output_path = tmp_path / "test_metadata.ome.tiff"
    stream = MemmapStream()
    stream = stream.create(str(output_path), dtype, dimensions)

    for frame in data_gen:
        stream.append(frame)

    stream.flush()

    # Create comprehensive OME metadata
    metadata = _create_test_ome_metadata()

    # Update the metadata
    stream.update_ome_metadata(metadata)

    # Read back and verify metadata was written
    import tifffile

    with tifffile.TiffFile(str(output_path)) as tif:
        ome_xml = tif.pages[0].description
        # Verify basic OME structure is present
        assert "Image:0" in ome_xml
        assert "Pixels:0" in ome_xml
        # Validate that the XML is valid OME-XML
        ome_metadata = ome_types.from_xml(ome_xml)
        assert len(ome_metadata.images) == 1
        assert ome_metadata.images[0].id == "Image:0"


def test_memmap_multiposition_metadata_update(tmp_path: Path) -> None:
    """Test updating OME metadata for multi-position datasets."""
    import ome_types

    data_gen, dimensions, dtype = omew.fake_data_for_sizes(
        sizes={"t": 2, "c": 2, "y": 32, "x": 32, "p": 2},
        chunk_sizes={"y": 16, "x": 16},
    )

    output_path = tmp_path / "test_multipos_metadata.ome.tiff"
    stream = MemmapStream()
    stream = stream.create(str(output_path), dtype, dimensions)

    for frame in data_gen:
        stream.append(frame)

    stream.flush()

    # Create comprehensive OME metadata with multiple images
    metadata = _create_multiposition_ome_metadata()

    # Update the metadata
    stream.update_ome_metadata(metadata)

    # Verify metadata was written to each position file
    import tifffile

    base_path = Path(str(output_path).replace(".ome.tiff", ""))
    for pos_idx in range(2):
        pos_file = base_path.with_name(f"{base_path.name}_p{pos_idx:03d}.ome.tiff")
        with tifffile.TiffFile(str(pos_file)) as tif:
            ome_xml = tif.pages[0].description
            assert f"Image:{pos_idx}" in ome_xml
            # Validate OME-XML
            ome_metadata = ome_types.from_xml(ome_xml)
            assert len(ome_metadata.images) == 1
            assert ome_metadata.images[0].id == f"Image:{pos_idx}"


def test_memmap_different_dtypes(tmp_path: Path) -> None:
    """Test with different data types."""
    import ome_types

    for dtype_name, np_dtype in [("uint8", np.uint8), ("uint16", np.uint16)]:
        data_gen, dimensions, dtype = omew.fake_data_for_sizes(
            sizes={"t": 2, "y": 32, "x": 32},
            chunk_sizes={"y": 16, "x": 16},
            dtype=np_dtype,
        )

        output_path = tmp_path / f"test_{dtype_name}_memmap.ome.tiff"
        stream = MemmapStream()
        stream = stream.create(str(output_path), dtype, dimensions)

        original_frames = list(data_gen)
        for frame in original_frames:
            stream.append(frame)

        stream.flush()

        # Verify data integrity and dtype
        disk_data = _read_tiff(output_path)
        assert disk_data.dtype == np_dtype

        # Validate OME-XML metadata
        import tifffile

        with tifffile.TiffFile(str(output_path)) as tif:
            ome_xml = tif.pages[0].description
            ome_metadata = ome_types.from_xml(ome_xml)
            assert ome_metadata.images[0].pixels.type.value == dtype_name


def test_memmap_file_extensions(tmp_path: Path) -> None:
    """Test that various TIFF file extensions are handled correctly."""
    data_gen, dimensions, dtype = omew.fake_data_for_sizes(
        sizes={"t": 1, "y": 32, "x": 32},
        chunk_sizes={"y": 16, "x": 16},
    )

    for ext in [".tif", ".tiff", ".ome.tif", ".ome.tiff"]:
        output_path = tmp_path / f"test{ext}"
        stream = MemmapStream()
        stream = stream.create(str(output_path), dtype, dimensions)

        for frame in data_gen:
            stream.append(frame)

        stream.flush()
        assert output_path.exists()

        # Reset generator for next iteration
        data_gen, dimensions, dtype = omew.fake_data_for_sizes(
            sizes={"t": 1, "y": 32, "x": 32},
            chunk_sizes={"y": 16, "x": 16},
        )


def test_memmap_metadata_update_error_handling(tmp_path: Path) -> None:
    """Test error handling during metadata updates."""
    import ome_types.model as ome

    data_gen, dimensions, dtype = omew.fake_data_for_sizes(
        sizes={"t": 2, "c": 2, "y": 32, "x": 32},
        chunk_sizes={"y": 16, "x": 16},
    )

    output_path = tmp_path / "test_error.ome.tiff"
    stream = MemmapStream()
    stream = stream.create(str(output_path), dtype, dimensions)

    for frame in data_gen:
        stream.append(frame)

    stream.flush()

    # Try to update with metadata missing the expected image
    # This should raise a RuntimeError
    pixels = ome.Pixels(
        id="Pixels:999",
        dimension_order="XYCZT",
        size_x=32,
        size_y=32,
        size_c=2,
        size_z=1,
        size_t=2,
        type="uint16",
        metadata_only=True,
    )
    image = ome.Image(id="Image:999", pixels=pixels)  # Wrong ID
    bad_metadata = ome.OME(images=[image])

    with pytest.raises(
        RuntimeError, match="Failed to create position-specific OME metadata"
    ):
        stream.update_ome_metadata(bad_metadata)


def test_memmap_invalid_metadata_type(tmp_path: Path) -> None:
    """Test that update_ome_metadata rejects non-OME metadata."""
    data_gen, dimensions, dtype = omew.fake_data_for_sizes(
        sizes={"t": 2, "y": 32, "x": 32},
        chunk_sizes={"y": 16, "x": 16},
    )

    output_path = tmp_path / "test_invalid_metadata.ome.tiff"
    stream = MemmapStream()
    stream = stream.create(str(output_path), dtype, dimensions)

    for frame in data_gen:
        stream.append(frame)

    stream.flush()

    # Try to update with invalid metadata type
    with pytest.raises(TypeError, match="Expected OME metadata"):
        stream.update_ome_metadata({"not": "ome"})  # type: ignore


def test_memmap_metadata_update_file_permissions(tmp_path: Path) -> None:
    """Test error handling when file is not writable during metadata update."""
    import os

    data_gen, dimensions, dtype = omew.fake_data_for_sizes(
        sizes={"t": 2, "y": 32, "x": 32},
        chunk_sizes={"y": 16, "x": 16},
    )

    output_path = tmp_path / "test_readonly.ome.tiff"
    stream = MemmapStream()
    stream = stream.create(str(output_path), dtype, dimensions)

    for frame in data_gen:
        stream.append(frame)

    stream.flush()

    # Make the file read-only
    os.chmod(output_path, 0o444)

    try:
        # Create valid metadata
        metadata = _create_test_ome_metadata()

        # Try to update metadata on read-only file - should raise RuntimeError
        with pytest.raises(RuntimeError, match="Failed to update OME metadata"):
            stream.update_ome_metadata(metadata)
    finally:
        # Restore write permissions for cleanup
        os.chmod(output_path, 0o644)


# Helper functions for metadata testing


def _create_test_ome_metadata() -> ome.OME:
    """Create a simple OME metadata object for testing."""
    import ome_types.model as ome

    # Create a simple image
    pixels = ome.Pixels(
        id="Pixels:0",
        dimension_order="XYCZT",
        size_x=32,
        size_y=32,
        size_c=2,
        size_z=1,
        size_t=2,
        type="uint16",
        metadata_only=True,
    )

    image = ome.Image(id="Image:0", pixels=pixels)

    return ome.OME(images=[image])


def _create_multiposition_ome_metadata() -> ome.OME:
    """Create OME metadata with multiple positions for testing."""
    import ome_types.model as ome

    images = []
    for pos_idx in range(2):
        pixels = ome.Pixels(
            id=f"Pixels:{pos_idx}",
            dimension_order="XYCZT",
            size_x=32,
            size_y=32,
            size_c=2,
            size_z=1,
            size_t=2,
            type="uint16",
            metadata_only=True,
        )
        image = ome.Image(id=f"Image:{pos_idx}", pixels=pixels)
        images.append(image)

    return ome.OME(images=images)
