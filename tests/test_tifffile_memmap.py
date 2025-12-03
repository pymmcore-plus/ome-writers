"""Tests for the memmap-based TifffileStream implementation."""

from __future__ import annotations

import os

import numpy as np
import pytest

import ome_writers as omew

try:
    import ome_types
    import tifffile
except ImportError:
    pytest.skip("tifffile or ome-types not available", allow_module_level=True)

from typing import TYPE_CHECKING

from ome_writers.backends._tifffile_memmap import TifffileStream as MemmapStream

if TYPE_CHECKING:
    from pathlib import Path


def test_memmap_stream_is_available() -> None:
    """Test that the memmap stream is available when dependencies are installed."""
    assert MemmapStream.is_available()


@pytest.mark.parametrize(
    ("sizes", "np_dtype"),
    [
        ({"t": 3, "c": 2, "z": 2, "y": 64, "x": 64}, np.uint16),  # full 5D
        ({"t": 1, "y": 32, "x": 32}, np.uint8),  # minimal 2D
        ({"t": 2, "y": 32, "x": 32}, np.uint16),  # different dtype
    ],
    ids=["5d-uint16", "2d-uint8", "3d-uint16"],
)
def test_memmap_write(sizes: dict[str, int], np_dtype: type, tmp_path: Path) -> None:
    """Test write functionality with various dimensions and dtypes."""
    data_gen, dims, dtype = omew.fake_data_for_sizes(
        sizes=sizes, chunk_sizes={"y": 16, "x": 16}, dtype=np_dtype
    )
    output_path = tmp_path / "test.ome.tiff"

    stream = MemmapStream().create(str(output_path), dtype, dims)
    original_frames = list(data_gen)
    for frame in original_frames:
        stream.append(frame)
    stream.flush()

    # Verify data integrity
    disk_data = tifffile.imread(str(output_path))
    original_data = np.array(original_frames).reshape(tuple(d.size for d in dims))
    np.testing.assert_array_equal(original_data, disk_data)
    assert disk_data.dtype == np_dtype

    # Validate OME-XML
    ome = ome_types.from_tiff(str(output_path))
    assert len(ome.images) == 1
    first_image = ome.images[0]
    assert first_image.pixels.size_t == sizes.get("t", 1)
    assert first_image.pixels.size_c == sizes.get("c", 1)
    assert first_image.pixels.size_z == sizes.get("z", 1)
    assert first_image.pixels.size_y == sizes["y"]
    assert first_image.pixels.size_x == sizes["x"]


def test_memmap_custom_flush_interval(tmp_path: Path) -> None:
    """Test that custom flush interval works."""
    data_gen, dims, dtype = omew.fake_data_for_sizes(
        sizes={"t": 10, "y": 32, "x": 32}, chunk_sizes={"y": 16, "x": 16}
    )
    output_path = tmp_path / "test.ome.tiff"

    stream = MemmapStream(flush_interval=5).create(str(output_path), dtype, dims)
    for frame in data_gen:
        stream.append(frame)
    stream.flush()
    assert output_path.exists()


def test_memmap_multiposition(tmp_path: Path) -> None:
    """Test multi-position acquisition with metadata validation."""
    data_gen, dims, dtype = omew.fake_data_for_sizes(
        sizes={"t": 3, "c": 2, "z": 2, "y": 32, "x": 32, "p": 3},
        chunk_sizes={"y": 16, "x": 16},
    )
    output_path = tmp_path / "test.ome.tiff"

    stream = MemmapStream().create(str(output_path), dtype, dims)
    for frame in data_gen:
        stream.append(frame)
    stream.flush()

    # Update with multi-position metadata
    images = [
        ome_types.model.Image(
            id=f"Image:{i}",
            pixels=ome_types.model.Pixels(
                id=f"Pixels:{i}",
                dimension_order="XYCZT",
                size_x=32,
                size_y=32,
                size_c=2,
                size_z=2,
                size_t=3,
                type="uint16",
                metadata_only=True,
            ),
        )
        for i in range(3)
    ]
    stream.update_ome_metadata(ome_types.model.OME(images=images))

    # Verify each position file
    base = tmp_path / "test"
    for i in range(3):
        pos_file = base.with_name(f"{base.name}_p{i:03d}.ome.tiff")
        assert pos_file.exists()
        data = tifffile.imread(str(pos_file))
        assert data.shape == (3, 2, 2, 32, 32)

        ome = ome_types.from_tiff(pos_file)
        assert ome.images[0].id == f"Image:{i}"


def test_memmap_overwrite_behavior(tmp_path: Path) -> None:
    """Test overwrite behavior."""
    sizes = {"t": 2, "y": 32, "x": 32}
    output_path = tmp_path / "test.ome.tiff"

    # First write
    data_gen, dims, dtype = omew.fake_data_for_sizes(
        sizes, chunk_sizes={"y": 16, "x": 16}
    )
    stream = MemmapStream().create(str(output_path), dtype, dims)
    for frame in data_gen:
        stream.append(frame)
    stream.flush()

    # Without overwrite should fail
    with pytest.raises(FileExistsError, match=r".*already exists"):
        MemmapStream().create(str(output_path), dtype, dims, overwrite=False)

    # With overwrite should succeed
    data_gen, dims, dtype = omew.fake_data_for_sizes(
        sizes, chunk_sizes={"y": 16, "x": 16}
    )
    stream = MemmapStream().create(str(output_path), dtype, dims, overwrite=True)
    for frame in data_gen:
        stream.append(frame)
    stream.flush()


def test_memmap_error_handling() -> None:
    """Test error handling in memmap stream."""
    with pytest.raises(RuntimeError, match="Stream is closed or uninitialized"):
        MemmapStream().append(np.zeros((64, 64), dtype=np.uint16))


def test_memmap_context_manager(tmp_path: Path) -> None:
    """Test context manager support."""
    data_gen, dims, dtype = omew.fake_data_for_sizes(
        sizes={"t": 2, "y": 32, "x": 32}, chunk_sizes={"y": 16, "x": 16}
    )
    output_path = tmp_path / "test.ome.tiff"
    stream = MemmapStream()

    with stream.create(str(output_path), dtype, dims) as s:
        assert s.is_active()
        for frame in data_gen:
            s.append(frame)

    assert not stream.is_active()
    assert output_path.exists()


def test_memmap_update_ome_metadata(tmp_path: Path) -> None:
    """Test updating OME metadata after flush."""
    data_gen, dims, dtype = omew.fake_data_for_sizes(
        sizes={"t": 2, "c": 2, "y": 32, "x": 32}, chunk_sizes={"y": 16, "x": 16}
    )
    output_path = tmp_path / "test.ome.tiff"

    stream = MemmapStream().create(str(output_path), dtype, dims)
    for frame in data_gen:
        stream.append(frame)
    stream.flush()

    # Valid metadata update
    metadata = ome_types.model.OME(
        images=[
            ome_types.model.Image(
                id="Image:0",
                pixels=ome_types.model.Pixels(
                    id="Pixels:0",
                    dimension_order="XYCZT",
                    size_x=32,
                    size_y=32,
                    size_c=2,
                    size_z=1,
                    size_t=2,
                    type="uint16",
                    metadata_only=True,
                ),
            )
        ]
    )
    stream.update_ome_metadata(metadata)

    ome = ome_types.from_tiff(output_path)
    assert ome.images[0].id == "Image:0"


@pytest.mark.parametrize(
    ("bad_metadata", "error_type", "error_match"),
    [
        ({"not": "ome"}, TypeError, "Expected OME metadata"),
        pytest.param(
            "wrong_image_id",  # sentinel for creating wrong ID metadata
            RuntimeError,
            "Failed to create position-specific OME metadata",
            id="wrong-image-id",
        ),
    ],
)
def test_memmap_metadata_update_errors(
    bad_metadata: object, error_type: type, error_match: str, tmp_path: Path
) -> None:
    """Test error handling during metadata updates."""
    data_gen, dims, dtype = omew.fake_data_for_sizes(
        sizes={"t": 2, "c": 2, "y": 32, "x": 32}, chunk_sizes={"y": 16, "x": 16}
    )
    output_path = tmp_path / "test.ome.tiff"

    stream = MemmapStream().create(str(output_path), dtype, dims)
    for frame in data_gen:
        stream.append(frame)
    stream.flush()

    if bad_metadata == "wrong_image_id":
        bad_metadata = ome_types.model.OME(
            images=[
                ome_types.model.Image(
                    id="Image:999",
                    pixels=ome_types.model.Pixels(
                        id="Pixels:999",
                        dimension_order="XYCZT",
                        size_x=32,
                        size_y=32,
                        size_c=2,
                        size_z=1,
                        size_t=2,
                        type="uint16",
                        metadata_only=True,
                    ),
                )
            ]
        )

    with pytest.raises(error_type, match=error_match):
        stream.update_ome_metadata(bad_metadata)  # type: ignore[arg-type]


def test_memmap_metadata_update_readonly_file(tmp_path: Path) -> None:
    """Test error handling when file is not writable."""
    data_gen, dims, dtype = omew.fake_data_for_sizes(
        sizes={"t": 2, "y": 32, "x": 32}, chunk_sizes={"y": 16, "x": 16}
    )
    output_path = tmp_path / "test.ome.tiff"

    stream = MemmapStream().create(str(output_path), dtype, dims)
    for frame in data_gen:
        stream.append(frame)
    stream.flush()

    os.chmod(output_path, 0o444)
    try:
        metadata = ome_types.model.OME(
            images=[
                ome_types.model.Image(
                    id="Image:0",
                    pixels=ome_types.model.Pixels(
                        id="Pixels:0",
                        dimension_order="XYCZT",
                        size_x=32,
                        size_y=32,
                        size_c=2,
                        size_z=1,
                        size_t=2,
                        type="uint16",
                        metadata_only=True,
                    ),
                )
            ]
        )
        with pytest.raises(RuntimeError, match="Failed to update OME metadata"):
            stream.update_ome_metadata(metadata)
    finally:
        os.chmod(output_path, 0o644)


def test_memmap_file_extensions(tmp_path: Path) -> None:
    """Test that various TIFF file extensions are handled correctly."""
    data_gen, dims, dtype = omew.fake_data_for_sizes(
        sizes={"t": 1, "y": 32, "x": 32}, chunk_sizes={"y": 16, "x": 16}
    )
    frames = list(data_gen)

    for ext in [".tif", ".tiff", ".ome.tif", ".ome.tiff"]:
        output_path = tmp_path / f"test{ext}"
        stream = MemmapStream().create(str(output_path), dtype, dims)
        for frame in frames:
            stream.append(frame)
        stream.flush()
        assert output_path.exists()
