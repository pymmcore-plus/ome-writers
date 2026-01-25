"""Tests for TIFF backend update_metadata functionality."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest

from ome_writers import (
    AcquisitionSettings,
    Dimension,
    Plate,
    Position,
    PositionDimension,
    create_stream,
)

if TYPE_CHECKING:
    from pathlib import Path

try:
    from ome_types import from_tiff
except ImportError:
    pytest.skip("ome_types not installed", allow_module_level=True)


def test_update_metadata_single_file(tmp_path: Path, tiff_backend: str) -> None:
    """Test update_metadata method for single-file TIFF streams."""
    settings = AcquisitionSettings(
        root_path=str(tmp_path / "test.ome.tiff"),
        dimensions=[
            Dimension(name="t", count=2, type="time"),
            Dimension(name="c", count=1, type="channel"),
            Dimension(name="y", count=32, type="space"),
            Dimension(name="x", count=32, type="space"),
        ],
        dtype="uint16",
        backend=tiff_backend,
    )

    with create_stream(settings) as stream:
        for _ in range(2):
            stream.append(np.random.randint(0, 1000, (32, 32), dtype=np.uint16))

    # Update metadata after context exits
    metadata = stream.get_metadata()
    metadata.images[0].name = "Updated Image"
    metadata.images[0].pixels.channels[0].name = "Updated Channel"
    stream.update_metadata(metadata)

    # Verify on disk
    ome_obj = from_tiff(str(tmp_path / "test.ome.tiff"))
    assert ome_obj.images[0].name == "Updated Image"
    assert ome_obj.images[0].pixels.channels[0].name == "Updated Channel"


def test_update_metadata_multiposition(tmp_path: Path, tiff_backend: str) -> None:
    """Test update_metadata method for multi-position TIFF streams."""
    settings = AcquisitionSettings(
        root_path=str(tmp_path / "multipos.ome.tiff"),
        dimensions=[
            PositionDimension(positions=["Pos0", "Pos1"]),
            Dimension(name="t", count=2, type="time"),
            Dimension(name="y", count=32, type="space"),
            Dimension(name="x", count=32, type="space"),
        ],
        dtype="uint16",
        backend=tiff_backend,
    )

    with create_stream(settings) as stream:
        for _ in range(4):
            stream.append(np.random.randint(0, 1000, (32, 32), dtype=np.uint16))

    # Verify default names are position names
    # Note: Each file contains companion OME-XML with ALL positions,
    # but the actual image data in each file corresponds to its position index
    for pos_idx, pos in enumerate(settings.positions):
        pos_file = tmp_path / f"multipos_p{pos_idx:03d}.ome.tiff"
        ome_obj = from_tiff(str(pos_file))
        # All files have all positions in metadata, check the one that matches this file
        assert len(ome_obj.images) == 2
        assert ome_obj.images[pos_idx].name == pos.name

    # Update metadata
    metadata = stream.get_metadata()
    metadata.images[0].name = "Position 0 Updated"
    metadata.images[1].name = "Position 1 Updated"
    stream.update_metadata(metadata)

    # Verify each position file
    for pos_idx in range(2):
        pos_file = tmp_path / f"multipos_p{pos_idx:03d}.ome.tiff"
        ome_obj = from_tiff(str(pos_file))
        assert ome_obj.images[0].name == f"Position {pos_idx} Updated"


def test_update_metadata_error_conditions(tmp_path: Path, tiff_backend: str) -> None:
    """Test error conditions in update_metadata method."""
    settings = AcquisitionSettings(
        root_path=str(tmp_path / "error.ome.tiff"),
        dimensions=[
            Dimension(name="y", count=32, type="space"),
            Dimension(name="x", count=32, type="space"),
        ],
        dtype="uint16",
        backend=tiff_backend,
    )

    with create_stream(settings) as stream:
        stream.append(np.random.randint(0, 1000, (32, 32), dtype=np.uint16))

    # Invalid metadata type should raise TypeError
    with pytest.raises(TypeError, match=r"Expected ome_types\.model\.OME"):
        stream.update_metadata({"not": "an ome object"})

    # Valid update should work
    metadata = stream.get_metadata()
    metadata.images[0].name = "Fixed"
    stream.update_metadata(metadata)

    ome_obj = from_tiff(str(tmp_path / "error.ome.tiff"))
    assert ome_obj.images[0].name == "Fixed"


def test_update_metadata_with_plates(tmp_path: Path, tiff_backend: str) -> None:
    """Test update_metadata with plate metadata for multi-position experiments."""
    settings = AcquisitionSettings(
        root_path=str(tmp_path / "plate.ome.tiff"),
        dimensions=[
            PositionDimension(
                positions=[
                    Position(name="Well_A01", plate_row="A", plate_column="1"),
                    Position(name="Well_A02", plate_row="A", plate_column="2"),
                ]
            ),
            Dimension(name="y", count=32, type="space"),
            Dimension(name="x", count=32, type="space"),
        ],
        dtype="uint16",
        backend=tiff_backend,
        plate=Plate(name="Test Plate", row_names=["A"], column_names=["1", "2"]),
    )

    with create_stream(settings) as stream:
        for _ in range(2):
            stream.append(np.random.randint(0, 1000, (32, 32), dtype=np.uint16))

    # Verify default names are position names
    # Note: Each file contains companion OME-XML with ALL positions,
    # but the actual image data in each file corresponds to its position index
    for pos_idx, expected_name in enumerate(["Well_A01", "Well_A02"]):
        pos_file = tmp_path / f"plate_p{pos_idx:03d}.ome.tiff"
        ome_obj = from_tiff(str(pos_file))
        # All files have all positions in metadata, check the one that matches this file
        assert len(ome_obj.images) == 2
        assert ome_obj.images[pos_idx].name == expected_name

    # Update metadata
    metadata = stream.get_metadata()
    metadata.images[0].name = "Well A01"
    metadata.images[1].name = "Well A02"
    stream.update_metadata(metadata)

    # Verify each well file has updated name
    for pos_idx in range(2):
        pos_file = tmp_path / f"plate_p{pos_idx:03d}.ome.tiff"
        ome_obj = from_tiff(str(pos_file))
        assert ome_obj.images[0].name == f"Well A0{pos_idx + 1}"


def test_tiff_metadata_physical_sizes_and_names(
    tmp_path: Path, tiff_backend: str
) -> None:
    """Test physical sizes, acquisition date, and image names."""
    settings = AcquisitionSettings(
        root_path=str(tmp_path / "test_metadata.ome.tiff"),
        dimensions=[
            Dimension(name="c", count=2, type="channel"),
            Dimension(name="z", count=3, type="space", scale=1.0, unit="micrometer"),
            Dimension(name="t", count=1, type="time"),
            Dimension(name="y", count=64, type="space", scale=0.5, unit="micrometer"),
            Dimension(name="x", count=64, type="space", scale=0.5, unit="micrometer"),
        ],
        dtype="uint16",
        backend=tiff_backend,
    )

    with create_stream(settings) as stream:
        for _ in range(6):  # 2 channels * 3 z-slices
            stream.append(np.random.randint(0, 1000, (64, 64), dtype=np.uint16))

    ome_obj = from_tiff(str(tmp_path / "test_metadata.ome.tiff"))
    pixels = ome_obj.images[0].pixels

    # Verify physical sizes and units
    assert pixels.physical_size_x == 0.5
    assert pixels.physical_size_x_unit.value == "µm"
    assert pixels.physical_size_y == 0.5
    assert pixels.physical_size_y_unit.value == "µm"
    assert pixels.physical_size_z == 1.0
    assert pixels.physical_size_z_unit.value == "µm"

    # Verify acquisition date
    assert ome_obj.images[0].acquisition_date is not None

    # Verify image name strips .ome extension
    assert ome_obj.images[0].name == "test_metadata"
    assert not ome_obj.images[0].name.endswith(".ome")


def test_tiff_multiposition_detailed_metadata(
    tmp_path: Path, tiff_backend: str
) -> None:
    """Test multiposition files have detailed TiffData blocks with UUIDs."""
    settings = AcquisitionSettings(
        root_path=str(tmp_path / "multipos.ome.tiff"),
        dimensions=[
            PositionDimension(positions=["Pos0", "Pos1"]),
            Dimension(name="c", count=2, type="channel"),
            Dimension(name="z", count=3, type="space", scale=1.0, unit="micrometer"),
            Dimension(name="t", count=1, type="time"),
            Dimension(name="y", count=32, type="space"),
            Dimension(name="x", count=32, type="space"),
        ],
        dtype="uint16",
        backend=tiff_backend,
    )

    with create_stream(settings) as stream:
        for _ in range(12):  # 2 positions * 2 channels * 3 z-slices
            stream.append(np.random.randint(0, 1000, (32, 32), dtype=np.uint16))

    # Check all position files contain the same detailed metadata
    for p_idx in range(len(settings.positions)):
        ome_obj = from_tiff(tmp_path / f"multipos_p{p_idx:03}.ome.tiff")

        # Should contain metadata for both positions (companion OME-XML)
        assert len(ome_obj.images) == 2
        assert [img.name for img in ome_obj.images] == ["Pos0", "Pos1"]
        assert [img.id for img in ome_obj.images] == ["Image:0", "Image:1"]

        for img_idx, image in enumerate(ome_obj.images):
            for td in image.pixels.tiff_data_blocks:
                assert td.plane_count == 6
                assert td.uuid is not None
                assert td.uuid.value.startswith("urn:uuid:")
                assert f"multipos_p{img_idx:03}.ome.tiff" in (td.uuid.file_name or "")
