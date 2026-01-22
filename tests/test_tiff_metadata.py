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
    import tifffile
    from ome_types import from_tiff, from_xml
except ImportError:
    pytest.skip("tifffile or ome_types not installed", allow_module_level=True)


def test_update_metadata_single_file(tmp_path: Path) -> None:
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
        backend="tiff",
    )

    with create_stream(settings) as stream:
        for _ in range(2):
            stream.append(np.random.randint(0, 1000, (32, 32), dtype=np.uint16))

    # Update metadata after context exits (TIFF requires files to be closed)
    metadata = stream.get_metadata()
    metadata.images[0].name = "Updated Image"
    metadata.images[0].pixels.channels[0].name = "Updated Channel"
    stream.update_metadata(metadata)

    # Verify on disk
    with tifffile.TiffFile(str(tmp_path / "test.ome.tiff")) as tif:
        ome_obj = from_xml(tif.ome_metadata)
        assert ome_obj.images[0].name == "Updated Image"
        assert ome_obj.images[0].pixels.channels[0].name == "Updated Channel"


def test_update_metadata_multiposition(tmp_path: Path) -> None:
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
        backend="tiff",
    )

    with create_stream(settings) as stream:
        for _ in range(4):
            stream.append(np.random.randint(0, 1000, (32, 32), dtype=np.uint16))

    # Update metadata after context exits
    metadata = stream.get_metadata()
    metadata.images[0].name = "Position 0 Updated"
    metadata.images[1].name = "Position 1 Updated"
    stream.update_metadata(metadata)

    # Verify each position file
    base_path = tmp_path / "multipos"
    for pos_idx in range(2):
        pos_file = base_path.with_name(f"{base_path.name}_p{pos_idx:03d}.ome.tiff")
        with tifffile.TiffFile(str(pos_file)) as tif:
            ome_obj = from_xml(tif.ome_metadata)
            assert ome_obj.images[0].name == f"Position {pos_idx} Updated"


def test_update_metadata_error_conditions(tmp_path: Path) -> None:
    """Test error conditions in update_metadata method."""
    settings = AcquisitionSettings(
        root_path=str(tmp_path / "error.ome.tiff"),
        dimensions=[
            Dimension(name="y", count=32, type="space"),
            Dimension(name="x", count=32, type="space"),
        ],
        dtype="uint16",
        backend="tiff",
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

    with tifffile.TiffFile(str(tmp_path / "error.ome.tiff")) as tif:
        ome_obj = from_xml(tif.ome_metadata)
        assert ome_obj.images[0].name == "Fixed"


def test_update_metadata_with_plates(tmp_path: Path) -> None:
    """Test update_metadata with plate metadata for multi-position experiments."""
    settings = AcquisitionSettings(
        root_path=str(tmp_path / "plate.ome.tiff"),
        dimensions=[
            PositionDimension(
                positions=[
                    Position(name="Well_A01", row="A", column="1"),
                    Position(name="Well_A02", row="A", column="2"),
                ]
            ),
            Dimension(name="y", count=32, type="space"),
            Dimension(name="x", count=32, type="space"),
        ],
        dtype="uint16",
        backend="tiff",
        plate=Plate(name="Test Plate", row_names=["A"], column_names=["1", "2"]),
    )

    with create_stream(settings) as stream:
        for _ in range(2):
            stream.append(np.random.randint(0, 1000, (32, 32), dtype=np.uint16))

    # Update metadata after context exits
    metadata = stream.get_metadata()
    metadata.images[0].name = "Well A01"
    metadata.images[1].name = "Well A02"
    stream.update_metadata(metadata)

    # Verify each well file has updated name
    base_path = tmp_path / "plate"
    for pos_idx in range(2):
        pos_file = base_path.with_name(f"{base_path.name}_p{pos_idx:03d}.ome.tiff")
        with tifffile.TiffFile(str(pos_file)) as tif:
            ome_obj = from_xml(tif.ome_metadata)
            assert ome_obj.images[0].name == f"Well A0{pos_idx + 1}"


def test_tiff_metadata_physical_sizes_and_acquisition_date(tmp_path: Path) -> None:
    """Test that physical sizes and acquisition date are correctly written."""
    settings = AcquisitionSettings(
        root_path=str(tmp_path / "test_metadata.ome.tiff"),
        dimensions=[
            Dimension(name="c", count=2, type="channel"),
            Dimension(name="z", count=3, type="space", scale=1.0, unit="µm"),
            Dimension(name="t", count=1, type="time"),
            Dimension(name="y", count=64, type="space", scale=0.5, unit="µm"),
            Dimension(name="x", count=64, type="space", scale=0.5, unit="µm"),
        ],
        dtype="uint16",
        backend="tiff",
    )

    with create_stream(settings) as stream:
        for _ in range(6):  # 2 channels * 3 z-slices
            stream.append(np.random.randint(0, 1000, (64, 64), dtype=np.uint16))

    # Read metadata using from_tiff (validates and parses)
    ome_obj = from_tiff(str(tmp_path / "test_metadata.ome.tiff"))

    # Verify physical sizes
    pixels = ome_obj.images[0].pixels
    assert pixels.physical_size_x == 0.5
    assert pixels.physical_size_x_unit.value == "µm"
    assert pixels.physical_size_y == 0.5
    assert pixels.physical_size_y_unit.value == "µm"
    assert pixels.physical_size_z == 1.0
    assert pixels.physical_size_z_unit.value == "µm"

    # Verify acquisition date is present
    assert ome_obj.images[0].acquisition_date is not None
