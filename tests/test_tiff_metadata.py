"""Tests for TIFF backend update_metadata functionality."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import ome_types.model as ome
import pytest
import tifffile
from ome_types import from_xml

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


pytest.importorskip("tifffile", reason="tifffile not available")
pytest.importorskip("ome_types", reason="ome_types not available")


def create_metadata(
    *,
    image_name: str,
    channel_name: str = "Test Channel",
    dtype: str = "uint16",
    size_t: int = 1,
    size_c: int = 1,
    size_z: int = 1,
    size_x: int = 32,
    size_y: int = 32,
    num_images: int = 1,
    include_plate: bool = False,
) -> ome.OME:
    """Create OME metadata object with customizable parameters."""
    if include_plate:
        plates = [
            ome.Plate(
                id="Plate:0",
                name="Test Plate",
                wells=[
                    ome.Well(
                        id=f"Well:{i}",
                        row=0,
                        column=i,
                        well_samples=[
                            ome.WellSample(
                                id=f"WellSample:{i}",
                                index=0,
                                image_ref=ome.ImageRef(id=f"Image:{i}"),
                            )
                        ],
                    )
                    for i in range(num_images)
                ],
            )
        ]
    else:
        plates = []

    return ome.OME(
        images=[
            ome.Image(
                id=f"Image:{i}",
                name=image_name.replace("Position 0", f"Position {i}"),
                pixels=ome.Pixels(
                    id=f"Pixels:{i}",
                    type=dtype,
                    size_x=size_x,
                    size_y=size_y,
                    size_z=size_z,
                    size_c=size_c,
                    size_t=size_t,
                    dimension_order="XYCZT",
                    channels=[
                        ome.Channel(
                            id=f"Channel:{i}",
                            name=channel_name.replace("P0", f"P{i}"),
                            samples_per_pixel=1,
                        )
                    ],
                ),
            )
            for i in range(num_images)
        ],
        creator="ome_writers test suite",
        plates=plates,
    )


def test_update_metadata_single_file(tmp_path: Path) -> None:
    """Test update_metadata method for single-file TIFF streams."""
    # Create acquisition settings for a simple 2D+T dataset
    settings = AcquisitionSettings(
        root_path=str(tmp_path / "test_update_metadata.ome.tiff"),
        dimensions=[
            Dimension(name="t", count=2, type="time"),
            Dimension(name="c", count=1, type="channel"),
            Dimension(name="z", count=1, type="space"),
            Dimension(name="y", count=32, chunk_size=16, type="space"),
            Dimension(name="x", count=32, chunk_size=16, type="space"),
        ],
        dtype="uint16",
        overwrite=True,
        backend="tiff",
    )

    # Create and write data
    with create_stream(settings) as stream:
        for _i in range(2):  # 2 time points
            frame = np.random.randint(0, 1000, (32, 32), dtype=np.uint16)
            stream.append(frame)

    # NEW API: Get auto-generated metadata instead of recreating it
    metadata = stream._backend.get_metadata()
    assert metadata is not None
    assert len(metadata.images) == 1

    # Modify the metadata
    metadata.images[0].name = "Updated Test Image"
    metadata.images[0].pixels.channels[0].name = "Updated Channel"

    # Update metadata after writing
    stream._backend.update_metadata(metadata)

    # Verify the metadata was updated by reading it back
    output_path = tmp_path / "test_update_metadata.ome.tiff"
    with tifffile.TiffFile(str(output_path)) as tif:
        ome_xml = tif.ome_metadata
        assert ome_xml is not None
        assert "Updated Test Image" in ome_xml
        assert "Updated Channel" in ome_xml

        # Also verify we can parse the OME-XML
        ome_obj = from_xml(ome_xml)
        assert ome_obj.images[0].name == "Updated Test Image"
        assert ome_obj.images[0].pixels.channels[0].name == "Updated Channel"


def test_update_metadata_multiposition(tmp_path: Path) -> None:
    """Test update_metadata method for multi-position TIFF streams."""

    # Create acquisition settings with 2 positions
    settings = AcquisitionSettings(
        root_path=str(tmp_path / "test_update_multipos.ome.tiff"),
        dimensions=[
            PositionDimension(positions=["Pos0", "Pos1"]),
            Dimension(name="t", count=2, type="time"),
            Dimension(name="c", count=1, type="channel"),
            Dimension(name="z", count=1, type="space"),
            Dimension(name="y", count=32, chunk_size=16, type="space"),
            Dimension(name="x", count=32, chunk_size=16, type="space"),
        ],
        dtype="uint16",
        overwrite=True,
        backend="tiff",
    )

    # Create and write data
    with create_stream(settings) as stream:
        for _p in range(2):  # 2 positions
            for _t in range(2):  # 2 time points
                frame = np.random.randint(0, 1000, (32, 32), dtype=np.uint16)
                stream.append(frame)

    # Create updated metadata with multiple images (positions)
    updated_metadata = create_metadata(
        image_name="Position 0 Updated",
        channel_name="Channel P0",
        size_t=2,
        size_c=1,
        size_z=1,
        size_x=32,
        size_y=32,
        num_images=2,
    )

    # Update metadata after writing
    stream._backend.update_metadata(updated_metadata)

    # Verify each position file has the correct metadata
    base_path = tmp_path / "test_update_multipos"

    for pos_idx in range(2):
        pos_file = base_path.with_name(f"{base_path.name}_p{pos_idx:03d}.ome.tiff")
        assert pos_file.exists()

        with tifffile.TiffFile(str(pos_file)) as tif:
            ome_xml = tif.ome_metadata
            expected_name = f"Position {pos_idx} Updated"
            expected_channel = f"Channel P{pos_idx}"

            assert ome_xml is not None
            assert expected_name in ome_xml
            assert expected_channel in ome_xml

            # Verify we can parse the OME-XML and it contains only this position's data
            ome_obj = from_xml(ome_xml)
            assert len(ome_obj.images) == 1  # Each file should have only one image
            assert ome_obj.images[0].name == expected_name
            assert ome_obj.images[0].pixels.channels[0].name == expected_channel
            assert ome_obj.images[0].id == f"Image:{pos_idx}"


def test_update_metadata_error_conditions(tmp_path: Path) -> None:
    """Test error conditions in update_metadata method."""
    # Create acquisition settings
    settings = AcquisitionSettings(
        root_path=str(tmp_path / "test_error_metadata.ome.tiff"),
        dimensions=[
            Dimension(name="t", count=1, type="time"),
            Dimension(name="y", count=32, chunk_size=16, type="space"),
            Dimension(name="x", count=32, chunk_size=16, type="space"),
        ],
        dtype="uint16",
        overwrite=True,
        backend="tiff",
    )

    # Create and write data
    with create_stream(settings) as stream:
        frame = np.random.randint(0, 1000, (32, 32), dtype=np.uint16)
        stream.append(frame)

    # Test with invalid metadata structure
    invalid_metadata = {"not": "an ome object"}

    # This should raise a TypeError due to wrong type
    with pytest.raises(TypeError, match=r"Expected ome_types\.model\.OME metadata"):
        stream._backend.update_metadata(invalid_metadata)

    # Test that we can successfully update with valid metadata after error
    valid_metadata = create_metadata(
        image_name="Test Image After Error",
        channel_name="Test",
        size_x=32,
        size_y=32,
    )

    # This should work without errors
    stream._backend.update_metadata(valid_metadata)

    # Verify the metadata was actually updated
    import tifffile

    output_path = tmp_path / "test_error_metadata.ome.tiff"
    with tifffile.TiffFile(str(output_path)) as tif:
        ome_xml = tif.ome_metadata
        assert ome_xml is not None
        assert "Test Image After Error" in ome_xml


def test_update_metadata_with_plates(tmp_path: Path) -> None:
    """Test update_metadata with plate metadata for multi-position experiments."""

    # Create acquisition settings with plate
    settings = AcquisitionSettings(
        root_path=str(tmp_path / "test_plate_metadata.ome.tiff"),
        dimensions=[
            PositionDimension(
                positions=[
                    Position(name="Well_A01", row="A", column="1"),
                    Position(name="Well_A02", row="A", column="2"),
                ]
            ),
            Dimension(name="t", count=1, type="time"),
            Dimension(name="c", count=1, type="channel"),
            Dimension(name="z", count=1, type="space"),
            Dimension(name="y", count=32, chunk_size=16, type="space"),
            Dimension(name="x", count=32, chunk_size=16, type="space"),
        ],
        dtype="uint16",
        overwrite=True,
        backend="tiff",
        plate=Plate(
            name="Test Plate",
            row_names=["A"],
            column_names=["1", "2"],
        ),
    )

    # Create and write data
    with create_stream(settings) as stream:
        for _p in range(2):  # 2 wells
            frame = np.random.randint(0, 1000, (32, 32), dtype=np.uint16)
            stream.append(frame)

    updated_metadata = create_metadata(
        image_name="Well A01 Field 1",
        channel_name="Channel",
        size_x=32,
        size_y=32,
        num_images=2,
        include_plate=True,
    )

    # Manually update the second image name
    updated_metadata.images[1].name = "Well A02 Field 1"

    # Update metadata after writing
    stream._backend.update_metadata(updated_metadata)

    # Verify that each position file contains only the relevant plate information
    base_path = tmp_path / "test_plate_metadata"

    for pos_idx in range(2):
        pos_file = base_path.with_name(f"{base_path.name}_p{pos_idx:03d}.ome.tiff")
        assert pos_file.exists()

        with tifffile.TiffFile(str(pos_file)) as tif:
            ome_xml = tif.ome_metadata
            assert ome_xml is not None
            ome_obj = from_xml(ome_xml)

            # Each file should have only one image and relevant plate info
            assert len(ome_obj.images) == 1
            assert ome_obj.images[0].id == f"Image:{pos_idx}"

            # Should have plate information, but only the relevant well
            if ome_obj.plates:
                assert len(ome_obj.plates) == 1
                plate_obj = ome_obj.plates[0]
                assert len(plate_obj.wells) == 1  # Only the relevant well
                well = plate_obj.wells[0]
                assert well.well_samples[0].image_ref is not None
                assert well.well_samples[0].image_ref.id == f"Image:{pos_idx}"
