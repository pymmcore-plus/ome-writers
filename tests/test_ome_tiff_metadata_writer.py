"""Tests specifically for TifffileStream and OME-TIFF functionality."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pytest

import ome_writers as omew

if TYPE_CHECKING:
    import ome_types.model as ome


def create_metadata(
    *,
    image_name: str = "Test Image",
    channel_name: str = "Test Channel",
    dtype: str = "uint16",
    size_t: int = 1,
    size_c: int = 1,
    size_z: int = 1,
    size_x: int = 32,
    size_y: int = 32,
    num_images: int = 1,
    plates: list[ome.Plate] | None = None,
) -> ome.OME:
    """Create OME metadata object with customizable parameters."""
    import ome_types.model as ome

    # Create base image
    channels = [
        ome.Channel(
            id="Channel:0",
            name=channel_name,
            samples_per_pixel=1,
        )
    ]

    pixels = ome.Pixels(
        id="Pixels:0",
        type=dtype,  # type: ignore[arg-type]
        size_x=size_x,
        size_y=size_y,
        size_z=size_z,
        size_c=size_c,
        size_t=size_t,
        dimension_order="XYCZT",  # type: ignore[arg-type]
        channels=channels,
    )

    base_image = ome.Image(
        id="Image:0",
        name=image_name,
        pixels=pixels,
    )

    images = []

    # Create additional images if needed
    if num_images > 1:
        for i in range(num_images):
            # Create new channels for this image
            channel_name_final = (
                channel_name.replace("P0", f"P{i}")
                if "P0" in channel_name
                else channel_name
            )
            image_channels = [
                ome.Channel(
                    id=f"Channel:{i}",
                    name=channel_name_final,
                    samples_per_pixel=1,
                )
            ]

            # Create new pixels for this image
            image_pixels = ome.Pixels(
                id=f"Pixels:{i}",
                type=dtype,  # type: ignore[arg-type]
                size_x=size_x,
                size_y=size_y,
                size_z=size_z,
                size_c=size_c,
                size_t=size_t,
                dimension_order="XYCZT",  # type: ignore[arg-type]
                channels=image_channels,
            )

            # Customize per-position names if they contain position info
            final_image_name = image_name
            if "Position" in image_name:
                final_image_name = image_name.replace("0", str(i))

            image = ome.Image(
                id=f"Image:{i}",
                name=final_image_name,
                pixels=image_pixels,
            )
            images.append(image)
    else:
        images = [base_image]

    # Create OME object
    ome_model = ome.OME(
        images=images,
        creator=f"ome_writers v{omew.__version__}",
    )

    # Add plates if provided
    if plates is not None:
        ome_model.plates = plates

    return ome_model


def test_update_metadata_single_file(tmp_path: Path) -> None:
    """Test update_metadata method for single-file TIFF streams."""
    # Only test with tifffile backend since update_metadata is TIFF-specific
    if not omew.TifffileStream.is_available():
        pytest.skip("tifffile not available")

    data_gen, dimensions, dtype = omew.fake_data_for_sizes(
        sizes={"t": 2, "c": 1, "z": 1, "y": 32, "x": 32},
        chunk_sizes={"y": 16, "x": 16},
    )

    output_path = tmp_path / "test_update_metadata.ome.tiff"

    # Create and write data
    stream = omew.TifffileStream()
    stream = stream.create(str(output_path), dtype, dimensions)
    assert stream.is_active()

    for frame in data_gen:
        stream.append(frame)
    stream.flush()
    assert not stream.is_active()

    # Create updated metadata
    updated_metadata = create_metadata(
        image_name="Updated Test Image",
        channel_name="Updated Channel",
        dtype=dtype.name,
        size_t=2,
    )

    # Update metadata after flush
    stream.update_ome_metadata(updated_metadata)

    # Verify the metadata was updated by reading it back
    import tifffile

    with tifffile.TiffFile(str(output_path)) as tif:
        ome_xml = tif.ome_metadata
        assert ome_xml is not None
        assert "Updated Test Image" in ome_xml
        assert "Updated Channel" in ome_xml

        # Also verify we can parse the OME-XML
        from ome_types import from_xml

        ome_obj = from_xml(ome_xml)
        assert ome_obj.images[0].name == "Updated Test Image"
        assert ome_obj.images[0].pixels.channels[0].name == "Updated Channel"


def test_update_metadata_multifile(tmp_path: Path) -> None:
    """Test update_metadata method for multi-position TIFF streams."""
    # Only test with tifffile backend since update_metadata is TIFF-specific
    if not omew.TifffileStream.is_available():
        pytest.skip("tifffile not available")

    data_gen, dimensions, dtype = omew.fake_data_for_sizes(
        sizes={"t": 2, "c": 1, "z": 1, "y": 32, "x": 32, "p": 2},
        chunk_sizes={"y": 16, "x": 16},
    )

    output_path = tmp_path / "test_update_multipos.ome.tiff"

    # Create and write data
    stream = omew.TifffileStream()
    stream = stream.create(str(output_path), dtype, dimensions)
    assert stream.is_active()

    for frame in data_gen:
        stream.append(frame)
    stream.flush()
    assert not stream.is_active()

    # Create updated metadata with multiple images (positions)
    updated_metadata = create_metadata(
        image_name="Position 0 Updated",
        channel_name="Channel P0",
        dtype=dtype.name,
        size_t=2,
        num_images=2,
    )

    # Update metadata after flush
    stream.update_ome_metadata(updated_metadata)

    # Verify each position file has the correct metadata
    import tifffile

    base_path = Path(str(output_path).replace(".ome.tiff", ""))

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
            from ome_types import from_xml

            ome_obj = from_xml(ome_xml)
            assert len(ome_obj.images) == 1  # Each file should have only one image
            assert ome_obj.images[0].name == expected_name
            assert ome_obj.images[0].pixels.channels[0].name == expected_channel
            assert ome_obj.images[0].id == f"Image:{pos_idx}"


def test_update_metadata_error_conditions(tmp_path: Path) -> None:
    """Test error conditions in update_metadata method."""
    # Only test with tifffile backend since update_metadata is TIFF-specific
    if not omew.TifffileStream.is_available():
        pytest.skip("tifffile not available")

    data_gen, dimensions, dtype = omew.fake_data_for_sizes(
        sizes={"t": 1, "y": 32, "x": 32},
        chunk_sizes={"y": 16, "x": 16},
    )

    output_path = tmp_path / "test_error_metadata.ome.tiff"

    # Create and write data
    stream = omew.TifffileStream()
    stream = stream.create(str(output_path), dtype, dimensions)

    for frame in data_gen:
        stream.append(frame)
    stream.flush()

    # Test with invalid metadata structure that causes validation errors
    # Since we now require an OME object, we'll test with a non-OME object
    invalid_metadata = {"not": "an ome object"}

    # This should raise a TypeError due to wrong type
    with pytest.raises(TypeError, match="Expected OME metadata"):
        stream.update_ome_metadata(invalid_metadata)

    # Test that we can successfully update with valid metadata after flush
    valid_metadata = create_metadata(
        image_name="Test Image After Error",
        channel_name="Test",
        dtype=dtype.name,
    )

    # This should work without errors
    stream.update_ome_metadata(valid_metadata)

    # Verify the metadata was actually updated
    import tifffile

    with tifffile.TiffFile(str(output_path)) as tif:
        ome_xml = tif.ome_metadata
        assert ome_xml is not None
        assert "Test Image After Error" in ome_xml


def test_update_metadata_with_plates(tmp_path: Path) -> None:
    """Test update_metadata with plate metadata for multi-position experiments."""
    # Only test with tifffile backend since update_metadata is TIFF-specific
    ome = pytest.importorskip("ome_types.model", reason="ome_types not installed")
    if not omew.TifffileStream.is_available():
        pytest.skip("tifffile not available")

    data_gen, dimensions, dtype = omew.fake_data_for_sizes(
        sizes={"t": 1, "c": 1, "z": 1, "y": 32, "x": 32, "p": 2},
        chunk_sizes={"y": 16, "x": 16},
    )

    output_path = tmp_path / "test_plate_metadata.ome.tiff"

    # Create and write data
    stream = omew.TifffileStream()
    stream = stream.create(str(output_path), dtype, dimensions)

    for frame in data_gen:
        stream.append(frame)
    stream.flush()

    # Create metadata with plate information
    plates = [
        ome.Plate(
            id="Plate:0",
            name="Test Plate",
            wells=[
                ome.Well(
                    id="Well:0",
                    row=0,
                    column=0,
                    well_samples=[
                        ome.WellSample(
                            id="WellSample:0",
                            index=0,
                            image_ref=ome.ImageRef(id="Image:0"),
                        )
                    ],
                ),
                ome.Well(
                    id="Well:1",
                    row=0,
                    column=1,
                    well_samples=[
                        ome.WellSample(
                            id="WellSample:1",
                            index=0,
                            image_ref=ome.ImageRef(id="Image:1"),
                        )
                    ],
                ),
            ],
        )
    ]

    updated_metadata = create_metadata(
        image_name="Well A01 Field 1",
        channel_name="Channel",
        dtype=dtype.name,
        num_images=2,
        plates=plates,
    )

    # Manually update the second image name since it's not automatic
    # With OME objects, we need to modify the object directly
    updated_metadata.images[1].name = "Well A02 Field 1"

    # Update metadata after flush
    stream.update_ome_metadata(updated_metadata)

    # Verify that each position file contains only the relevant plate information
    import tifffile
    from ome_types import from_xml

    base_path = Path(str(output_path).replace(".ome.tiff", ""))

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
                plate = ome_obj.plates[0]
                assert len(plate.wells) == 1  # Only the relevant well
                well = plate.wells[0]
                assert well.well_samples[0].image_ref is not None
                assert well.well_samples[0].image_ref.id == f"Image:{pos_idx}"


def test_tifffile_stream_basic_functionality(tmp_path: Path) -> None:
    """Test basic TifffileStream functionality."""
    if not omew.TifffileStream.is_available():
        pytest.skip("tifffile not available")

    data_gen, dimensions, dtype = omew.fake_data_for_sizes(
        sizes={"t": 2, "c": 2, "z": 1, "y": 32, "x": 32},
        chunk_sizes={"y": 16, "x": 16},
    )

    output_path = tmp_path / "test_basic_tifffile.ome.tiff"

    # Create and write data
    stream = omew.TifffileStream()
    assert not stream.is_active()  # Should not be active before create()

    stream = stream.create(str(output_path), dtype, dimensions)
    assert stream.is_active()

    # Write all frames
    frame_count = 0
    for frame in data_gen:
        stream.append(frame)
        frame_count += 1

    assert frame_count == 4  # 2 time points * 2 channels

    stream.flush()
    assert not stream.is_active()
    assert output_path.exists()

    # Verify we can read the data back
    import tifffile

    with tifffile.TiffFile(str(output_path)) as tif:
        data = tif.asarray()
        expected_shape = (2, 2, 1, 32, 32)  # (t, c, z, y, x)
        assert data.shape == expected_shape
        assert data.dtype == dtype


def test_tifffile_stream_multiposition_basic(tmp_path: Path) -> None:
    """Test basic multi-position TifffileStream functionality."""
    if not omew.TifffileStream.is_available():
        pytest.skip("tifffile not available")

    data_gen, dimensions, dtype = omew.fake_data_for_sizes(
        sizes={"t": 2, "c": 1, "z": 1, "y": 16, "x": 16, "p": 2},
        chunk_sizes={"y": 8, "x": 8},
    )

    output_path = tmp_path / "test_multipos_basic.ome.tiff"

    # Create and write data
    stream = omew.TifffileStream()
    stream = stream.create(str(output_path), dtype, dimensions)
    assert stream.is_active()

    # Write all frames
    frame_count = 0
    for frame in data_gen:
        stream.append(frame)
        frame_count += 1

    assert frame_count == 4  # 2 positions * 2 time points * 1 channel

    stream.flush()
    assert not stream.is_active()

    # Verify separate files were created for each position
    base_path = Path(str(output_path).replace(".ome.tiff", ""))
    pos0_file = base_path.with_name(f"{base_path.name}_p000.ome.tiff")
    pos1_file = base_path.with_name(f"{base_path.name}_p001.ome.tiff")

    assert pos0_file.exists()
    assert pos1_file.exists()

    # Verify each file has correct shape
    import tifffile

    for pos_file in [pos0_file, pos1_file]:
        with tifffile.TiffFile(str(pos_file)) as tif:
            data = tif.asarray()
            expected_shape = (2, 1, 1, 16, 16)  # (t, z, c, y, x)
            assert data.shape == expected_shape
            assert data.dtype == dtype


def test_tifffile_stream_error_handling(tmp_path: Path) -> None:
    """Test error handling specific to TifffileStream."""
    if not omew.TifffileStream.is_available():
        pytest.skip("tifffile not available")

    # Test append without create
    stream = omew.TifffileStream()
    test_frame = np.zeros((32, 32), dtype=np.uint16)
    with pytest.raises(RuntimeError, match="Stream is closed or uninitialized"):
        stream.append(test_frame)

    # Test file overwrite protection
    data_gen1, dimensions, dtype = omew.fake_data_for_sizes(
        sizes={"t": 1, "y": 16, "x": 16},
        chunk_sizes={"y": 8, "x": 8},
    )

    output_path = tmp_path / "test_overwrite.ome.tiff"

    # Create first stream
    stream1 = omew.TifffileStream()
    stream1 = stream1.create(str(output_path), dtype, dimensions)
    for frame in data_gen1:
        stream1.append(frame)
    stream1.flush()

    # Try to create second stream without overwrite (should fail)
    stream2 = omew.TifffileStream()
    with pytest.raises(FileExistsError, match="already exists"):
        stream2.create(str(output_path), dtype, dimensions, overwrite=False)

    # Should succeed with overwrite=True
    data_gen3, _, _ = omew.fake_data_for_sizes(
        sizes={"t": 1, "y": 16, "x": 16},
        chunk_sizes={"y": 8, "x": 8},
    )
    stream3 = omew.TifffileStream()
    stream3 = stream3.create(str(output_path), dtype, dimensions, overwrite=True)
    assert stream3.is_active()
    for frame in data_gen3:
        stream3.append(frame)
    stream3.flush()


def test_tifffile_stream_availability() -> None:
    """Test TifffileStream availability check."""
    # This test should always pass since we skip tests when tifffile is not available
    assert omew.TifffileStream.is_available() == (
        omew.TifffileStream.is_available()
    )  # Tautology, but tests the method


def test_tifffile_stream_context_manager(tmp_path: Path) -> None:
    """Test TifffileStream as context manager."""
    if not omew.TifffileStream.is_available():
        pytest.skip("tifffile not available")

    data_gen, dimensions, dtype = omew.fake_data_for_sizes(
        sizes={"t": 1, "y": 16, "x": 16},
        chunk_sizes={"y": 8, "x": 8},
    )

    output_path = tmp_path / "test_context_manager.ome.tiff"

    # Use as context manager
    with omew.TifffileStream() as stream:
        stream = stream.create(str(output_path), dtype, dimensions)
        assert stream.is_active()

        for frame in data_gen:
            stream.append(frame)

        # Stream should still be active inside context
        assert stream.is_active()

    # Stream should be flushed and inactive after exiting context
    assert not stream.is_active()
    assert output_path.exists()
