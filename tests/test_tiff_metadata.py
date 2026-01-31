"""Tests for TIFF backend update_metadata functionality."""

from __future__ import annotations

from functools import partial
from pathlib import Path as PathlibPath
from typing import TYPE_CHECKING
from unittest.mock import patch

import numpy as np
import pytest

from ome_writers import (
    AcquisitionSettings,
    Channel,
    Dimension,
    Plate,
    Position,
    PositionDimension,
    create_stream,
)

try:
    import ome_types
    from ome_types import from_tiff, from_xml
except ImportError:
    pytest.skip("ome_types not installed", allow_module_level=True)

from ome_writers._backends._ome_xml import MetadataMode, prepare_metadata

if TYPE_CHECKING:
    from pathlib import Path


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
        format={"name": "ome-tiff", "backend": tiff_backend},
    )

    with create_stream(settings) as stream:
        for _ in range(2):
            stream.append(np.random.randint(0, 1000, (32, 32), dtype=np.uint16))

    # Update metadata after context exits
    metadata = stream.get_metadata()
    # For single position, get position 0's metadata
    metadata[0].images[0].name = "Updated Image"
    metadata[0].images[0].pixels.channels[0].name = "Updated Channel"
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
        format={"name": "ome-tiff", "backend": tiff_backend},
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
    # For multi-position, each position's metadata contains all images
    # Update all positions to have the same updated metadata
    for p_idx in range(2):
        metadata[p_idx].images[0].name = "Position 0 Updated"
        metadata[p_idx].images[1].name = "Position 1 Updated"
    stream.update_metadata(metadata)

    # Verify each position file has the updated metadata
    for pos_idx in range(2):
        pos_file = tmp_path / f"multipos_p{pos_idx:03d}.ome.tiff"
        ome_obj = from_tiff(str(pos_file))
        assert ome_obj.images[0].name == "Position 0 Updated"
        assert ome_obj.images[1].name == "Position 1 Updated"


def test_update_metadata_error_conditions(tmp_path: Path, tiff_backend: str) -> None:
    """Test error conditions in update_metadata method."""
    settings = AcquisitionSettings(
        root_path=str(tmp_path / "error.ome.tiff"),
        dimensions=[
            Dimension(name="y", count=32, type="space"),
            Dimension(name="x", count=32, type="space"),
        ],
        dtype="uint16",
        format={"name": "ome-tiff", "backend": tiff_backend},
    )

    with create_stream(settings) as stream:
        stream.append(np.random.randint(0, 1000, (32, 32), dtype=np.uint16))

    # Invalid metadata type (not a dict) should raise TypeError
    with pytest.raises(TypeError, match=r"Expected dict\[int, ome_types.model.OME\]"):
        stream.update_metadata("not a dict")

    # Invalid value type should raise TypeError
    with pytest.raises(TypeError, match=r"Expected ome_types.model.OME for position"):
        stream.update_metadata({0: "not an ome object"})

    # Valid update should work
    metadata = stream.get_metadata()
    metadata[0].images[0].name = "Fixed"
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
        format={"name": "ome-tiff", "backend": tiff_backend},
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

    # Verify plate structure
    ome_obj = from_tiff(str(tmp_path / "plate_p000.ome.tiff"))
    assert len(ome_obj.plates) == 1
    plate = ome_obj.plates[0]
    assert plate.id == "Plate:0"
    assert plate.name == "Test Plate"
    assert plate.rows == 1
    assert plate.columns == 2
    assert len(plate.wells) == 2
    # Verify naming conventions are inferred correctly
    assert plate.row_naming_convention.value == "letter"
    assert plate.column_naming_convention.value == "number"

    # Verify wells and well samples link to images
    well_sample_refs = {}
    for well in plate.wells:
        for ws in well.well_samples:
            well_sample_refs[ws.index] = ws.image_ref.id if ws.image_ref else None
    assert well_sample_refs == {0: "Image:0", 1: "Image:1"}

    # Update metadata
    metadata = stream.get_metadata()
    # For multi-position, each position's metadata contains all images
    # Update all positions to have the same updated metadata
    for p_idx in range(2):
        metadata[p_idx].images[0].name = "Well A01"
        metadata[p_idx].images[1].name = "Well A02"
    stream.update_metadata(metadata)

    # Verify each well file has updated names
    for pos_idx in range(2):
        pos_file = tmp_path / f"plate_p{pos_idx:03d}.ome.tiff"
        ome_obj = from_tiff(str(pos_file))
        assert ome_obj.images[0].name == "Well A01"
        assert ome_obj.images[1].name == "Well A02"


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
        format={"name": "ome-tiff", "backend": tiff_backend},
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
    assert ome_obj.images[0].name == "0"
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
        format={"name": "ome-tiff", "backend": tiff_backend},
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


def test_prepare_meta(tmp_path: Path) -> None:
    """Test _prepare_meta function for TIFF backend."""
    from ome_writers._backends._ome_xml import MetadataMode, prepare_metadata

    settings = AcquisitionSettings(
        root_path=tmp_path / "test.ome.tiff",
        dimensions=[
            PositionDimension(positions=["Pos0", "Pos1"]),
            Dimension(name="t", count=2, type="time"),
            Dimension(name="c", count=1, type="channel"),
            Dimension(name="y", count=32, type="space"),
            Dimension(name="x", count=32, type="space"),
        ],
        dtype="uint16",
        format="tifffile",
    )
    for mode in MetadataMode:
        meta = prepare_metadata(settings, mode)
        assert isinstance(meta, dict)
        assert all(str(tmp_path) in key for key in meta.keys())
        companion = mode == MetadataMode.MULTI_MASTER_COMPANION
        assert sum(key.endswith("companion.ome") for key in meta) == companion


def test_channel_metadata_in_tiff(tmp_path: Path, tiff_backend: str) -> None:
    """Test that channel names are correctly written and read in TIFF files."""
    settings = AcquisitionSettings(
        root_path=str(tmp_path / "channel_names.ome.tiff"),
        dimensions=[
            Dimension(
                name="c",
                type="channel",
                coords=["DAPI", Channel(name="FITC", color="green")],
            ),
            Dimension(name="y", count=16, type="space"),
            Dimension(name="x", count=16, type="space"),
        ],
        dtype="uint16",
        format=tiff_backend,
    )

    with create_stream(settings) as stream:
        for _ in range(2):  # 2 channels
            stream.append(np.random.randint(0, 1000, (16, 16), dtype=np.uint16))

    ome_obj = from_tiff(str(tmp_path / "channel_names.ome.tiff"))
    pixels = ome_obj.images[0].pixels
    channel_names = [ch.name for ch in pixels.channels]
    assert channel_names == ["DAPI", "FITC"]
    channel_colors = [ch and ch.color.as_named() for ch in pixels.channels]
    # white is the default OMX-XML color, and all channels have a default color object.
    assert channel_colors == ["white", "green"]


def test_frame_metadata_single_position(tmp_path: Path, tiff_backend: str) -> None:
    """Test frame_metadata appears in OME-XML for single position."""
    root = tmp_path / "single.ome.tiff"
    settings = AcquisitionSettings(
        root_path=root,
        dimensions=[
            Dimension(name="t", count=3, type="time"),
            Dimension(name="y", count=16, type="space"),
            Dimension(name="x", count=16, type="space"),
        ],
        dtype="uint16",
        overwrite=True,
        format=tiff_backend,
    )

    # Write frames with metadata
    with create_stream(settings) as stream:
        for t in range(3):
            frame = np.random.randint(0, 1000, (16, 16), dtype=np.uint16)
            metadata = {
                "delta_t": t * 1.5,
                "exposure_time": 0.01,
                "position_x": 100.0,
                "position_y": 200.0,
                "position_z": 50.0,
                "temperature": 37.0 + t * 0.1,
            }
            stream.append(frame, frame_metadata=metadata)

    # Read OME-XML from TIFF
    ome_obj = from_tiff(str(root))
    image = ome_obj.images[0]
    planes = image.pixels.planes

    # Check we have 3 planes
    assert len(planes) == 3

    # Verify recognized keys mapped to Plane attributes
    assert planes[0].delta_t == 0.0
    assert planes[0].delta_t_unit.value == "s"
    assert planes[0].exposure_time == 0.01
    assert planes[0].exposure_time_unit.value == "s"
    assert planes[0].position_x == 100.0
    assert planes[0].position_y == 200.0
    assert planes[0].position_z == 50.0
    assert planes[1].delta_t == 1.5
    assert planes[2].delta_t == 3.0

    # Check StructuredAnnotations exist with MapAnnotations
    assert ome_obj.structured_annotations is not None
    map_annots = ome_obj.structured_annotations.map_annotations
    assert len(map_annots) == 3

    # Verify first frame's MapAnnotation contains all metadata
    extras = {m.k: m.value for m in map_annots[0].value.ms}
    assert extras["temperature"] == "37.0"

    # Verify AnnotationRefs link Planes to MapAnnotations
    assert len(planes[0].annotation_refs) == 1
    assert planes[0].annotation_refs[0].id == map_annots[0].id


def test_frame_metadata_multiposition(tmp_path: Path, tiff_backend: str) -> None:
    """Test frame_metadata is position-specific for multi-position."""
    root = tmp_path / "multipos.ome.tiff"
    settings = AcquisitionSettings(
        root_path=root,
        dimensions=[
            PositionDimension(positions=["Pos0", "Pos1"]),
            Dimension(name="t", count=2, type="time"),
            Dimension(name="y", count=16, type="space"),
            Dimension(name="x", count=16, type="space"),
        ],
        dtype="uint16",
        overwrite=True,
        format=tiff_backend,
    )

    # Write frames with position-specific metadata
    with create_stream(settings) as stream:
        for p_name in ["Pos0", "Pos1"]:
            for t in range(2):
                frame = np.random.randint(0, 1000, (16, 16), dtype=np.uint16)
                metadata = {
                    "delta_t": t * 1.0,
                    "position_name": p_name,
                    "frame_number": t,
                }
                stream.append(frame, frame_metadata=metadata)

    # Verify each position file has its own frame_metadata
    for pos_idx, pos_name in enumerate(["Pos0", "Pos1"]):
        pos_file = tmp_path / f"multipos_p{pos_idx:03d}.ome.tiff"
        ome_obj = from_tiff(str(pos_file))

        # Get this position's image
        image = ome_obj.images[pos_idx]
        planes = image.pixels.planes

        # Each position should have 2 planes
        assert len(planes) == 2

        # Verify Plane attributes
        assert planes[0].delta_t == 0.0
        assert planes[1].delta_t == 1.0

        # Get MapAnnotations for this position
        structured_annots = ome_obj.structured_annotations
        assert structured_annots is not None
        map_annots = structured_annots.map_annotations
        # NOTE!!
        # this is actually a bug... there should be 4 total annotations,
        # but we're not currently updating the "master" OME-XML correctly
        # when writing multi-position files. we update each one differently.
        # it's interesting: it might kinda be ok... you only get the structured
        # annotations associated with the position you read...
        # but not sure it's spec-compliant. for now, just test what we have.
        assert len(map_annots) == 2

        # Verify metadata is position-specific
        for frame_idx, annot in enumerate(map_annots):
            annot_dict = {m.k: m.value for m in annot.value.ms}
            assert annot_dict["position_name"] == pos_name
            assert annot_dict["frame_number"] == str(frame_idx)

        # Verify AnnotationRefs link Planes to MapAnnotations
        for frame_idx, plane in enumerate(planes):
            assert len(plane.annotation_refs) == 1
            assert plane.annotation_refs[0].id == map_annots[frame_idx].id


# =============================================================================
# Test metadata modes
# =============================================================================


MULTI_FILE_MODES = [
    MetadataMode.MULTI_REDUNDANT,
    MetadataMode.MULTI_MASTER_TIFF,
    MetadataMode.MULTI_MASTER_COMPANION,
    # TODO: multi-position single file still has to be implemented
    # MetadataMode.SINGLE_FILE
]


# this will be removed once we expose the modes via the public API
def _write_with_mode(
    tmp_path: Path,
    dimensions: list[Dimension | PositionDimension],
    mode: MetadataMode,
    plate: Plate | None = None,
) -> None:
    """Write test data with specified mode."""
    settings = AcquisitionSettings(
        root_path=tmp_path / "test.ome.tiff",
        dimensions=dimensions,
        dtype="uint16",
        overwrite=True,
        format="tifffile",
        plate=plate,
    )

    num_frames = int(np.prod(settings.shape[:-2]))
    frame_shape = settings.shape[-2:]

    with patch(
        "ome_writers._backends._tifffile.prepare_metadata",
        side_effect=partial(prepare_metadata, mode=mode),
    ):
        with create_stream(settings) as stream:
            for i in range(num_frames):
                frame = np.full(frame_shape, fill_value=i, dtype=settings.dtype)
                stream.append(frame)


def _get_full_ome(tmp_path: Path, mode: MetadataMode) -> ome_types.OME | None:
    """Get OME model with full metadata for the given mode."""
    if mode == MetadataMode.MULTI_MASTER_COMPANION:
        companion = next(tmp_path.glob("*.companion.ome"))
        with open(companion, encoding="utf-8") as f:
            return from_xml(f.read())
    elif mode == MetadataMode.MULTI_MASTER_TIFF:
        master = next(f for f in tmp_path.glob("*.ome.tiff") if "_p000" in f.name)
        return from_tiff(str(master))
    elif mode == MetadataMode.MULTI_REDUNDANT:
        any_file = next(tmp_path.glob("*.ome.tiff"))
        return from_tiff(str(any_file))


@pytest.mark.parametrize("mode", MULTI_FILE_MODES)
def test_basic_multiposition(tmp_path: Path, mode: MetadataMode) -> None:
    """Test basic multi-position without plate."""
    dimensions = [
        PositionDimension(positions=["Pos0", "Pos1"]),
        Dimension(name="c", count=2, type="channel"),
        Dimension(name="y", count=32, type="space"),
        Dimension(name="x", count=32, type="space"),
    ]

    _write_with_mode(tmp_path, dimensions, mode)

    # Verify file structure
    tiff_files = list(tmp_path.glob("*.ome.tiff"))
    assert len(tiff_files) == 2

    # Get full metadata
    ome = _get_full_ome(tmp_path, mode)
    assert len(ome.images) == 2
    assert [img.name for img in ome.images] == ["Pos0", "Pos1"]

    # Verify dimensions
    for img in ome.images:
        assert img.pixels.size_x == 32
        assert img.pixels.size_y == 32
        assert img.pixels.size_c == 2
        assert img.pixels.size_z == 1
        assert img.pixels.size_t == 1


@pytest.mark.parametrize("mode", MULTI_FILE_MODES)
def test_5d_with_physical_sizes(tmp_path: Path, mode: MetadataMode) -> None:
    """Test full 5D acquisition with physical pixel sizes."""
    dimensions = [
        Dimension(name="t", count=2, type="time"),
        PositionDimension(positions=["Pos0", "Pos1"]),
        Dimension(name="c", count=3, type="channel"),
        Dimension(name="z", count=4, type="space", scale=2.0, unit="micrometer"),
        Dimension(name="y", count=64, type="space", scale=0.5, unit="micrometer"),
        Dimension(name="x", count=64, type="space", scale=0.5, unit="micrometer"),
    ]

    _write_with_mode(tmp_path, dimensions, mode)

    ome = _get_full_ome(tmp_path, mode)
    assert len(ome.images) == 2

    for img in ome.images:
        pix = img.pixels
        assert pix.size_x == 64
        assert pix.size_y == 64
        assert pix.size_z == 4
        assert pix.size_c == 3
        assert pix.size_t == 2
        assert pix.physical_size_x == 0.5
        assert pix.physical_size_y == 0.5
        assert pix.physical_size_z == 2.0


@pytest.mark.parametrize("mode", MULTI_FILE_MODES)
def test_plate_basic(tmp_path: Path, mode: MetadataMode) -> None:
    """Test plate with one field per well."""
    dimensions = [
        PositionDimension(
            positions=[
                Position(name="A1", plate_row="A", plate_column="1"),
                Position(name="A2", plate_row="A", plate_column="2"),
                Position(name="B1", plate_row="B", plate_column="1"),
            ]
        ),
        Dimension(name="y", count=32, type="space"),
        Dimension(name="x", count=32, type="space"),
    ]
    plate = Plate(name="Test Plate", row_names=["A", "B"], column_names=["1", "2"])

    _write_with_mode(tmp_path, dimensions, mode, plate=plate)

    ome = _get_full_ome(tmp_path, mode)
    assert len(ome.images) == 3
    assert len(ome.plates) == 1

    plate_obj = ome.plates[0]
    assert plate_obj.name == "Test Plate"
    assert plate_obj.rows == 2
    assert plate_obj.columns == 2
    assert len(plate_obj.wells) == 3


@pytest.mark.parametrize("mode", MULTI_FILE_MODES)
def test_plate_multiple_fields(tmp_path: Path, mode: MetadataMode) -> None:
    """Test plate with multiple fields per well."""
    dimensions = [
        PositionDimension(
            positions=[
                Position(name="fov1", plate_row="A", plate_column="1"),
                Position(name="fov2", plate_row="A", plate_column="1"),
                Position(name="fov1", plate_row="A", plate_column="2"),
            ]
        ),
        Dimension(name="y", count=32, type="space"),
        Dimension(name="x", count=32, type="space"),
    ]
    plate = Plate(name="Multi-Field", row_names=["A"], column_names=["1", "2"])

    _write_with_mode(tmp_path, dimensions, mode, plate=plate)

    ome = _get_full_ome(tmp_path, mode)
    plate_obj = ome.plates[0]

    # Find well A1 - should have 2 fields
    well_a1 = next(w for w in plate_obj.wells if w.row == 0 and w.column == 0)
    assert len(well_a1.well_samples) == 2

    # Find well A2 - should have 1 field
    well_a2 = next(w for w in plate_obj.wells if w.row == 0 and w.column == 1)
    assert len(well_a2.well_samples) == 1


@pytest.mark.parametrize("mode", MULTI_FILE_MODES)
def test_file_structure_by_mode(tmp_path: Path, mode: MetadataMode) -> None:
    """Verify correct file structure for each metadata mode."""
    dimensions = [
        PositionDimension(positions=["Pos0", "Pos1"]),
        Dimension(name="y", count=32, type="space"),
        Dimension(name="x", count=32, type="space"),
    ]

    _write_with_mode(tmp_path, dimensions, mode)

    tiff_files = sorted(tmp_path.glob("*.ome.tiff"))
    companion_files = list(tmp_path.glob("*.companion.ome"))

    assert len(tiff_files) == 2

    if mode == MetadataMode.MULTI_REDUNDANT:
        # Each TIFF has full metadata, no companion
        assert len(companion_files) == 0
        root_uuids = set()
        for fidx, tiff_file in enumerate(tiff_files):
            ome = from_tiff(str(tiff_file))
            assert ome.binary_only is None
            assert len(ome.images) == 2
            root_uuids.add(ome.uuid)
            # TiffData should have UUID children with FileName
            for iidx, img in enumerate(ome.images):
                for td in img.pixels.tiff_data_blocks:
                    assert td.uuid is not None, "TiffData must have UUID in multi-file"
                    assert td.uuid.file_name is not None, "UUID must have FileName"
                    if fidx == iidx:
                        assert td.uuid.file_name == tiff_file.name
        # Each file has unique UUID
        assert len(root_uuids) == 2

    elif mode == MetadataMode.MULTI_MASTER_TIFF:
        # First TIFF is master, others have BinaryOnly, no companion
        assert len(companion_files) == 0
        master = next(f for f in tiff_files if "_p000" in f.name)
        master_ome = from_tiff(str(master))
        assert master_ome.binary_only is None
        assert len(master_ome.images) == 2
        # Master should have TiffData with UUIDs
        for img in master_ome.images:
            for td in img.pixels.tiff_data_blocks:
                assert td.uuid is not None, "Master TiffData must have UUID"
                assert td.uuid.file_name is not None, "UUID must have FileName"

        for tiff_file in tiff_files:
            if tiff_file != master:
                ome = from_tiff(str(tiff_file))
                assert ome.binary_only is not None
                assert ome.binary_only.uuid == master_ome.uuid
                assert PathlibPath(ome.binary_only.metadata_file).name == master.name

    elif mode == MetadataMode.MULTI_MASTER_COMPANION:
        # Companion has full metadata, all TIFFs have BinaryOnly
        assert len(companion_files) == 1
        with open(companion_files[0], encoding="utf-8") as f:
            companion_ome = from_xml(f.read())
        assert companion_ome.binary_only is None
        assert len(companion_ome.images) == 2
        # Companion should have TiffData with UUIDs
        for img in companion_ome.images:
            for td in img.pixels.tiff_data_blocks:
                assert td.uuid is not None, "Companion TiffData must have UUID"
                assert td.uuid.file_name is not None, "UUID must have FileName"

        for tiff_file in tiff_files:
            ome = from_tiff(str(tiff_file))
            assert ome.binary_only is not None
            assert ome.binary_only.uuid == companion_ome.uuid
            assert (
                PathlibPath(ome.binary_only.metadata_file).name
                == companion_files[0].name
            )


@pytest.mark.parametrize("mode", MULTI_FILE_MODES)
def test_pixel_data_integrity(tmp_path: Path, mode: MetadataMode) -> None:
    """Verify pixel data can be read back correctly."""
    import tifffile

    dimensions = [
        PositionDimension(positions=["Pos0", "Pos1"]),
        Dimension(name="z", count=2, type="space"),
        Dimension(name="y", count=32, type="space"),
        Dimension(name="x", count=32, type="space"),
    ]

    _write_with_mode(tmp_path, dimensions, mode)

    # Read back and verify
    tiff_files = sorted(tmp_path.glob("*.ome.tiff"))
    for pos_idx, tiff_file in enumerate(tiff_files):
        with tifffile.TiffFile(str(tiff_file)) as tif:
            assert len(tif.pages) == 2
            for z_idx in range(2):
                expected_value = pos_idx * 2 + z_idx
                data = tif.pages[z_idx].asarray()
                assert data.shape == (32, 32)
                assert np.all(data == expected_value)
