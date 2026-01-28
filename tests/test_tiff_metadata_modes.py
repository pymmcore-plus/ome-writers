"""End-to-end tests for all MetadataMode implementations in OME-TIFF.

Tests the three multi-file metadata modes:
- MULTI_REDUNDANT: Each file has full OME-XML
- MULTI_MASTER_TIFF: First TIFF is master with full metadata
- MULTI_MASTER_COMPANION: Companion XML has full metadata

See ome-tiff-spec.md for detailed specification.
"""

from __future__ import annotations

from functools import partial
from typing import TYPE_CHECKING
from unittest.mock import patch

import numpy as np
import ome_types
import pytest

from ome_writers import (
    AcquisitionSettings,
    Dimension,
    Plate,
    Position,
    PositionDimension,
    create_stream,
)
from ome_writers._backends._ome_xml import (
    MetadataMode,
    prepare_metadata,
)

if TYPE_CHECKING:
    from pathlib import Path

try:
    from ome_types import from_tiff, from_xml

    HAS_OME_TYPES = True
except ImportError:
    HAS_OME_TYPES = False


pytestmark = pytest.mark.skipif(not HAS_OME_TYPES, reason="ome_types not installed")

MULTI_FILE_MODES = [
    MetadataMode.MULTI_REDUNDANT,
    MetadataMode.MULTI_MASTER_TIFF,
    MetadataMode.MULTI_MASTER_COMPANION,
    # TODO: multi-position single file still has to be implemented
    # MetadataMode.SINGLE_FILE
]


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
        backend="tifffile",
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
        with open(companion) as f:
            return from_xml(f.read())
    elif mode == MetadataMode.MULTI_MASTER_TIFF:
        master = next(f for f in tmp_path.glob("*.ome.tiff") if "_p000" in f.name)
        return from_tiff(str(master))
    elif mode == MetadataMode.MULTI_REDUNDANT:
        any_file = next(tmp_path.glob("*.ome.tiff"))
        return from_tiff(str(any_file))


# =============================================================================
# Tests
# =============================================================================


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
                Position(name="A1_F1", plate_row="A", plate_column="1"),
                Position(name="A1_F2", plate_row="A", plate_column="1"),
                Position(name="A2_F1", plate_row="A", plate_column="2"),
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
        for tiff_file in tiff_files:
            ome = from_tiff(str(tiff_file))
            assert ome.binary_only is None
            assert len(ome.images) == 2
            root_uuids.add(ome.uuid)
        # Each file has unique UUID
        assert len(root_uuids) == 2

    elif mode == MetadataMode.MULTI_MASTER_TIFF:
        # First TIFF is master, others have BinaryOnly, no companion
        assert len(companion_files) == 0
        master = next(f for f in tiff_files if "_p000" in f.name)
        master_ome = from_tiff(str(master))
        assert master_ome.binary_only is None
        assert len(master_ome.images) == 2

        for tiff_file in tiff_files:
            if tiff_file != master:
                ome = from_tiff(str(tiff_file))
                assert ome.binary_only is not None
                assert ome.binary_only.uuid == master_ome.uuid

    elif mode == MetadataMode.MULTI_MASTER_COMPANION:
        # Companion has full metadata, all TIFFs have BinaryOnly
        assert len(companion_files) == 1
        with open(companion_files[0]) as f:
            companion_ome = from_xml(f.read())
        assert companion_ome.binary_only is None
        assert len(companion_ome.images) == 2

        for tiff_file in tiff_files:
            ome = from_tiff(str(tiff_file))
            assert ome.binary_only is not None
            assert ome.binary_only.uuid == companion_ome.uuid


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
