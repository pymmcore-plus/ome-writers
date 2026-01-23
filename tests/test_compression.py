"""Tests for compression support across backends."""

from __future__ import annotations

import importlib.util
import json
from typing import TYPE_CHECKING, get_args

import numpy as np
import pytest

from ome_writers import AcquisitionSettings, Dimension, _schema, create_stream

if TYPE_CHECKING:
    from pathlib import Path

TIFF_COMPRESSIONS = get_args(_schema.TiffCompression)
ZARR_COMPRESSIONS = get_args(_schema.ZarrCompression)


@pytest.mark.parametrize("compression", TIFF_COMPRESSIONS)
def test_tiff_compression(compression: str, tmp_path: Path, tiff_backend: str) -> None:
    """Test TIFF backend with different compression options."""
    try:
        import tifffile
    except ImportError:
        pytest.skip("tifffile package is required for TIFF tests")

    if compression == "lzw" and not importlib.util.find_spec("imagecodecs"):
        pytest.skip("LZW compression requires imagecodecs package")

    settings = AcquisitionSettings(
        root_path=str(tmp_path / "test.ome.tiff"),
        dimensions=[
            Dimension(name="z", count=2, type="space"),
            Dimension(name="y", count=64, type="space"),
            Dimension(name="x", count=64, type="space"),
        ],
        dtype="uint16",
        compression=compression,
        backend=tiff_backend,
    )

    # Write data
    with create_stream(settings) as stream:
        for _ in range(2):
            frame = np.random.randint(0, 1000, (64, 64), dtype="uint16")
            stream.append(frame)

    # Verify compression in TIFF metadata
    with tifffile.TiffFile(tmp_path / "test.ome.tiff") as tif:
        expected_compression = (
            tifffile.COMPRESSION.LZW
            if compression == "lzw"
            else tifffile.COMPRESSION.NONE
        )
        assert tif.pages[0].compression == expected_compression


@pytest.mark.parametrize("compression", ZARR_COMPRESSIONS)
def test_zarr_compression(compression: str, zarr_backend: str, tmp_path: Path) -> None:
    """Test Zarr backends with different compression options."""
    settings = AcquisitionSettings(
        root_path=str(tmp_path / "test.ome.zarr"),
        backend=zarr_backend,
        dimensions=[
            Dimension(name="z", count=2, type="space"),
            Dimension(name="y", count=64, type="space"),
            Dimension(name="x", count=64, type="space"),
        ],
        dtype="uint16",
        compression=compression,
    )

    # Write data
    try:
        stream = create_stream(settings)
    except NotImplementedError as e:
        if "does not support settings" in str(e):
            pytest.xfail(f"{zarr_backend} does not support compression '{compression}'")
        raise

    with stream:
        for _ in range(2):
            frame = np.random.randint(0, 1000, (64, 64), dtype="uint16")
            stream.append(frame)

    # Verify compression in zarr metadata
    zarr_json = tmp_path / "test.ome.zarr" / "0" / "zarr.json"
    metadata = json.loads(zarr_json.read_text())

    # Check codecs based on compression type
    # acquire-zarr uses sharding, so codecs are nested differently
    compression_codecs = get_compression_codecs(metadata.get("codecs", []))
    if compression == "none":
        # Should have no compression codec (only bytes codec)
        assert len(compression_codecs) == 0, "Expected no compression codecs"
    elif compression.startswith("blosc-"):
        assert compression_codecs[0]["name"] == "blosc"
        assert compression_codecs[0]["configuration"]["shuffle"] == "shuffle"
        if compression == "blosc-zstd":
            assert compression_codecs[0]["configuration"]["clevel"] == 3
            assert compression_codecs[0]["configuration"]["cname"] == "zstd"
        elif compression == "blosc-lz4":
            assert compression_codecs[0]["configuration"]["clevel"] == 5
            assert compression_codecs[0]["configuration"]["cname"] == "lz4"
    elif compression == "zstd":
        assert compression_codecs[0]["name"] == "zstd"
        assert compression_codecs[0]["configuration"]["level"] == 3


def get_compression_codecs(codec_list: list[dict]) -> list[dict]:
    """Extract compression codecs, handling sharding for acquire-zarr."""
    result = []
    for codec in codec_list:
        if codec.get("name") == "sharding_indexed":
            # For acquire-zarr, codecs are nested inside sharding
            inner = codec.get("configuration", {}).get("codecs", [])
            result.extend(c for c in inner if c.get("name") != "bytes")
        elif codec.get("name") != "bytes":
            result.append(codec)
    return result
