"""Tests for coverage of _auto.py create_stream function."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest

from ome_writers._auto import create_stream
from ome_writers._dimensions import Dimension

if TYPE_CHECKING:
    from pathlib import Path


def test_create_stream_with_invalid_plate_type_tiff(tmp_path: Path) -> None:
    """Test create_stream with invalid plate type for tiff backend."""
    pytest.importorskip("ome_types")
    pytest.importorskip("yaozarrs")
    from yaozarrs.v05 import Column, PlateDef, PlateWell, Row

    # Create a yaozarrs PlateDef (wrong type for tiff)
    plate = PlateDef(
        name="Test",
        rows=[Row(name="A")],
        columns=[Column(name="1")],
        wells=[PlateWell(path="A/1", rowIndex=0, columnIndex=0)],
        field_count=1,
    )

    dimensions = [
        Dimension(label="p", size=1),
        Dimension(label="y", size=32),
        Dimension(label="x", size=32),
    ]

    with pytest.raises(TypeError, match=r"tiff.*Plate"):
        create_stream(
            str(tmp_path / "test.tiff"),
            dimensions=dimensions,
            dtype=np.uint16,
            backend="tiff",
            plate=plate,
        )


@pytest.mark.parametrize("backend", ["acquire-zarr", "tensorstore", "zarr"])
def test_create_stream_with_invalid_plate_type_zarr(
    tmp_path: Path, backend: str
) -> None:
    """Test create_stream with invalid plate type for zarr backends."""
    pytest.importorskip("ome_types")
    pytest.importorskip("yaozarrs")
    from ome_types.model import Plate as OMEPlate

    # Create an OME-Types Plate (wrong type for zarr backends)
    plate = OMEPlate(id="Plate:0", name="Test")

    dimensions = [
        Dimension(label="p", size=1),
        Dimension(label="y", size=32),
        Dimension(label="x", size=32),
    ]

    with pytest.raises(TypeError, match="PlateDef"):
        create_stream(
            str(tmp_path / "test.zarr"),
            dimensions=dimensions,
            dtype=np.uint16,
            backend=backend,
            plate=plate,
        )


def test_create_stream_tiff_plate_missing_ome_types(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Test error when ome-types is missing for tiff backend with plate."""
    # Mock ome_types as unavailable
    import sys

    monkeypatch.setitem(sys.modules, "ome_types", None)
    monkeypatch.setitem(sys.modules, "ome_types.model", None)

    dimensions = [
        Dimension(label="p", size=1),
        Dimension(label="y", size=32),
        Dimension(label="x", size=32),
    ]

    with pytest.raises(ImportError, match="ome-types is required"):
        create_stream(
            str(tmp_path / "test.tiff"),
            dimensions=dimensions,
            dtype=np.uint16,
            backend="tiff",
            plate=object(),  # Some dummy plate object
        )


def test_create_stream_zarr_plate_missing_yaozarrs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Test error when yaozarrs is missing for zarr backend with plate."""
    # Mock yaozarrs as unavailable
    import sys

    monkeypatch.setitem(sys.modules, "yaozarrs", None)
    monkeypatch.setitem(sys.modules, "yaozarrs.v05", None)

    dimensions = [
        Dimension(label="p", size=1),
        Dimension(label="y", size=32),
        Dimension(label="x", size=32),
    ]

    with pytest.raises(ImportError, match="yaozarrs is required"):
        create_stream(
            str(tmp_path / "test.zarr"),
            dimensions=dimensions,
            dtype=np.uint16,
            backend="tensorstore",
            plate=object(),  # Some dummy plate object
        )
