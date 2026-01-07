"""Test plate/HCS functionality with yaozarrs streams."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest

import ome_writers as omew

if TYPE_CHECKING:
    from pathlib import Path

try:
    from yaozarrs import v05
except ImportError:
    pytest.skip("yaozarrs not installed", allow_module_level=True)


@pytest.mark.skipif(
    not omew.TensorStoreZarrStream.is_available(),
    reason="tensorstore not available",
)
def test_tensorstore_plate_write(tmp_path: Path) -> None:
    """Test writing HCS plate data with TensorStoreZarrStream."""
    output = tmp_path / "test_plate.ome.zarr"

    # Define a simple 2x2 plate with 2 fields per well
    plate_def = v05.PlateDef(
        name="Test Plate",
        rows=[v05.Row(name="A"), v05.Row(name="B")],
        columns=[v05.Column(name="1"), v05.Column(name="2")],
        wells=[
            v05.PlateWell(path="A/1", rowIndex=0, columnIndex=0),
            v05.PlateWell(path="A/2", rowIndex=0, columnIndex=1),
            v05.PlateWell(path="B/1", rowIndex=1, columnIndex=0),
            v05.PlateWell(path="B/2", rowIndex=1, columnIndex=1),
        ],
        field_count=2,
    )

    # Total positions = 4 wells * 2 fields = 8
    total_positions = len(plate_def.wells) * (plate_def.field_count or 1)

    dims = [
        omew.Dimension(label="p", size=total_positions),
        omew.Dimension(label="z", size=2),
        omew.Dimension(label="y", size=32),
        omew.Dimension(label="x", size=32),
    ]

    stream = omew.TensorStoreZarrStream()
    stream.create(
        str(output),
        np.dtype("uint16"),
        dims,
        overwrite=True,
        plate=plate_def,
    )

    # Write data for each position
    for pos_idx in range(total_positions):
        for z in range(dims[1].size):
            # Create unique values per position/z
            value = pos_idx * 100 + z
            frame = np.full((dims[2].size, dims[3].size), value, dtype="uint16")
            stream.append(frame)

    stream.flush()

    # Validate the zarr store structure and metadata
    from .conftest import validate_zarr_store

    validate_zarr_store(output)

    # Verify well structure exists
    assert (output / "A" / "1" / "0" / "0").exists()  # Well A/1, field 0, dataset 0
    assert (output / "A" / "1" / "1" / "0").exists()  # Well A/1, field 1, dataset 0
    assert (output / "B" / "2" / "0" / "0").exists()  # Well B/2, field 0, dataset 0

    # Verify data values
    import tensorstore as ts

    spec = {
        "driver": "zarr3",
        "kvstore": {"driver": "file", "path": str(output / "A" / "1" / "0" / "0")},
    }
    arr = ts.open(spec).result()
    assert arr.shape == (2, 32, 32)
    assert arr[0, 0, 0].read().result() == 0  # pos=0, z=0
    assert arr[1, 0, 0].read().result() == 1  # pos=0, z=1

    spec = {
        "driver": "zarr3",
        "kvstore": {"driver": "file", "path": str(output / "A" / "1" / "1" / "0")},
    }
    arr = ts.open(spec).result()
    assert arr[0, 0, 0].read().result() == 100  # pos=1, z=0
    assert arr[1, 0, 0].read().result() == 101  # pos=1, z=1


@pytest.mark.skipif(
    not omew.ZarrPythonStream.is_available(),
    reason="zarr-python not available",
)
def test_zarr_python_plate_write(tmp_path: Path) -> None:
    """Test writing HCS plate data with ZarrPythonStream."""
    output = tmp_path / "test_plate_zarr.ome.zarr"

    plate_def = v05.PlateDef(
        name="Test Plate Zarr",
        rows=[v05.Row(name="A")],
        columns=[v05.Column(name="1")],
        wells=[v05.PlateWell(path="A/1", rowIndex=0, columnIndex=0)],
        field_count=1,
    )

    dims = [
        omew.Dimension(label="p", size=1),
        omew.Dimension(label="y", size=16),
        omew.Dimension(label="x", size=16),
    ]

    stream = omew.ZarrPythonStream()
    stream.create(
        str(output),
        np.dtype("uint8"),
        dims,
        overwrite=True,
        plate=plate_def,
    )

    # Write a single frame
    frame = np.full((dims[1].size, dims[2].size), 42, dtype="uint8")
    stream.append(frame)
    stream.flush()

    # Validate the zarr store structure
    from .conftest import validate_zarr_store

    validate_zarr_store(output)

    # Verify data values - try tensorstore first, then zarr-python
    import zarr
    from zarr import Array

    store = zarr.open_group(output, mode="r")
    arr: Array = store["A/1/0/0"]  # type: ignore[assignment]
    assert arr.shape == (16, 16)
    assert np.all(arr[:] == 42)


@pytest.mark.skipif(
    not omew.TensorStoreZarrStream.is_available(),
    reason="tensorstore not available",
)
def test_plate_dimension_mismatch(tmp_path: Path) -> None:
    """Test that providing mismatched plate dimensions raises an error."""
    output = tmp_path / "test_plate_mismatch.ome.zarr"

    plate_def = v05.PlateDef(
        name="Test Plate",
        rows=[v05.Row(name="A")],
        columns=[v05.Column(name="1"), v05.Column(name="2")],
        wells=[
            v05.PlateWell(path="A/1", rowIndex=0, columnIndex=0),
            v05.PlateWell(path="A/2", rowIndex=0, columnIndex=1),
        ],
        field_count=2,  # 2 wells * 2 fields = 4 positions expected
    )

    # Provide wrong number of positions (3 instead of 4)
    dims = [
        omew.Dimension(label="p", size=3),  # Wrong! Should be 4
        omew.Dimension(label="z", size=2),
        omew.Dimension(label="y", size=32),
        omew.Dimension(label="x", size=32),
    ]

    stream = omew.TensorStoreZarrStream()
    with pytest.raises(
        ValueError,
        match=r"Position dimension size \(3\) does not match plate structure "
        r"\(2 wells \* 2 fields = 4 positions\)",
    ):
        stream.create(
            str(output),
            np.dtype("uint16"),
            dims,
            overwrite=True,
            plate=plate_def,
        )
