"""Test acquire-zarr metadata correctness."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import pytest

import ome_writers as omew
from ome_writers._util import fake_data_for_sizes

if TYPE_CHECKING:
    from pathlib import Path


@pytest.mark.parametrize(
    "sizes",
    [
        {"t": 2, "c": 3, "z": 4, "y": 32, "x": 32},
        {"t": 2, "z": 4, "c": 3, "y": 32, "x": 32},
        {"c": 3, "t": 2, "z": 4, "y": 32, "x": 32},
    ],
)
def test_acquire_zarr_metadata(sizes: dict[str, int], tmp_path: Path) -> None:
    """Test that acquire-zarr metadata is correctly reordered to OME-NGFF TCZYX."""
    pytest.importorskip("acquire_zarr")

    output_path = tmp_path / "test_metadata.zarr"

    # Generate fake data and dimensions
    data_gen, dims, dtype = fake_data_for_sizes(sizes)

    # Create stream
    stream = omew.create_stream(
        path=str(output_path),
        dimensions=dims,
        dtype=dtype,
        backend="acquire-zarr",
        overwrite=True,
    )

    # Write data
    for frame in data_gen:
        stream.append(frame)

    stream.flush()

    # Check group metadata
    group_zarr_json = output_path / "zarr.json"
    assert group_zarr_json.exists()

    with open(group_zarr_json) as f:
        group_meta = json.load(f)

    # Check OME metadata axes are in TCZYX order
    ome_meta = group_meta["attributes"]["ome"]
    axes = ome_meta["multiscales"][0]["axes"]
    expected_axes = [
        {"name": "t", "type": "time", "unit": "second"},
        {"name": "c", "type": "channel", "unit": "unknown"},
        {"name": "z", "type": "space", "unit": "micrometer"},
        {"name": "y", "type": "space", "unit": "micrometer"},
        {"name": "x", "type": "space", "unit": "micrometer"},
    ]
    assert axes == expected_axes

    # Check array metadata for position 0
    array_zarr_json = output_path / "0" / "zarr.json"
    assert array_zarr_json.exists()

    with open(array_zarr_json) as f:
        array_meta = json.load(f)

    # Check dimension names are in TCZYX order
    assert array_meta["dimension_names"] == ["t", "c", "z", "y", "x"]

    # Check shape matches expected sizes in TCZYX order
    expected_shape = [sizes.get(dim, 32) for dim in ["t", "c", "z", "y", "x"]]
    assert array_meta["shape"] == expected_shape

    # Check chunk_grid configuration
    chunk_shape = array_meta["chunk_grid"]["configuration"]["chunk_shape"]
    expected_chunk_shape = [1, 1, 1, 32, 32]  # t,c,z chunk 1, y,x full
    assert chunk_shape == expected_chunk_shape

    # Check codecs chunk_shape
    codecs_chunk_shape = array_meta["codecs"][0]["configuration"]["chunk_shape"]
    assert codecs_chunk_shape == expected_chunk_shape

    # Check transpose codec is present if dimensions were reordered
    original_order = [d.label for d in dims[:-2]]  # exclude y,x
    ome_order = ["t", "c", "z"]
    if original_order != ome_order:
        # Should have transpose codec
        codecs = array_meta["codecs"][0]["configuration"]["codecs"]
        assert len(codecs) >= 1
        transpose_codec = codecs[0]
        assert transpose_codec["name"] == "transpose"
        # The order should map original to OME
        expected_order = [original_order.index(label) for label in ome_order] + [3, 4]
        assert transpose_codec["configuration"]["order"] == expected_order
