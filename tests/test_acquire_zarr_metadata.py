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
        {"t": 2, "c": 3, "z": 4, "y": 32, "x": 32},  # TCZYX - OME-NGFF canonical order
        {"t": 2, "z": 4, "c": 3, "y": 32, "x": 32},  # TZCYX - non-canonical
        {"c": 3, "t": 2, "z": 4, "y": 32, "x": 32},  # CTZYX - non-canonical
    ],
)
def test_acquire_zarr_metadata(sizes: dict[str, int], tmp_path: Path) -> None:
    """Test that acquire-zarr metadata preserves acquisition order."""
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

    # Check OME metadata axes are in acquisition order (not TCZYX)
    ome_meta = group_meta["attributes"]["ome"]
    axes = ome_meta["multiscales"][0]["axes"]

    # Build expected axes based on actual dimension order from dims
    expected_axes = []
    axis_info = {
        "t": {"type": "time", "unit": "second"},
        "c": {"type": "channel", "unit": "unknown"},
        "z": {"type": "space", "unit": "micrometer"},
        "y": {"type": "space", "unit": "micrometer"},
        "x": {"type": "space", "unit": "micrometer"},
    }
    for dim in dims:
        if dim.label in axis_info:
            expected_axes.append({"name": dim.label, **axis_info[dim.label]})

    assert axes == expected_axes

    # Check array metadata for position 0
    array_zarr_json = output_path / "0" / "zarr.json"
    assert array_zarr_json.exists()

    with open(array_zarr_json) as f:
        array_meta = json.load(f)

    # Check dimension names are in acquisition order (not TCZYX)
    expected_dim_names = [d.label for d in dims]
    assert array_meta["dimension_names"] == expected_dim_names

    # Check shape matches expected sizes in acquisition order
    expected_shape = [d.size for d in dims]
    assert array_meta["shape"] == expected_shape

    # Check chunk_grid configuration
    chunk_shape = array_meta["chunk_grid"]["configuration"]["chunk_shape"]
    expected_chunk_shape = [
        d.chunk_size if d.chunk_size is not None else d.size for d in dims
    ]
    assert chunk_shape == expected_chunk_shape

    # Check codecs chunk_shape
    codecs_chunk_shape = array_meta["codecs"][0]["configuration"]["chunk_shape"]
    assert codecs_chunk_shape == expected_chunk_shape

    try:
        from yaozarrs import validate_zarr_store
    except ImportError:
        print("yaozarrs not installed, skipping zarr store validation")
        return

    # Validate the Zarr store using zarrs library when dimensions follow canonical order
    # (the OME-NGFF v0.5 canonical order: [time,] [channel,] space)
    # The zarrs validator requires axes to be in the order: [time,] [channel,] space
    # Even if some dimensions are missing, validate that present ones maintain order
    dim_labels = [d.label for d in dims if d.label not in "pyx"]
    canonical_order = ["t", "c", "z"]
    filtered_canonical = [d for d in canonical_order if d in dim_labels]
    is_canonical_order = dim_labels == filtered_canonical
    if is_canonical_order:
        validate_zarr_store(output_path)
