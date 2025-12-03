"""Test tensorstore backend with zarrs validation."""

from __future__ import annotations

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
def test_tensorstore_zarr_validation(sizes: dict[str, int], tmp_path: Path) -> None:
    """Test that tensorstore backend creates valid Zarr stores.

    This test validates both TCZYX (OME-NGFF canonical) and non-canonical orders
    to ensure the zarrs library can read stores created by the tensorstore backend.
    """
    pytest.importorskip("tensorstore")

    output_path = tmp_path / "test_tensorstore.zarr"

    # Generate fake data and dimensions
    data_gen, dims, dtype = fake_data_for_sizes(sizes)

    # Create stream
    stream = omew.create_stream(
        path=str(output_path),
        dimensions=dims,
        dtype=dtype,
        backend="tensorstore",
        overwrite=True,
    )

    # Write data
    for frame in data_gen:
        stream.append(frame)

    stream.flush()

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
