"""Test tensorstore backend with zarrs validation."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

import ome_writers as omew
from ome_writers._util import fake_data_for_sizes

from .conftest import validate_path

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
    validate_path(output_path)
