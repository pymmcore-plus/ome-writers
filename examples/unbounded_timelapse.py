"""Example of using ome_writers with unbounded first dimension.

This simulates visiting two positions for an "unknown" number of timepoints.
(e.g. run for an hour, regardless of how long the inter-frame interval ends up being).
"""

import numpy as np

from ome_writers import (
    AcquisitionSettings,
    ArraySettings,
    Dimension,
    PositionDimension,
    create_stream,
)

settings = AcquisitionSettings(
    root_path="example_unbounded.ome.zarr",
    array_settings=ArraySettings(
        dimensions=[
            Dimension(name="t", count=None, chunk_size=1, type="time"),  # unbounded
            PositionDimension(positions=["Pos0", "Pos1"]),
            Dimension(name="y", count=256, chunk_size=64, type="space"),
            Dimension(name="x", count=256, chunk_size=64, type="space"),
        ],
        dtype="uint16",
    ),
    overwrite=True,
    backend="auto",
)

N = 3  # actual count of first dimension (in reality, could be conditional)
numframes = np.prod(settings.array_settings.shape[1:-2]) * N  # type: ignore
with create_stream(settings) as stream:
    for i in range(numframes):
        stream.append(np.full((256, 256), i, dtype=settings.array_settings.dtype))


# Validate the output
try:
    import yaozarrs

    yaozarrs.validate_zarr_store(settings.root_path)
    print("✓ Zarr store is valid")
except ImportError:
    print("⚠ yaozarrs not installed; skipping validation")
