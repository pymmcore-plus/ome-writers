"""Example of using ome_writers with unbounded first dimension.

This simulates visiting two positions for an "unknown" number of timepoints.
(e.g. run for an hour, regardless of how long the inter-frame interval ends up being).
"""

import sys

import numpy as np

from ome_writers import (
    AcquisitionSettings,
    Dimension,
    PositionDimension,
    create_stream,
)

# "tiff", "zarr", "tensorstore", "auto"
BACKEND = "auto" if len(sys.argv) < 2 else sys.argv[1]
# --------

suffix = ".ome.tiff" if BACKEND == "tiff" else ".ome.zarr"
settings = AcquisitionSettings(
    root_path=f"example_unbounded{suffix}",
    dimensions=[
        Dimension(name="t", count=None, chunk_size=1, type="time"),  # unbounded
        PositionDimension(positions=["Pos0", "Pos1"]),
        Dimension(name="y", count=256, chunk_size=64, type="space"),
        Dimension(name="x", count=256, chunk_size=64, type="space"),
    ],
    dtype="uint16",
    overwrite=True,
    backend=BACKEND,
)

N = 3  # actual count of first dimension (in reality, could be conditional)
numframes = np.prod(settings.shape[1:-2]) * N
with create_stream(settings) as stream:
    for i in range(numframes):
        stream.append(np.full((256, 256), i, dtype=settings.dtype))


if BACKEND != "tiff":
    # Validate the output
    try:
        import yaozarrs

        yaozarrs.validate_zarr_store(settings.root_path)
        print("✓ Zarr store is valid")
    except ImportError:
        print("⚠ yaozarrs not installed; skipping validation")
