"""Example of using ome_writers with unbounded first dimension.

This simulates visiting two positions for an "unknown" number of timepoints.
(e.g. run for an hour, regardless of how long the inter-frame interval ends up being).
"""

import sys

import numpy as np

from ome_writers import AcquisitionSettings, Dimension, PositionDimension, create_stream

# Derive backend from command line argument (default: auto)
BACKEND = "auto" if len(sys.argv) < 2 else sys.argv[1]
suffix = ".ome.tiff" if BACKEND == "tifffile" else ".ome.zarr"

# create acquisition settings
settings = AcquisitionSettings(
    root_path=f"example_unbounded{suffix}",
    # declare dimensions in order of acquisition (slowest to fastest)
    dimensions=[
        # count=None makes this an unbounded dimension
        # only the first dimension can be unbounded
        Dimension(name="t", count=None, chunk_size=1, type="time"),
        PositionDimension(positions=["Pos0", "Pos1"]),
        Dimension(name="y", count=256, chunk_size=64, type="space"),
        Dimension(name="x", count=256, chunk_size=64, type="space"),
    ],
    dtype="uint16",
    overwrite=True,
    backend=BACKEND,
)

# actual count of first dimension for the sake of this example
# (in reality, it would likely be conditional, e.g. in a generator loop)
actual_count = 3
numframes = np.prod(settings.shape[1:-2]) * actual_count
frame_shape = settings.shape[-2:]

with create_stream(settings) as stream:
    for i in range(numframes):
        stream.append(np.full(frame_shape, fill_value=i, dtype=settings.dtype))

if settings.format == "zarr":
    import yaozarrs

    yaozarrs.validate_zarr_store(settings.root_path)
    print("✓ Zarr store is valid")
