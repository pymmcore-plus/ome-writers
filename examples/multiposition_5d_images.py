"""Basic example of using ome_writers to write a multiposition 5D image series."""

import sys
from pathlib import Path

import numpy as np

from ome_writers import AcquisitionSettings, Dimension, PositionDimension, create_stream

# Derive format/backend from command line argument (default: auto)
FORMAT = "auto" if len(sys.argv) < 2 else sys.argv[1]

# create acquisition settings
settings = AcquisitionSettings(
    root_path="example_5d_series",
    # declare dimensions in order of acquisition (slowest to fastest)
    dimensions=[
        Dimension(name="t", count=2, chunk_size=1, type="time"),
        PositionDimension(positions=["Pos0", "Pos1"]),
        Dimension(name="c", count=3, chunk_size=1, type="channel"),
        Dimension(name="z", count=4, chunk_size=1, type="space", scale=5, unit="um"),
        Dimension(name="y", count=256, chunk_size=64, type="space", scale=2, unit="um"),
        Dimension(name="x", count=256, chunk_size=64, type="space", scale=2, unit="um"),
    ],
    dtype="uint16",
    overwrite=True,
    format=FORMAT,
)

num_frames = np.prod(settings.shape[:-2])
frame_shape = settings.shape[-2:]

# create stream and write frames
with create_stream(settings) as stream:
    for i in range(num_frames):
        frame = np.full(frame_shape, fill_value=i, dtype=settings.dtype)
        stream.append(frame)


if settings.format.name == "ome-zarr":
    import yaozarrs

    yaozarrs.validate_zarr_store(settings.output_path)
    print("✓ Zarr store is valid")

if settings.format.name == "ome-tiff":
    from ome_types import from_tiff

    for file in Path(settings.output_path).glob("*.tiff"):
        from_tiff(file)
        print(f"✓ TIFF file {file} is valid")
