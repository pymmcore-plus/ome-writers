"""Example using ome_writers to write a 5D image with transposed storage order."""

import sys

import numpy as np

from ome_writers import AcquisitionSettings, Dimension, create_stream

# Derive format/backend from command line argument (default: auto)
FORMAT = "auto" if len(sys.argv) < 2 else sys.argv[1]

# create acquisition settings
settings = AcquisitionSettings(
    root_path="example_transposed_5d_image",
    # declare dimensions in order of acquisition, NOT storage order
    dimensions=[
        Dimension(name="t", count=2, chunk_size=1, type="time"),
        Dimension(name="z", count=4, chunk_size=1, type="space", scale=5),
        Dimension(name="c", count=3, chunk_size=1, type="channel"),
        Dimension(name="y", count=256, chunk_size=64, type="space", scale=0.1),
        Dimension(name="x", count=256, chunk_size=64, type="space", scale=0.1),
    ],
    # use storage_order to specify desired order on disk
    # valid options are any permutation of the dimension names
    # or the string literals: "ome" (format-compliant) or "acquisition" (as-declared)
    storage_order=["t", "c", "z", "y", "x"],  # aka, ome-zarr standard order
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
