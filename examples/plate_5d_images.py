"""Basic example of using ome_writers to write OME HCS plate format."""

import sys

import numpy as np

from ome_writers import (
    AcquisitionSettings,
    Dimension,
    Plate,
    Position,
    PositionDimension,
    create_stream,
)

# Derive backend from command line argument (default: auto)
BACKEND = "auto" if len(sys.argv) < 2 else sys.argv[1]
suffix = ".ome.tiff" if BACKEND == "tifffile" else ".ome.zarr"

# create acquisition settings
settings = AcquisitionSettings(
    root_path=f"example_5d_plate{suffix}",
    # declare dimensions in order of acquisition (slowest to fastest)
    dimensions=[
        Dimension(name="t", count=2, chunk_size=1, type="time"),
        PositionDimension(
            # order should match stage position traversal during acquisition
            positions=[
                Position(name="fov0", plate_row="A", plate_column="1"),
                Position(name="fov0", plate_row="A", plate_column="2"),
                # note ... two fovs in same well
                Position(name="fov0", plate_row="C", plate_column="4"),
                Position(name="fov1", plate_row="C", plate_column="4"),
            ]
        ),
        Dimension(name="c", count=3, chunk_size=1, type="channel"),
        Dimension(name="z", count=4, chunk_size=1, type="space"),
        Dimension(name="y", count=256, chunk_size=64, type="space"),
        Dimension(name="x", count=256, chunk_size=64, type="space"),
    ],
    dtype="uint16",
    plate=Plate(
        name="Example Plate",
        row_names=["A", "B", "C", "D"],
        column_names=["1", "2", "3", "4", "5", "6", "7", "8"],
    ),
    overwrite=True,
    backend=BACKEND,
)

num_frames = np.prod(settings.shape[:-2])
frame_shape = settings.shape[-2:]

# create stream and write frames
with create_stream(settings) as stream:
    for i in range(num_frames):
        stream.append(np.full(frame_shape, fill_value=i, dtype=settings.dtype))


if settings.format == "zarr":
    import yaozarrs

    yaozarrs.validate_zarr_store(settings.root_path)
    print("✓ Zarr store is valid")
