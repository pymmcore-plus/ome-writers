"""Basic example of using ome_writers to write a multiposition 5D image series."""

import sys

import numpy as np

from ome_writers import AcquisitionSettings, Dimension, PositionDimension, create_stream

# Derive backend from command line argument (default: auto)
BACKEND = "auto" if len(sys.argv) < 2 else sys.argv[1]
suffix = ".ome.tiff" if BACKEND == "tifffile" else ".ome.zarr"
UM = "micrometer"

# create acquisition settings
settings = AcquisitionSettings(
    root_path=f"example_5d_series{suffix}",
    # declare dimensions in order of acquisition (slowest to fastest)
    dimensions=[
        Dimension(name="t", count=2, chunk_size=1, type="time"),
        PositionDimension(positions=["Pos0", "Pos1"]),
        Dimension(name="c", count=3, chunk_size=1, type="channel"),
        Dimension(name="z", count=4, chunk_size=1, type="space", scale=5, unit=UM),
        Dimension(name="y", count=256, chunk_size=64, type="space", scale=2, unit=UM),
        Dimension(name="x", count=256, chunk_size=64, type="space", scale=2, unit=UM),
    ],
    dtype="uint16",
    overwrite=True,
    backend=BACKEND,
)

num_frames = np.prod(settings.shape[:-2])
frame_shape = settings.shape[-2:]

# create stream and write frames
with create_stream(settings) as stream:
    for i in range(num_frames):
        frame = np.full(frame_shape, fill_value=i, dtype=settings.dtype)
        stream.append(frame)


if settings.format == "zarr":
    import yaozarrs

    yaozarrs.validate_zarr_store(settings.root_path)
    print("✓ Zarr store is valid")

if settings.format == "tiff":
    from ome_types import from_tiff

    files = [f"{settings.root_path[:-9]}_p{pos:03d}.ome.tiff" for pos in range(2)]
    for idx, file in enumerate(files):
        from_tiff(file)
        print(f"✓ TIFF file {idx} is valid")
