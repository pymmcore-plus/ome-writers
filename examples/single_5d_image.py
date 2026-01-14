"""Basic example of using ome_writers to write a single 5D image."""

from typing import cast

import numpy as np

from ome_writers import AcquisitionSettings, ArraySettings, Dimension, create_stream

settings = AcquisitionSettings(
    root_path="example_5d_image.ome.zarr",
    array_settings=ArraySettings(
        dimensions=[
            Dimension(name="t", count=10, chunk_size=1, type="time"),
            Dimension(name="c", count=2, chunk_size=1, type="channel"),
            Dimension(name="z", count=5, chunk_size=1, type="space", scale=5),
            Dimension(name="y", count=256, chunk_size=64, type="space", scale=0.1),
            Dimension(name="x", count=256, chunk_size=64, type="space", scale=0.1),
        ],
        dtype="uint16",
    ),
    overwrite=True,
    backend="auto",
)

shape = cast("tuple[int, ...]", settings.array_settings.shape)
with create_stream(settings) as stream:
    for i in range(np.prod(shape[:-2])):
        frame = np.full(shape[-2:], i, dtype=settings.array_settings.dtype)
        stream.append(frame)


# Validate the output
try:
    import yaozarrs

    yaozarrs.validate_zarr_store(settings.root_path)
    print("✓ Zarr store is valid")
except ImportError:
    print("⚠ yaozarrs not installed; skipping validation")
