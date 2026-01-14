"""Basic example of using ome_writers to write a single 5D image."""

from typing import cast

import numpy as np

from ome_writers import AcquisitionSettings, ArraySettings, Dimension, create_stream

settings = AcquisitionSettings(
    root_path="example_unbounded.ome.zarr",
    array_settings=ArraySettings(
        dimensions=[
            Dimension(name="z", count=5, chunk_size=1, type="space"),
            Dimension(name="t", count=None, chunk_size=1, type="time"),  # unbounded
            Dimension(name="y", count=256, chunk_size=64, type="space"),
            Dimension(name="x", count=256, chunk_size=64, type="space"),
        ],
        dtype="uint16",
        storage_order="ngff",
    ),
    overwrite=True,
    backend="auto",
)

ntime = 3
actual_shape = cast("tuple[int, ...]", (ntime, *settings.array_settings.shape[1:]))
with create_stream(settings) as stream:
    for i in range(np.prod(actual_shape[:-2])):
        stream.append(
            np.full(actual_shape[-2:], i, dtype=settings.array_settings.dtype)
        )


# Validate the output
try:
    import yaozarrs

    yaozarrs.validate_zarr_store(settings.root_path)
    print("✓ Zarr store is valid")
except ImportError:
    print("⚠ yaozarrs not installed; skipping validation")
