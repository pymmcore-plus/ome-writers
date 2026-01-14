"""Basic example of using ome_writers to write a single 5D image."""

from typing import cast

import numpy as np

from ome_writers import (
    AcquisitionSettings,
    ArraySettings,
    Dimension,
    PositionDimension,
    create_stream,
)

dimensions = [
    Dimension(name="t", count=10, chunk_size=1, type="time"),
    PositionDimension(positions=["Pos0", "Pos1"]),
    Dimension(name="c", count=2, chunk_size=1, type="channel"),
    Dimension(name="z", count=5, chunk_size=1, type="space"),
    Dimension(name="y", count=256, chunk_size=64, type="space"),
    Dimension(name="x", count=256, chunk_size=64, type="space"),
]

# or... with the helper function:
# from ome_writers.schema_pydantic import dims_from_standard_axes
# dimensions = dims_from_standard_axes(
#     sizes={"t": 10, "p": ["Pos0", "Pos1"], "c": 2, "z": 5, "y": 256, "x": 256},
#     chunk_shapes={"y": 64, "x": 64},
# )

settings = AcquisitionSettings(
    root_path="example_5d_series.ome.zarr",
    array_settings=ArraySettings(dimensions=dimensions, dtype="uint16"),
    overwrite=True,
    backend="auto",
)

shape = cast("tuple[int, ...]", settings.array_settings.shape)
with create_stream(settings) as stream:
    for i in range(np.prod(shape[:-2])):
        stream.append(np.full(shape[-2:], i, dtype=settings.array_settings.dtype))


# Validate the output
try:
    import yaozarrs

    yaozarrs.validate_zarr_store(settings.root_path)
    print("✓ Zarr store is valid")
except ImportError:
    print("⚠ yaozarrs not installed; skipping validation")
