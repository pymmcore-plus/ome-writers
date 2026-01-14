"""Basic example of using ome_writers to write a single 5D image."""

from typing import cast

import numpy as np

from ome_writers import create_stream
from ome_writers.schema_pydantic import (
    AcquisitionSettings,
    ArraySettings,
    Dimension,
    PositionDimension,
)

dimensions = [
    Dimension(name="t", count=10, chunk_size=1, type="time"),
    PositionDimension(positions=["Pos0"]),  # type: ignore
    Dimension(name="c", count=2, chunk_size=1, type="channel"),
    Dimension(name="z", count=5, chunk_size=1, type="space"),
    Dimension(name="y", count=512, chunk_size=64, type="space"),
    Dimension(name="x", count=512, chunk_size=64, type="space"),
]

# or... use the helper function:
# from ome_writers.schema_pydantic import dims_from_standard_axes
# dimensions = dims_from_standard_axes(
#     sizes={"t": 10, "p": ["Pos0"], "c": 2, "z": 5, "y": 512, "x": 512},
#     chunk_shapes={"y": 64, "x": 64},
# )

settings = AcquisitionSettings(
    root_path="output.ome.zarr",
    array_settings=ArraySettings(dimensions=dimensions, dtype="uint16"),
    overwrite=True,
    backend="auto",
)

shape = cast("tuple[int, ...]", settings.array_settings.shape)
dtype = settings.array_settings.dtype
with create_stream(settings) as stream:
    for i in range(np.prod(shape[:-2])):
        stream.append(np.full(shape[-2:], i, dtype=dtype))

# Validate the output
try:
    import yaozarrs

    yaozarrs.validate_zarr_store(settings.root_path)
    print("✓ Zarr store is valid")
except ImportError:
    print("⚠ yaozarrs not installed; skipping validation")
