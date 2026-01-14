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

# or... in

settings = AcquisitionSettings(
    root_path="output.ome.zarr",
    array_settings=ArraySettings(
        dimensions=[
            Dimension(name="t", count=10, chunk_size=1, type="time"),
            PositionDimension(positions=["Pos0"]),  # type: ignore
            Dimension(name="c", count=2, chunk_size=1, type="channel"),
            Dimension(name="z", count=5, chunk_size=1, type="space"),
            Dimension(name="y", count=512, chunk_size=64, type="space"),
            Dimension(name="x", count=512, chunk_size=64, type="space"),
        ],
        dtype="uint16",
    ),
    overwrite=True,
    backend="auto",
)


with create_stream(settings) as stream:
    shape = cast("tuple[int, ...]", settings.array_settings.shape)
    for i in range(np.prod(shape[:-2])):
        frame = np.full(shape[-2:], i, dtype=settings.array_settings.dtype)
        stream.append(frame)

# Validate the output
import yaozarrs

yaozarrs.validate_zarr_store(settings.root_path)
print("✓ Zarr store is valid")
