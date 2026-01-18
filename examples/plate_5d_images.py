"""Basic example of using ome_writers to write a single 5D image."""

import sys
from typing import cast

import numpy as np

from ome_writers import (
    AcquisitionSettings,
    Dimension,
    Plate,
    Position,
    PositionDimension,
    create_stream,
)

# "tiff", "zarr", "tensorstore", "auto"
BACKEND = "auto" if len(sys.argv) < 2 else sys.argv[1]

# --------

suffix = ".ome.tiff" if BACKEND == "tiff" else ".ome.zarr"
settings = AcquisitionSettings(
    root_path=f"example_5d_plate{suffix}",
    dimensions=[
        Dimension(name="t", count=2, chunk_size=1, type="time"),
        PositionDimension(
            positions=[
                Position(name="fov0", row="A", column="1"),
                Position(name="fov0", row="A", column="2"),
                Position(name="fov0", row="C", column="4"),
                Position(name="fov1", row="C", column="4"),  # TWO fov in same well
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

shape = cast("tuple[int, ...]", settings.shape)
with create_stream(settings) as stream:
    for i in range(np.prod(shape[:-2])):
        stream.append(np.full(shape[-2:], i, dtype=settings.dtype))


if settings.format == "zarr":
    # Validate the output
    try:
        import yaozarrs

        yaozarrs.validate_zarr_store(settings.root_path)
        print("✓ Zarr store is valid")
    except ImportError:
        print("⚠ yaozarrs not installed; skipping validation")
