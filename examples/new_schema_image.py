"""Basic example of using ome_writers to write a single 5D image."""

from typing import cast

import numpy as np

from ome_writers import create_stream
from ome_writers.schema import AcquisitionSettings, ArraySettings

array_settings = ArraySettings.from_standard_axes(
    sizes={"t": 10, "c": 2, "z": 5, "y": 512, "x": 512},
    dtype="uint16",
    chunk_shapes={"y": 64, "x": 64},
)

# the above is shorthand for:
# ArraySettings(
#     dimensions=[
#         Dimension(name="t", size_px=10, chunk_size_px=1, type="time"),
#         Dimension(name="c", size_px=2, chunk_size_px=1, type="channel"),
#         Dimension(name="z", size_px=5, chunk_size_px=1, type="space"),
#         Dimension(name="y", size_px=512, chunk_size_px=64, type="space"),
#         Dimension(name="x", size_px=512, chunk_size_px=64, type="space"),
#     ],
#     dtype=np.dtype("uint16"),
# )

settings = AcquisitionSettings(
    root_path="output.ome.zarr",
    arrays=[array_settings],
    overwrite=True,
    backend="auto",
)


with create_stream(settings) as stream:
    shape = cast("tuple[int, ...]", array_settings.shape)
    for i in range(np.prod(shape[:-2])):
        frame = np.full(shape[-2:], i, dtype=array_settings.dtype)
        stream.append(frame)

# Validate the output
import yaozarrs

yaozarrs.validate_zarr_store(settings.root_path)
print("✓ Zarr store is valid")
