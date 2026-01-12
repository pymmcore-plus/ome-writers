"""Basic example of using ome_writers to write a single 5D image."""

import numpy as np

from ome_writers import create_stream
from ome_writers.schema import AcquisitionSettings, ArraySettings, Dimension

settings = AcquisitionSettings(
    root_path="output.ome.zarr",
    arrays=[
        ArraySettings(
            dimensions=[
                Dimension(name="t", size_px=10, chunk_size_px=1, type="time"),
                Dimension(name="c", size_px=2, chunk_size_px=1, type="channel"),
                Dimension(name="z", size_px=5, chunk_size_px=1, type="space"),
                Dimension(name="y", size_px=512, chunk_size_px=64, type="space"),
                Dimension(name="x", size_px=512, chunk_size_px=64, type="space"),
            ],
            dtype=np.dtype("uint16"),
        )
    ],
    overwrite=True,
    backend="auto",
)

# convenience method
settings = AcquisitionSettings(
    root_path="output.ome.zarr",
    arrays=[
        ArraySettings.from_standard_axes(
            sizes={"t": 10, "c": 2, "z": 5, "y": 512, "x": 512},
            dtype="uint16",
            chunk_shapes={"y": 64, "x": 64},
        )
    ],
    overwrite=True,
    backend="auto",
)


stream = create_stream(settings)  # type: ignore
# for frame in data:
#     stream.append(frame)
