"""Basic example of using ome_writers."""

from __future__ import annotations

from pathlib import Path

import acquire_zarr as az
import numpy as np

AXIS_ORDER = ["t", "c", "z", "y", "x"]
SIZES = {"t": 10, "c": 2, "z": 5, "y": 512, "x": 512}
CHUNK_SHAPES = {"t": 1, "c": 1, "z": 1, "y": 64, "x": 64}
DTYPE = np.uint16

# --------------------------------------------------
assert set(AXIS_ORDER[-2:]) == {"y", "x"}, "Last two axes must be YX"
SHAPE = tuple(SIZES[dim] for dim in AXIS_ORDER)
DATA = np.random.randint(0, 65536, size=SHAPE, dtype=DTYPE)
_extra = "".join(f"_{dim}{SIZES[dim]}" for dim in AXIS_ORDER[:-2])
DEST = Path(f"example_az{_extra}.zarr")
_kind_map = {"t": az.DimensionType.TIME, "c": az.DimensionType.CHANNEL}

array_dimensions = [
    az.Dimension(
        name=dim,
        kind=_kind_map.get(dim, az.DimensionType.SPACE),
        array_size_px=SIZES[dim],
        chunk_size_px=CHUNK_SHAPES[dim],
        shard_size_chunks=1,
    )
    for dim in AXIS_ORDER
]


settings = az.StreamSettings(
    arrays=[
        az.ArraySettings(
            dimensions=array_dimensions,
            data_type=DATA.dtype,
            compression=az.CompressionSettings(
                compressor=az.Compressor.NONE,
                codec=az.CompressionCodec.NONE,
            ),
        )
    ],
    overwrite=True,
    store_path=str(DEST),
    version=az.ZarrVersion.V3,
)
stream = az.ZarrStream(settings)

for idx in np.ndindex(SHAPE[:-2]):
    stream.append(DATA[idx])
stream.close()

print("Data written successfully to", DEST)
