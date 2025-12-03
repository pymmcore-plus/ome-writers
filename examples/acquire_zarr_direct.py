"""Basic example of using ome_writers."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from acquire_zarr import (
    ArraySettings,
    Dimension,
    DimensionType,
    StreamSettings,
    ZarrStream,
    ZarrVersion,
)

# TCZYX
dtype = np.dtype("uint16")
data = np.random.randint(0, 65536, size=(10, 2, 5, 512, 512), dtype=dtype)
nt, nc, nz, ny, nx = data.shape


dimensions = [
    Dimension(
        name="t",
        kind=DimensionType.TIME,
        array_size_px=nt,
        chunk_size_px=1,
        shard_size_chunks=1,
    ),
    Dimension(
        name="c",
        kind=DimensionType.CHANNEL,
        array_size_px=nc,
        chunk_size_px=1,
        shard_size_chunks=1,
    ),
    Dimension(
        name="z",
        kind=DimensionType.SPACE,
        array_size_px=nz,
        chunk_size_px=1,
        shard_size_chunks=1,
    ),
    Dimension(
        name="y",
        kind=DimensionType.SPACE,
        array_size_px=ny,
        chunk_size_px=64,
        shard_size_chunks=1,
    ),
    Dimension(
        name="x",
        kind=DimensionType.SPACE,
        array_size_px=nx,
        chunk_size_px=64,
        shard_size_chunks=1,
    ),
]

output = Path(f"example_az_nt{nt}_nc{nc}_nz{nz}.zarr").expanduser()
settings = StreamSettings(
    arrays=[
        ArraySettings(
            output_key="0",
            dimensions=dimensions,
            data_type=dtype,
        )
    ],
    overwrite=True,
    store_path=str(output),
    version=ZarrVersion.V3,
)
stream = ZarrStream(settings)

for t, c, z in np.ndindex(nt, nc, nz):
    stream.append(data[t, c, z])
stream.close()

print("Data written successfully to", output)
