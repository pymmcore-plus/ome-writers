"""Basic example of using ome_writers."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from acquire_zarr import (
    DataType,
    Dimension,
    DimensionType,
    StreamSettings,
    ZarrStream,
    ZarrVersion,
)

# TCZYX
data = np.random.randint(0, 65536, size=(10, 2, 5, 512, 512), dtype=np.uint16)
nt, nc, nz, ny, nx = data.shape

output = Path("~/Desktop/some_path_ts.zarr").expanduser()
settings = StreamSettings(
    store_path=str(output),
    data_type=DataType.UINT16,
    version=ZarrVersion.V3,
    overwrite=True,
)
settings.dimensions.extend(
    [
        Dimension(
            name="t",
            type=DimensionType.TIME,
            array_size_px=nt,
            chunk_size_px=1,
            shard_size_chunks=1,
        ),
        Dimension(
            name="c",
            type=DimensionType.CHANNEL,
            array_size_px=nc,
            chunk_size_px=1,
            shard_size_chunks=1,
        ),
        Dimension(
            name="z",
            type=DimensionType.SPACE,
            array_size_px=nz,
            chunk_size_px=1,
            shard_size_chunks=1,
        ),
        Dimension(
            name="y",
            type=DimensionType.SPACE,
            array_size_px=ny,
            chunk_size_px=64,
            shard_size_chunks=1,
        ),
        Dimension(
            name="x",
            type=DimensionType.SPACE,
            array_size_px=nx,
            chunk_size_px=64,
            shard_size_chunks=1,
        ),
    ]
)
stream = ZarrStream(settings)

for t, c, z in np.ndindex(nt, nc, nz):
    stream.append(data[t, c, z])
del stream  # flush

print("Data written successfully to", output)
