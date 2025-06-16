"""Basic example of using ome_writers."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from acquire_zarr import (
    DataType,
    Dimension,
    DimensionType,
    StreamSettings,
    ZarrStream,
    ZarrVersion,
)

from ome_writers._util import fake_data_for_sizes

if TYPE_CHECKING:
    from ome_writers import DimensionInfo


def _dim_toaqz_dim(
    dim: DimensionInfo,
    shard_size_chunks: int = 1,
) -> Dimension:
    return Dimension(
        name=dim.label,
        type=getattr(DimensionType, dim.ome_dim_type.upper()),
        array_size_px=dim.size,
        chunk_size_px=(dim.chunk_size if dim.chunk_size is not None else dim.size),
        shard_size_chunks=shard_size_chunks,
    )


sizes = {"t": 10, "z": 10, "c": 1, "y": 256, "x": 256, "p": 2}
chunks_sizes = {"y": 64, "x": 64}
output = Path("~/Desktop/some_path_ts.zarr").expanduser()

plane_iter, dims, dtype = fake_data_for_sizes(sizes=sizes, chunk_sizes=chunks_sizes)

settings = StreamSettings(
    store_path=output,
    data_type=getattr(DataType, np.dtype(dtype).name.upper()),
    version=ZarrVersion.V3,
    overwrite=True,
    output_key="",
)
settings.dimensions.extend([_dim_toaqz_dim(dim) for dim in dims])
stream = ZarrStream(settings)

for plane in plane_iter:
    stream.append(plane)
del stream  # flush

print("Data written successfully to", output)
