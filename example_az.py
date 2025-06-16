"""Basic example of using ome_writers."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import acquire_zarr as az
import numpy as np

from ome_writers._util import fake_data_for_sizes

if TYPE_CHECKING:
    from ome_writers import Dimension


def _dim_toaqz_dim(
    dim: Dimension,
    shard_size_chunks: int = 1,
) -> az.Dimension:
    return az.Dimension(
        name=dim.label,
        type=getattr(az.DimensionType, dim.ome_dim_type.upper()),
        array_size_px=dim.size,
        chunk_size_px=(dim.chunk_size if dim.chunk_size is not None else dim.size),
        shard_size_chunks=shard_size_chunks,
    )


output = Path("~/Desktop/some_path_ts.zarr").expanduser()
plane_iter, dims, dtype = fake_data_for_sizes(
    sizes={"t": 10, "z": 10, "c": 1, "y": 256, "x": 256, "p": 2},
    chunk_sizes={"y": 64, "x": 64},
)

settings = az.StreamSettings(
    store_path=str(output),
    data_type=getattr(az.DataType, np.dtype(dtype).name.upper()),
    version=az.ZarrVersion.V3,
    overwrite=True,
)
settings.dimensions.extend([_dim_toaqz_dim(dim) for dim in dims])
stream = az.ZarrStream(settings)

for plane in plane_iter:
    stream.append(plane)
del stream  # flush

print("Data written successfully to", output)
