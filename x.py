from __future__ import annotations

import time
from itertools import product

import numpy as np

from ome_writers import create_stream
from ome_writers import DimensionInfo

nt = 10
nz = 10
nc = 1
dims = [
    DimensionInfo(label="t", size=nt, unit=(1.0, "s"), chunk_size=1),
    DimensionInfo(label="z", size=nz, unit=(1.0, "um"), chunk_size=1),
    DimensionInfo(label="c", size=nc, chunk_size=1),
    DimensionInfo(label="y", size=256, unit=(1.0, "um"), chunk_size=64),
    DimensionInfo(label="x", size=256, unit=(1.0, "um"), chunk_size=64),
]
data = np.random.randint(0, 65536, size=(nt, nz, nc, 256, 256), dtype=np.uint16)

stream = create_stream("~/Desktop/some_path_ts.zarr", data.dtype, dims)
total_start = time.perf_counter_ns()
for t, z, c in product(range(nt), range(nz), range(nc)):
    start_plane = time.perf_counter_ns()
    stream.append(data[t, z, c])
    elapsed = time.perf_counter_ns() - start_plane

# Close (or flush) the stream to finalize writes.
stream.flush()
total_elapsed = time.perf_counter_ns() - total_start
tot_ms = total_elapsed / 1e6
print("Total time:", tot_ms, "ms")