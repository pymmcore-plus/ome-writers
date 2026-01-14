"""Benchmark different Zarr backends for writing OME-Zarr data."""

from __future__ import annotations

import time
from pathlib import Path

from ome_writers import fake_data_for_sizes
from ome_writers.backends import _old_tensorstore, _yaozarrs


def _run_bench(cls: type) -> float:
    # Generate fake data with specified sizes and chunk sizes
    plane_iter, dims, dtype = fake_data_for_sizes(
        # The order of dimensions is always determines the order in which frames will be
        # appended to the stream.
        sizes={"p": 2, "t": 10, "c": 2, "z": 7, "y": 1024, "x": 1024},
        chunk_sizes={"y": 128, "x": 128},
    )

    # setup output path and create stream
    output = Path(f"bench_{cls.__module__}_{cls.__name__}.zarr").expanduser()
    start = time.perf_counter()
    stream = cls()
    stream.create(str(output), dtype, dims, overwrite=True)
    for plane in plane_iter:
        stream.append(plane)

    stream.flush()
    stop = time.perf_counter()
    return stop - start


time1 = _run_bench(_yaozarrs.TensorStoreZarrStream)
print(f"New yaozarrs stream time: {time1:.2f} seconds")

time1 = _run_bench(_old_tensorstore.OldTensorStoreZarrStream)
print(f"Old tensorstore stream time: {time1:.2f} seconds")

time1 = _run_bench(_yaozarrs.ZarrPythonStream)
print(f"new zp stream time: {time1:.2f} seconds")
