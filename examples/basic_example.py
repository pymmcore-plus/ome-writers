"""Basic example of using ome_writers."""

from __future__ import annotations

from pathlib import Path

from ome_writers import create_stream, fake_data_for_sizes

plane_iter, dims, dtype = fake_data_for_sizes(
    sizes={"t": 10, "z": 7, "c": 2, "y": 256, "x": 256},
    chunk_sizes={"y": 64, "x": 64},
)

OUT = Path("~/Desktop/some_path_ts.zarr").expanduser()
stream = create_stream(
    OUT,
    dtype,
    dims,
    backend="acquire-zarr",  # or "tensorstore", or "tiff"
    overwrite=True,
)

for plane in plane_iter:
    stream.append(plane)

stream.flush()

print("Data written successfully to", OUT)
