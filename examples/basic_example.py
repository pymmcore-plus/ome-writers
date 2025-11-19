"""Basic example of using ome_writers."""

from __future__ import annotations

from pathlib import Path

from ome_writers import create_stream, fake_data_for_sizes

plane_iter, dims, dtype = fake_data_for_sizes(
    # The order of dimensions is always determines the order in which frames will be
    # appended to the stream.
    sizes={"t": 10, "p": 2, "c": 2, "z": 7, "y": 256, "x": 256},
    chunk_sizes={"y": 64, "x": 64},
)

OUT = Path("example.zarr").expanduser()
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
