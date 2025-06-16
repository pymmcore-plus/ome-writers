"""Basic example of using ome_writers."""

from __future__ import annotations

from pathlib import Path

from ome_writers import BackendName, create_stream, fake_data_for_sizes

# keys must be one of 't', 'z', 'c', 'y', 'x', or 'p'
SIZES = {"t": 10, "z": 10, "c": 1, "y": 256, "x": 256, "p": 2}
CHUNKS = {"y": 64, "x": 64}

OUT = Path("~/Desktop/some_path_ts.zarr").expanduser()
BACKEND: BackendName = "acquire-zarr"


plane_iter, dims, dtype = fake_data_for_sizes(sizes=SIZES, chunk_sizes=CHUNKS)
stream = create_stream(OUT, dtype, dims, backend=BACKEND)

for plane in plane_iter:
    stream.append(plane)
stream.flush()

print("Data written successfully to", OUT)
