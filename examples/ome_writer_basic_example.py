"""Basic example of using ome_writers."""

from __future__ import annotations

from pathlib import Path

from ome_writers import create_stream, fake_data_for_sizes

# Generate fake data with specified sizes and chunk sizes
plane_iter, dims, dtype = fake_data_for_sizes(
    # NOTE: For "tensorstore" and "zarr" backends, the dimension order in the output
    # is always normalized to the NGFF specification, which is TCZYX (Time, Channel, Z,
    # Y, X), regardless of the acquisition order specified in the sizes dict below.
    sizes={"t": 10, "p": 2, "c": 2, "z": 7, "y": 256, "x": 256},
    chunk_sizes={"y": 64, "x": 64},
)

# setup output path and create stream
output = Path("example_basic.ome.zarr").expanduser()
stream = create_stream(
    output,
    dtype,
    dims,
    backend="tensorstore",  # or "acquire-zarr", "zarr" or "tiff"
    overwrite=True,
)

# Write data
for plane in plane_iter:
    stream.append(plane)

stream.flush()

print("Data written successfully to", output)
