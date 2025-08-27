"""Tests for ome-writers library."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from ome_writers import AcquireZarrStream, fake_data_for_sizes

data_gen, dimensions, dtype = fake_data_for_sizes(
    sizes={"t": 1, "y": 32, "x": 32},
    chunk_sizes={"t": 1, "y": 16, "x": 16},
    dtype=np.uint8,
)

# Create the stream
stream = AcquireZarrStream()
tmp_path = Path("~/Desktop").expanduser()

# Set output path
output_path = tmp_path / "test_2d_thing.zarr"

stream = stream.create(str(output_path), dtype, dimensions, overwrite=True)
assert stream.is_active()

# Get the data from the generator
for data in data_gen:
    stream.append(data)
stream.flush()

assert not stream.is_active()
assert output_path.exists()
print("Stream created and data written successfully.")
