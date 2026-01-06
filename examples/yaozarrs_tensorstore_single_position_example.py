"""Single-position OME-Zarr writing example using TensorStoreZarrStream."""

from __future__ import annotations

from pathlib import Path

import numpy as np

import ome_writers as omew

# Generate test data
data_gen, dimensions, dtype = omew.fake_data_for_sizes(
    sizes={"t": 3, "c": 2, "z": 4, "y": 64, "x": 64},
    chunk_sizes={"t": 1, "c": 1, "z": 1, "y": 32, "x": 32},
    dtype=np.uint16,
)

output_path = Path("example_yzr_single_pos.ome.zarr").expanduser()

# Create stream and write data
stream = omew.TensorStoreZarrStream()
stream.create(str(output_path), dtype, dimensions, overwrite=True)

# Write all frames
for frame in data_gen:
    stream.append(frame)

stream.flush()
print(f"✓ Wrote single-position data to {output_path}")
