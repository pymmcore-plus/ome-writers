"""Multi-position OME-Zarr writing example using YaozarrsStream."""

from __future__ import annotations

from pathlib import Path

import numpy as np

import ome_writers as omew

# Generate test data with positions
data_gen, dimensions, dtype = omew.fake_data_for_sizes(
    sizes={"p": 3, "t": 2, "c": 2, "y": 32, "x": 32},
    chunk_sizes={"p": 1, "t": 1, "c": 1, "y": 32, "x": 32},
    dtype=np.uint8,
)

output_path = Path("multiposition.ome.zarr").expanduser()

stream = omew.YaozarrsStream()
stream.create(str(output_path), dtype, dimensions, overwrite=True)

# Write all frames
for frame in data_gen:
    stream.append(frame)

stream.flush()
print(f"✓ Wrote multi-position data to {output_path}")
