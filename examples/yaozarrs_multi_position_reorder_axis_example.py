"""Multi-position OME-Zarr with axis reordering example using YaozarrsStream."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import zarr

import ome_writers as omew

# Define dimensions in NON-NGFF order (z and c are swapped)
# Acquisition order: p, t, z, c, y, x
# NGFF storage order (per position): t, c, z, y, x (automatic)
dimensions = [
    omew.Dimension(label="p", size=3, chunk_size=1),  # 3 positions
    omew.Dimension(label="t", size=2, chunk_size=1),
    omew.Dimension(label="z", size=3, chunk_size=1),  # Z before C
    omew.Dimension(label="c", size=2, chunk_size=1),  # C after Z
    omew.Dimension(label="y", size=32, chunk_size=16),
    omew.Dimension(label="x", size=32, chunk_size=16),
]

output_path = Path("multipos_reordered.ome.zarr").expanduser()

stream = omew.YaozarrsStream()
stream.create(str(output_path), np.dtype("uint16"), dimensions, overwrite=True)

# Append frames in acquisition order (p, t, z, c)
print("Writing frames in acquisition order: p→t→z→c→y→x")
for p in range(3):
    for t in range(2):
        for z in range(3):
            for c in range(2):
                value = p * 1000 + t * 100 + z * 10 + c
                frame = np.full((32, 32), value, dtype="uint16")
                stream.append(frame)

stream.flush()
print(f"✓ Wrote data to {output_path}")
print("  Storage order (per position) automatically changed to NGFF: t→c→z→y→x")
print()

# Verify the reordering for each position
store = zarr.open_group(output_path, mode="r")

for p in range(3):
    # Multi-position uses bioformats2raw layout: position/resolution
    array = store[f"{p}/0"]
    print(f"\nPosition {p}:")
    print(f"  Array shape: {array.shape} (t, c, z, y, x)")

    # Check a specific value to verify correct transposition
    # Acquisition: p=p, t=1, z=2, c=1 → value=p*1000+121
    # Storage: [t=1, c=1, z=2]
    expected_value = p * 1000 + 121
    actual_value = array[1, 1, 2, 0, 0]
    print(
        f"  Verification: array[t=1,c=1,z=2] = {actual_value} (expected "
        f"{expected_value}){'✓' if actual_value == expected_value else '✗'}"
    )
