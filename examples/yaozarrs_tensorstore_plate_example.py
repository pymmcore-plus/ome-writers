"""Example of using ome_writers with plate (HCS) metadata.

This example demonstrates how to create an OME-Zarr plate structure using
yaozarrs plate metadata with the TensorStoreZarrStream backend.
"""

from pathlib import Path

import numpy as np

import ome_writers as omew

try:
    from yaozarrs import v05
except ImportError as e:
    raise ImportError(
        "This example requires yaozarrs. Please install it via "
        "pip install yaozarrs[write-tensorstore]"
    ) from e

# --------------------------CONFIGURATION SECTION--------------------------#
# Define output path
output = Path("example_plate.ome.zarr").expanduser()

# Define plate layout
plate_def = v05.PlateDef(
    name="Example HCS Plate",
    rows=[v05.Row(name="A"), v05.Row(name="B")],
    columns=[v05.Column(name="1"), v05.Column(name="2")],
    wells=[
        v05.PlateWell(path="A/1", rowIndex=0, columnIndex=0),
        v05.PlateWell(path="A/2", rowIndex=0, columnIndex=1),
        v05.PlateWell(path="B/1", rowIndex=1, columnIndex=0),
        v05.PlateWell(path="B/2", rowIndex=1, columnIndex=1),
    ],
    field_count=2,  # 2 fields of view per well
    acquisitions=[
        v05.Acquisition(
            id=0,
            name="Initial scan",
            maximumfieldcount=2,
        ),
    ],
)

# Define image dimensions (same for all wells/fields)
# Note: When using plate layout, you need a 'p' (position) dimension
# that matches the total number of positions (wells * fields_per_well)
total_positions = len(plate_def.wells) * (plate_def.field_count or 1)
dims = [
    omew.Dimension(label="p", size=total_positions),  # 4 wells * 2 fields = 8 positions
    omew.Dimension(label="c", size=2),  # 2 channels
    omew.Dimension(label="z", size=3),  # 3 z-slices
    omew.Dimension(label="y", size=64),  # 64x64 pixels
    omew.Dimension(label="x", size=64),
]
# -------------------------------------------------------------------------#

# Create the stream with plate metadata
stream = omew.TensorStoreZarrStream()
stream.create(
    str(output),
    np.dtype("uint16"),
    dims,
    overwrite=True,
    plate=plate_def,
)

# Write data for each position (well/field combination)
# The stream automatically maps position index to well/field path
position_idx = 0
for well in plate_def.wells:
    for field_idx in range(plate_def.field_count or 1):
        print(
            f"Writing well {well.path}, field {field_idx} (position {position_idx})..."
        )

        # Write frames for this position
        for c in range(dims[1].size):  # Note: dims[0] is 'p', so channels are dims[1]
            for z in range(dims[2].size):
                # Create unique values for each position/channel/z combination
                value = (position_idx * 1000) + (c * 10) + z
                frame = np.full((dims[3].size, dims[4].size), value, dtype="uint16")
                stream.append(frame)

        position_idx += 1

stream.flush()
print(f"\n✓ Wrote HCS plate data to {output}")
print(f"  Wells: {len(plate_def.wells)}")
print(f"  Fields per well: {plate_def.field_count}")
print(f"  Total images: {len(plate_def.wells) * (plate_def.field_count or 1)}")
