"""Example showing how to use useq grid_plan with ome-writers.

When using a grid_plan in useq-schema, the grid positions are automatically
combined with stage positions into a single position dimension. This is because
OME-NGFF and OME-TIFF do not have a separate "grid" concept - all positions
are treated the same regardless of whether they come from stage_positions or
grid_plan.

The total number of positions will be: num_stage_positions * num_grid_points
"""

import numpy as np
import useq

import ome_writers as omew

# Create a sequence with stage positions and a grid plan
# This creates 2 stage positions * 4 grid points = 8 total positions
seq = useq.MDASequence(
    axis_order="ptgcz",
    stage_positions=[(0.0, 0.0), (100.0, 100.0)],
    time_plan={"interval": 0.1, "loops": 2},
    channels=["DAPI", "FITC"],
    z_plan={"range": 3, "step": 1.0},
    grid_plan=useq.GridRowsColumns(rows=2, columns=2),
)

# Convert to ome-writers dimensions
# The grid "g" axis will be combined with "p" axis
dims = omew.dims_from_useq(seq, image_width=32, image_height=32)

print("Sequence info:")
print(f"  useq axis order: {seq.axis_order}")
print(f"  useq sizes: {dict(seq.sizes)}")
print(f"  Total frames: {len(list(seq))}")
print()
print("ome-writers dimensions:")
for dim in dims:
    print(f"  {dim.label}: size={dim.size}")
print()

# Note: The position dimension now has size=8 (2 stage * 4 grid)
p_dim = next(d for d in dims if d.label == "p")
print(
    f"Total positions: {p_dim.size} (from {len(seq.stage_positions)} stage positions * "
    f"{seq.grid_plan.num_positions()} grid points)"
)
print()

# Create a stream and write data
output_path = "test_grid_output.zarr"
with omew.create_stream(
    path=output_path,
    dimensions=dims,
    dtype=np.uint16,
    backend="zarr",
    overwrite=True,
) as stream:
    print("Writing frames...")
    for i, event in enumerate(seq):
        # Get indices
        p = event.index.get("p", 0)
        g = event.index.get("g", 0)
        t = event.index.get("t", 0)
        c = event.index.get("c", 0)
        z = event.index.get("z", 0)

        # Calculate the flattened position index
        # For axis_order="ptgcz", frames are ordered as:
        # p0g0t0c0z0, p0g0t0c0z1, ..., p0g1t0c0z0, ..., p1g0t0c0z0, ...
        flattened_p = p * seq.grid_plan.num_positions() + g

        # Create a frame (in practice, this would be your actual image data)
        frame = np.full((32, 32), i, dtype=np.uint16)

        stream.append(frame)

        if i < 5 or i == len(list(seq)) - 1:
            print(
                f"  Frame {i}: useq(p={p}, g={g}, t={t}, c={c}, z={z}) "
                f"-> flattened position={flattened_p}"
            )
        elif i == 5:
            print("  ...")

print(f"\nData written to {output_path}")
print("\nThe grid positions have been flattened into the position dimension.")
print("In the output Zarr, you'll see 8 position arrays (p/0 through p/7).")
