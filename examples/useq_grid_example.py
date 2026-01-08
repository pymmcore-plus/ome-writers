"""Example showing how to use useq grid_plan with ome-writers.

When using a grid_plan in useq-schema, the grid positions are automatically
combined with stage positions into a single position dimension. This is because
OME-NGFF and OME-TIFF do not have a separate "grid" concept - all positions
are treated the same regardless of whether they come from stage_positions or
grid_plan.

Useq supports two types of grid plans:
1. Global grid_plan: applied to all positions
   Total positions = num_stage_positions * num_grid_points
2. Per-position grid_plan: each position can have its own grid (via nested sequences)
   Total positions = sum of grid points for each position
"""

import numpy as np
import useq

import ome_writers as omew

# Example 1: Global grid plan (applied to all positions)
print("=" * 70)
print("Example 1: Global grid_plan")
print("=" * 70)
seq1 = useq.MDASequence(
    axis_order="ptgcz",
    stage_positions=[(0.0, 0.0), (100.0, 100.0)],
    time_plan={"interval": 0.1, "loops": 2},
    channels=["DAPI", "FITC"],
    z_plan={"range": 3, "step": 1.0},
    grid_plan=useq.GridRowsColumns(rows=2, columns=2),
)

dims1 = omew.dims_from_useq(seq1, image_width=32, image_height=32)
print(f"Stage positions: {len(seq1.stage_positions)}")
print(f"Grid points per position: {seq1.grid_plan.num_positions()}")
p_dim1 = next(d for d in dims1 if d.label == "p")
print(f"Total positions in ome-writers: {p_dim1.size}")
print(
    f"  (= {len(seq1.stage_positions)} stage * "
    f"{seq1.grid_plan.num_positions()} grid points)\n"
)

# Example 2: Per-position grid plan (different grids for different positions)
print("=" * 70)
print("Example 2: Per-position grid_plan (nested sequences)")
print("=" * 70)
seq2 = useq.MDASequence(
    axis_order="ptgcz",
    stage_positions=[
        useq.Position(
            x=0.0,
            y=0.0,
            sequence=useq.MDASequence(
                grid_plan=useq.GridRowsColumns(rows=1, columns=2)
            ),
        ),
        (10.0, 10.0),  # No grid for this position
    ],
    time_plan={"interval": 0.1, "loops": 2},
    channels=["DAPI", "FITC"],
    z_plan={"range": 3, "step": 1.0},
)

dims2 = omew.dims_from_useq(seq2, image_width=32, image_height=32)
print(f"Stage positions: {len(seq2.stage_positions)}")
print("  Position 0: 2 grid points")
print("  Position 1: 1 grid point (no grid)")
p_dim2 = next(d for d in dims2 if d.label == "p")
print(f"Total positions in ome-writers: {p_dim2.size}")
print("  (= 2 + 1 = 3 unique position combinations)\n")

# Use the second example for demonstration
seq = seq2
dims = dims2

print("=" * 70)
print("Writing data with per-position grid")
print("=" * 70)

# Count actual unique positions
unique_positions = set()
for event in seq:
    p = event.index.get("p", 0)
    g = event.index.get("g", 0)
    unique_positions.add((p, g))
print(f"Unique (p, g) combinations: {sorted(unique_positions)}")

# Convert to ome-writers dimensions
print("\nome-writers dimensions:")
for dim in dims:
    print(f"  {dim.label}: size={dim.size}")
print()

# Create a stream and write data
output_path = "example_test_grid_output.zarr"
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

        # Create a frame (in practice, this would be your actual image data)
        frame = np.full((32, 32), i, dtype=np.uint16)

        stream.append(frame)

        if i < 5 or i == len(list(seq)) - 1:
            print(f"  Frame {i}: useq(p={p}, g={g}, t={t}, c={c}, z={z})")
        elif i == 5:
            print("  ...")

print(f"\nData written to {output_path}")
print("\nThe grid positions have been flattened into the position dimension.")
print(f"In the output Zarr, you'll see {p_dim2.size} position arrays.")
