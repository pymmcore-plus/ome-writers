"""Example of acquiring HCS data into an OME-NGFF Zarr using acquire-zarr."""

from pathlib import Path

import acquire_zarr as aqz
import numpy as np
import zarr


# Configure field of view arrays (matching ngff_zarr_hcs_pymmc_example.py specs)
# Specs:
# 96-well plate,
# 3 wells (A1, A2, B1),
# 2 FOVs per well,
# 3 timepoints,
# 1 channel,
# 512x512
def create_fov_array(fov_name: str) -> aqz.ArraySettings:
    """Create ArraySettings for a field of view."""
    return aqz.ArraySettings(
        output_key=fov_name,
        data_type=np.uint16,
        dimensions=[
            aqz.Dimension(
                name="t",
                kind=aqz.DimensionType.TIME,
                array_size_px=3,  # 3 timepoints
                chunk_size_px=1,
                shard_size_chunks=1,
            ),
            aqz.Dimension(
                name="c",
                kind=aqz.DimensionType.CHANNEL,
                array_size_px=1,  # 1 channel (FITC)
                chunk_size_px=1,
                shard_size_chunks=1,
            ),
            aqz.Dimension(
                name="y",
                kind=aqz.DimensionType.SPACE,
                array_size_px=512,
                chunk_size_px=256,
                shard_size_chunks=2,
            ),
            aqz.Dimension(
                name="x",
                kind=aqz.DimensionType.SPACE,
                array_size_px=512,
                chunk_size_px=256,
                shard_size_chunks=2,
            ),
        ],
    )


output = Path("example_az_hcs.zarr").expanduser()
# Create acquisition metadata
acquisition = aqz.Acquisition(id=0, name="Measurement_01")

# Configure wells matching ngff_zarr_hcs_pymmc_example.py (A1, A2, B1 with 2 FOVs each)
well_a1 = aqz.Well(
    row_name="A",
    column_name="1",
    images=[
        aqz.FieldOfView(
            path="fov1", acquisition_id=0, array_settings=create_fov_array("fov1")
        ),
        aqz.FieldOfView(
            path="fov2", acquisition_id=0, array_settings=create_fov_array("fov2")
        ),
    ],
)
well_a2 = aqz.Well(
    row_name="A",
    column_name="2",
    images=[
        aqz.FieldOfView(
            path="fov1", acquisition_id=0, array_settings=create_fov_array("fov1")
        ),
        aqz.FieldOfView(
            path="fov2", acquisition_id=0, array_settings=create_fov_array("fov2")
        ),
    ],
)
well_b1 = aqz.Well(
    row_name="B",
    column_name="1",
    images=[
        aqz.FieldOfView(
            path="fov1", acquisition_id=0, array_settings=create_fov_array("fov1")
        ),
        aqz.FieldOfView(
            path="fov2", acquisition_id=0, array_settings=create_fov_array("fov2")
        ),
    ],
)

# Configure the plate (96-well format)
plate = aqz.Plate(
    path="experiment_plate",
    name="96-well HCS Experiment",
    row_names=["A", "B", "C", "D", "E", "F", "G", "H"],
    column_names=["1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11", "12"],
    wells=[well_a1, well_a2, well_b1],  # 3 wells: A1, A2, B1
    acquisitions=[acquisition],
)

# Create stream with HCS configuration
settings = aqz.StreamSettings(
    store_path=str(output), overwrite=True, hcs_plates=[plate]
)

stream = aqz.ZarrStream(settings)

# Write sample data to each well and FOV
# Frame size: (1 channel, 512, 512) matching spec from ngff example
frame_data = np.random.randint(0, 2**16, (1, 512, 512), dtype=np.uint16)

# Write frames for each well and FOV (3 timepoints each)
wells = [("A", "1"), ("A", "2"), ("B", "1")]
for row, col in wells:
    for _ in range(3):  # 3 timepoints
        for fov_num in range(1, 3):  # 2 FOVs per well
            key = f"experiment_plate/{row}/{col}/fov{fov_num}"
            stream.append(frame_data, key=key)

# Close when done
stream.close()

print("Data written successfully to", output)

# open zarr group and print structure
gp = zarr.open_group(output, mode="r")
print(gp.tree())
