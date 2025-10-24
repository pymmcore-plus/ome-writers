import acquire_zarr as aqz
import numpy as np

# Import specific classes to avoid aliasing issues
from acquire_zarr import Acquisition, FieldOfView, Plate, Well
from rich import print
from useq import (
    GridRowsColumns,
    MDASequence,
    WellPlatePlan,
    register_well_plates,
)

import ome_writers as omew

register_well_plates(
    {
        "1536-well": {
            "rows": 32,
            "columns": 48,
            "well_spacing": 2.25,
            "well_size": 1.55,
        }
    }
)

# Create a simple plate plan with 3 wells
plate_plan = WellPlatePlan(
    plate="1536-well",
    a1_center_xy=(0.0, 0.0),
    selected_wells=([30, 28, 1], [32, 10, 0]),
    well_points_plan=GridRowsColumns(rows=1, columns=2),
)
seq = MDASequence(stage_positions=plate_plan)

plate = omew.plate_from_useq(seq)

ngff_plate = omew.plate_to_yaozarrs_v5(plate)

print(ngff_plate.model_dump())

# Configure field of view arrays for the plate
fov1_array = aqz.ArraySettings(
    output_key="fov1",  # Relative to the well: plate/A/1/fov1
    data_type=np.uint16,
    downsampling_method=aqz.DownsamplingMethod.MEAN,
    dimensions=[
        aqz.Dimension(
            name="t",
            kind=aqz.DimensionType.TIME,
            array_size_px=0,
            chunk_size_px=10,
            shard_size_chunks=1,
        ),
        aqz.Dimension(
            name="c",
            kind=aqz.DimensionType.CHANNEL,
            array_size_px=3,
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

fov2_array = aqz.ArraySettings(
    output_key="fov2",
    data_type=np.uint16,
    downsampling_method=aqz.DownsamplingMethod.MEAN,
    dimensions=[
        aqz.Dimension(
            name="t",
            kind=aqz.DimensionType.TIME,
            array_size_px=0,
            chunk_size_px=10,
            shard_size_chunks=1,
        ),
        aqz.Dimension(
            name="c",
            kind=aqz.DimensionType.CHANNEL,
            array_size_px=3,
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

# Create acquisition metadata
acquisition = Acquisition(
    id=0,
    name="Acquisition 1",
    start_time=1343731272000,  # Unix timestamp in milliseconds
)

# Configure wells with fields of view
# For each well in the plate, create a Well object
wells_aqz = []
for well_pos in plate.wells:
    # Each well has 2 FOV (from GridRowsColumns(rows=1, columns=2))
    well = Well(
        row_name=well_pos.path.split("/")[0],
        column_name=well_pos.path.split("/")[1],
        images=[
            FieldOfView(
                path="fov1",
                acquisition_id=0,
                array_settings=fov1_array,
            ),
            FieldOfView(
                path="fov2",
                acquisition_id=0,
                array_settings=fov2_array,
            ),
        ],
    )
    wells_aqz.append(well)

# Configure the HCS plate for acquire-zarr
plate_aqz = Plate(
    path="experiment_plate",
    name=plate.name or "1536-well Plate",
    row_names=plate.rows,
    column_names=plate.columns,
    wells=wells_aqz,
    acquisitions=[acquisition],
)

# Create stream with HCS configuration
settings = aqz.StreamSettings(
    store_path="/Users/fdrgsp/Desktop/t/acq_z.zarr",
    overwrite=True,
    hcs_plates=[plate_aqz],
)

stream = aqz.ZarrStream(settings)

# Write data to specific fields of view
for well_pos in plate.wells:
    row_name = well_pos.path.split("/")[0]
    col_name = well_pos.path.split("/")[1]

    # Write 2 time points for each FOV
    for _ in range(2):
        # FOV 1
        frame_data = np.random.randint(0, 2**16, (3, 512, 512), dtype=np.uint16)
        stream.append(frame_data, key=f"experiment_plate/{row_name}/{col_name}/fov1")

        # FOV 2
        frame_data = np.random.randint(0, 2**16, (3, 512, 512), dtype=np.uint16)
        stream.append(frame_data, key=f"experiment_plate/{row_name}/{col_name}/fov2")

# Close when done
stream.close()

print("\n✅ HCS Zarr store created successfully!")
print("   Path: /Users/fdrgsp/Desktop/t/acq_z.zarr")
print(f"   Wells: {len(plate.wells)}")
print(f"   Fields per well: {plate.field_count}")
print("   Time points: 2")
