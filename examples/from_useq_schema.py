# /// script
# requires-python = ">=3.11,<3.14"
# dependencies = [
#     "ome-writers[all]",
#     "useq-schema",
#     "rich"
# ]
#
# [tool.uv.sources]
# ome-writers = { path = "../" }
# ///
"""Example of using ome_writers with useq.MDASequence."""

import sys

import numpy as np
import useq
from rich import print

from ome_writers import AcquisitionSettings, create_stream, dims_from_useq

# Setup the AcquisitionSettings, converting the MDASequence to ome-writers Dimensions
# Derive backend from command line argument (default: auto)
BACKEND = "auto" if len(sys.argv) < 2 else sys.argv[1]
suffix = ".ome.tiff" if BACKEND == "tifffile" else ".ome.zarr"
UM = "micrometer"

# Create a MDASequence, which will be used to run the MDA with pymmcore-plus
seq = useq.MDASequence(
    axis_order="ptcz",
    stage_positions=[
        useq.Position(x=0.0, y=0.0, name="single_pos"),
        useq.Position(
            x=10.0,
            y=10.0,
            name="grid_pos",
            sequence=useq.MDASequence(
                grid_plan=useq.GridRowsColumns(rows=1, columns=2)
            ),
        ),
    ],
    time_plan={"interval": 0.1, "loops": 3},
    channels=["DAPI", "Cy5"],
    z_plan={"range": 2, "step": 1.0},
)

# Convert the useq.MDASequence to ome-writers Dimensions
# Pass pixel_size_um for x/y scale, and units dict for z scale
dims = dims_from_useq(
    seq,
    image_width=512,
    image_height=512,
    pixel_size_um=0.1,  # sets scale for x and y dimensions
    units={
        "z": (3.0, UM),  # sets scale and unit for z dimension
    },
)

# create acquisition settings
settings = AcquisitionSettings(
    root_path=f"example_useq_schema{suffix}",
    # declare dimensions in order of acquisition (slowest to fastest)
    dimensions=dims,
    dtype="uint16",
    overwrite=True,
    backend=BACKEND,
)

# calculate total number of frames and frame shape
num_frames = np.prod(settings.shape[:-2])
frame_shape = settings.shape[-2:]

# create stream and write frames
with create_stream(settings) as stream:
    for i in range(num_frames):
        frame = np.full(frame_shape, fill_value=i, dtype=settings.dtype)
        stream.append(frame)

if settings.format == "zarr":
    import yaozarrs

    yaozarrs.validate_zarr_store(settings.root_path)
    print("✓ Zarr store is valid")

if settings.format == "tiff":
    from ome_types import from_tiff

    npos = len(settings.positions)
    base = settings.root_path.replace(f"{suffix}", "")
    files = [f"{base}_p{p:03d}{suffix}" for p in range(npos)]
    for idx, file in enumerate(files):
        from_tiff(file)
        print(f"✓ TIFF file {idx} is valid")
