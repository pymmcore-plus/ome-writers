"""Example of using ome_writers with useq.MDASequence."""

from pathlib import Path

import numpy as np
from useq import MDASequence

import ome_writers as omew

# Define output path
output_path = Path("~/Desktop/some_path_ts.zarr").expanduser()

# Choose backend: acquire-zarr, tensorstore, or tiff
backend = "acquire-zarr"
# backend = "tensorstore"
# backend = "tiff"

# Create a MDASequence. NOTE: the axis_order determines the order in which frames will
# be appended to the stream.
seq = MDASequence(
    axis_order="tpcz",
    stage_positions=[(0.0, 0.0), (10.0, 10.0)],
    time_plan={"interval": 0.5, "loops": 5},
    channels=["DAPI", "FITC"],
    z_plan={"range": 3, "step": 1.0},
)

# Convert the MDASequence to ome_writers dimensions
dims = omew.dims_from_useq(seq, image_width=32, image_height=32)

# Create an OME-ZARR stream using the acquire-zarr backend
stream = omew.create_stream(
    path=output_path,
    dimensions=dims,
    dtype=np.uint8,
    backend=backend,
    overwrite=True,
)

# Simulate acquisition and append frames to the stream iterating over the MDASequence
for event in seq:
    p = event.index.get("p", 0)
    t = event.index.get("t", 0)
    c = event.index.get("c", 0)
    # create a dummy frame
    frame = np.random.randint(0, 255, size=(32, 32), dtype=np.uint8)
    stream.append(frame)

stream.flush()

print("Data written successfully to", output_path)
