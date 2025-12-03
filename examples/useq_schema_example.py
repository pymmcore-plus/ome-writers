"""Example of using ome_writers with useq.MDASequence."""

from pathlib import Path

import numpy as np

import ome_writers as omew

try:
    import useq
except ImportError as e:
    raise ImportError(
        "This example requires useq-schema. Please install it via "
        "pip install useq-schema"
    ) from e

# --------------------------CONFIGURATION SECTION--------------------------#
# Define output path
output = Path("example_useq_schema").expanduser()

# Choose backend: acquire-zarr, tensorstore, or tiff
backend = "tensorstore"
# backend = "acquire-zarr"
# backend = "tiff"

# Create a MDASequence.
# NOTE: axis_order determines the order in which frames will be appended to the stream.
seq = useq.MDASequence(
    axis_order="pzc",
    stage_positions=[(0.0, 0.0), (10.0, 10.0)],
    time_plan={"interval": 0.5, "loops": 10},
    channels=["DAPI", "FITC"],
    z_plan={"range": 2, "step": 1.0},
)
# -------------------------------------------------------------------------#

# Convert the MDASequence to ome_writers dimensions
dims = omew.dims_from_useq(seq, image_width=32, image_height=32)

# Create an stream using the selected backend
ext = "tiff" if backend == "tiff" else "zarr"
path = output / f"{ext}_example.ome.{ext}"

stream = omew.create_stream(
    path=str(path),
    dimensions=dims,
    dtype=np.uint8,
    backend=backend,
    overwrite=True,
)

# Simulate acquisition and append frames to the stream iterating over the MDASequence
for event in seq:
    print(f"Event Index: {event.index}")
    # create a dummy frame
    frame = np.random.randint(0, 255, size=(32, 32), dtype=np.uint8)
    stream.append(frame)

stream.flush()

print("Data written successfully to", path)
