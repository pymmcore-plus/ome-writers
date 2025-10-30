"""Example of using ome_writers with useq.MDASequence."""

from contextlib import suppress
from pathlib import Path

import numpy as np
from pytest import skip

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
output_path = Path("~/Desktop/").expanduser()

# Choose backend: acquire-zarr, tensorstore, or tiff
# backend = "acquire-zarr"
backend = "tensorstore"
# backend = "tiff"

# Create a simple plate plan with 3 wells, 3 fov per well
plate_plan = useq.WellPlatePlan(
    plate="96-well",
    a1_center_xy=(0.0, 0.0),
    selected_wells=([0, 0, 1], [0, 1, 0]),  # A1, A2, B1
    well_points_plan=useq.GridRowsColumns(rows=1, columns=2),  # 2 FOV per well
)

# Create a MDASequence. NOTE: the axis_order determines the order in which frames will
# be appended to the stream.
seq = useq.MDASequence(
    axis_order="ptc",
    stage_positions=plate_plan,
    time_plan={"interval": 0.1, "loops": 3},
    channels=["FITC"],
)
# -------------------------------------------------------------------------#

# Convert the MDASequence to ome_writers dimensions
dims = omew.dims_from_useq(seq, image_width=32, image_height=32)

# Create an stream using the selected backend
ext = "tiff" if backend == "tiff" else "zarr"
path = output_path / f"{ext}_example.ome.{ext}"

stream = omew.create_stream(
    path=str(path),
    dimensions=dims,
    dtype=np.uint8,
    backend=backend,
    plate=omew.plate_from_useq(seq),
    overwrite=True,
)

# Simulate acquisition and append frames to the stream iterating over the MDASequence
for event in seq:
    print(f"Event Index: {event.index}")
    # create a dummy frame
    frame = np.random.randint(0, 255, size=(32, 32), dtype=np.uint8)
    stream.append(frame)

stream.flush()

print("Data written successfully to", output_path / f"{ext}_example.ome.{ext}")


# --------------------------------------------------------------------------------
# skip tiff and tensorstore for now since plate support is not yet implemented
if backend not in ("acquire-zarr"):
    skip("Plate support is not yet implemented for this backend.")
# --------------------------------------------------------------------------------

if backend in {"acquire-zarr", "tensorstore"}:
    with suppress(ImportError):
        from yaozarrs import validate_zarr_store

        validate_zarr_store(path)
        print("Zarr store validated successfully.")

elif backend == "tiff":
    with suppress(ImportError):
        import tifffile
        from ome_types import validate_xml
        n_pos = len(seq.stage_positions)
        for pos in range(len(seq.stage_positions)):
            if n_pos == 1:
                tiff_path = path
            else:
                tiff_path = output_path / f"{ext}_example_p{pos:03d}.ome.{ext}"
            with tifffile.TiffFile(tiff_path) as tif:
                assert tif.ome_metadata is not None
                validate_xml(tif.ome_metadata)
                print(f"OME-TIFF file for position {pos} validated successfully.")
