"""Example of using ome_writers with useq.MDASequence and pymmcore-plus."""

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

try:
    from pymmcore_plus import CMMCorePlus
    from pymmcore_plus.metadata import FrameMetaV1
except ImportError as e:
    raise ImportError(
        "This example requires pymmcore-plus. Please install it via "
        "pip install pymmcore-plus"
    ) from e

# --------------------------CONFIGURATION SECTION--------------------------#
# Define output path
output_path = Path("example_pymmp_hcs.zarr")

# Choose backend: acquire-zarr, tensorstore, or tiff
backend = "acquire-zarr"
# backend = "tensorstore"
# backend = "tiff"  # currently no plate support

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

# Initialize pymmcore-plus core and load system configuration
core = CMMCorePlus.instance()
core.loadSystemConfiguration()

# Convert useq MDASequence to ome-writers Plate and Dimensions
plate = omew.plate_from_useq(seq)
dims = omew.dims_from_useq(
    seq, image_width=core.getImageWidth(), image_height=core.getImageHeight()
)

# Create an stream using the selected backend
ext = "tiff" if backend == "tiff" else "zarr"
path = output_path / f"hcs_{ext}_example.ome.{ext}"

stream = omew.create_stream(
    path=str(path),
    dimensions=dims,
    dtype=np.uint16,
    backend=backend,
    overwrite=True,
    plate=plate,
)


# Append frames to the stream on frameReady event
@core.mda.events.frameReady.connect
def _on_frame_ready(
    frame: np.ndarray, event: useq.MDAEvent, frame_meta: FrameMetaV1
) -> None:
    stream.append(frame)


# Flush and close the stream on sequenceFinished event
@core.mda.events.sequenceFinished.connect
def _on_sequence_finished(sequence: useq.MDASequence) -> None:
    stream.flush()
    print("Data written successfully to", path)

    if backend in {"acquire-zarr", "tensorstore"}:
        try:
            from yaozarrs import validate_zarr_store
        except ImportError:
            print("yaozarrs is not installed; skipping Zarr validation.")
        else:
            validate_zarr_store(path)
            print("Zarr store validated successfully.")

    elif backend == "tiff":
        try:
            import tifffile
            from ome_types import validate_xml
        except ImportError:
            print(
                "tifffile or ome-types is not installed; skipping OME-TIFF validation."
            )
        else:
            # Validate OME-TIFF metadata for each position
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


# Start the acquisition
core.mda.run(seq)
