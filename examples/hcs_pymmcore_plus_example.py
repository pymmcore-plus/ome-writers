"""Example of using ome_writers with useq.MDASequence and pymmcore-plus."""

import warnings
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
output_path = Path("~/Desktop/").expanduser()

# Choose backend: acquire-zarr, tensorstore, or tiff
backend = "acquire-zarr"
# backend = "tensorstore". # currently no plate support
# backend = "tiff"  # currently no plate support

# Only used if backend is "tiff". Leave True by default
tiff_memmap = True

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

# for now skip tiff and tensorstore backends since plate support is not yet implemented
if backend in ("tiff", "tensorstore"):
    raise NotImplementedError(
        f"Plate support is not yet implemented for the {backend} backend. "
        "Use acquire-zarr backend for HCS plate support."
    )

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

    # skip tiff and tensorstore for now since plate support is not yet implemented
    if backend not in ("acquire-zarr"):
        return

    # open zarr group and print structure and validate OME-NGFF JSON with yaozarrs
    try:
        import zarr
    except ImportError:
        warnings.warn(
            "Skipping zarr structure printout and OME-NGFF validation because zarr is "
            "not installed. Please install it via 'pip install zarr>=3' to enable these"
            " features.",
            stacklevel=2,
        )
        return

    try:
        from yaozarrs import validate_ome_json
    except ImportError:
        warnings.warn(
            "Skipping OME-NGFF validation because yaozarrs is not installed. Please "
            "install it via 'pip install yaozarrs' to enable this feature.",
            stacklevel=2,
        )
        return

    gp = zarr.open_group(path, mode="r")
    print(gp.tree())

    # validate OME-NGFF JSON at root
    import json

    zarr_json_path = path / "96-well" / "zarr.json"
    assert zarr_json_path.exists(), "zarr.json should exist at root"
    with open(zarr_json_path) as f:
        root_meta = json.load(f)
        validate_ome_json(json.dumps(root_meta))


# Start the acquisition
core.mda.run(seq)
