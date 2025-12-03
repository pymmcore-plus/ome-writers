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
output = Path("example_pymmcore_plus").expanduser()

# Choose backend: acquire-zarr, tensorstore, or tiff
backend = "tensorstore"
# backend = "acquire-zarr"
# backend = "tiff"

# Create a MDASequence
# NOTE: axis_order determines the order in which frames will be appended to the stream.
seq = useq.MDASequence(
    axis_order="ptcz",
    stage_positions=[(0.0, 0.0), (10.0, 10.0)],
    time_plan={"interval": 0.1, "loops": 3},
    channels=["DAPI", "FITC"],
    z_plan={"range": 2, "step": 1.0},
)
# -------------------------------------------------------------------------#

# Initialize pymmcore-plus core and load system configuration
core = CMMCorePlus.instance()
core.loadSystemConfiguration()

# Convert the MDASequence to ome_writers dimensions
dims = omew.dims_from_useq(
    seq, image_width=core.getImageWidth(), image_height=core.getImageHeight()
)

# Create an stream using the selected backend
ext = "tiff" if backend == "tiff" else "zarr"
path = output / f"{ext}_example.ome.{ext}"
stream = omew.create_stream(
    path=str(path),
    dimensions=dims,
    dtype=np.uint16,
    backend=backend,
    overwrite=True,
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


# Start the acquisition
core.mda.run(seq)
