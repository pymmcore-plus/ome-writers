# /// script
# requires-python = ">=3.11,<3.14"
# dependencies = [
#     "ome-writers[all]",
#     "pymmcore-plus>=0.16.0",
# ]
#
# [tool.uv.sources]
# ome-writers = { path = "../" }
# ///
"""Example of using ome_writers with useq.MDASequence and pymmcore-plus."""

import sys

from ome_writers import AcquisitionSettings, create_stream, dims_from_useq

try:
    import useq
    from pymmcore_plus import CMMCorePlus
except ImportError as e:
    raise ImportError(
        "This example requires pymmcore-plus. Please install it via "
        "pip install pymmcore-plus"
    ) from e


# use command line argument to select backend:
# `uv run examples/pymmcore_plus_example.py tiff` for OME-TIFF
# `uv run examples/pymmcore_plus_example.py tensorstore`  (or `zarr`, `acquire-zarr`)
BACKEND = "auto" if len(sys.argv) < 2 else sys.argv[1]
SUFFIX = ".ome.tiff" if BACKEND == "tiff" else ".ome.zarr"

# --------------------------CONFIGURATION SECTION--------------------------#


# Initialize pymmcore-plus core and load system configuration
core = CMMCorePlus()
core.loadSystemConfiguration()

# Create a MDASequence, which will be used to run the MDA with pymmcore-plus
seq = useq.MDASequence(
    axis_order="ptcz",
    # stage_positions=[(0.0, 0.0), (10.0, 10.0)],
    time_plan={"interval": 0.1, "loops": 3},
    channels=["DAPI", "Cy5"],
    z_plan={"range": 2, "step": 1.0},
)

# Setup the AcquisitionSettings, converting the MDASequence to ome-writers Dimensions
settings = AcquisitionSettings(
    root_path=f"example_pymmcore_plus{SUFFIX}",
    dimensions=dims_from_useq(
        seq,
        image_width=core.getImageWidth(),
        image_height=core.getImageHeight(),
    ),
    dtype=f"uint{core.getImageBitDepth()}",
    overwrite=True,
    backend=BACKEND,
)

# Open the stream and run the sequence
with create_stream(settings) as stream:
    # Append frames to the stream on frameReady event
    core.mda.events.frameReady.connect(lambda frame, *_: stream.append(frame))
    core.mda.run(seq)
