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
# backend = "tensorstore"
# backend = "tiff"

# Only used if backend is "tiff". Leave True by default
tiff_memmap = True

# Create a MDASequence. NOTE: the axis_order determines the order in which frames will
# be appended to the stream.
seq = useq.MDASequence(
    axis_order="ptcz",
    stage_positions=[(0.0, 0.0), (10.0, 10.0)],
    time_plan={"interval": 0.5, "loops": 10},
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
path = output_path / f"{ext}_example.ome.{ext}"
if backend == "tiff":
    stream = omew.TifffileStream(use_memmap=tiff_memmap)
    stream.create(
        path=str(path),
        dimensions=dims,
        dtype=np.uint16,
        overwrite=True,
    )
else:
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

    if backend not in ("acquire-zarr", "tensorstore"):
        return

    # open zarr group and print structure and validate OME-NGFF JSON with yaozarrs
    try:
        import zarr
    except ImportError:
        warnings.warn(
            "Skipping zarr structure printout and OME-NGFF validation because zarr is "
            "not installed. Please install it via 'pip install zarr>=3' to enable these"
            " features.", stacklevel=2
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
    zarr_json_path = path / "zarr.json"
    assert zarr_json_path.exists(), "zarr.json should exist at root"
    with open(zarr_json_path) as f:
        root_meta = json.load(f)
        validate_ome_json(json.dumps(root_meta))


# Start the acquisition
core.mda.run(seq)
