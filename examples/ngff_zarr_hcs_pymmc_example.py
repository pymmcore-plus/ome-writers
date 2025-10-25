"""HCS example using ome-writers and acquire-zarr with pymmcore_plus and useq."""

# import json
from pathlib import Path

import numpy as np
import useq
import zarr
from pymmcore_plus import CMMCorePlus
from pymmcore_plus.metadata import FrameMetaV1
from useq import GridRowsColumns, MDASequence, WellPlatePlan

# from yaozarrs import validate_ome_json
import ome_writers as omew

output_path = Path(__file__).parent / "acq_z.zarr"

# Create a simple plate plan with 3 wells, 3 fov per well
plate_plan = WellPlatePlan(
    plate="96-well",
    a1_center_xy=(0.0, 0.0),
    selected_wells=([0, 0, 1], [0, 1, 0]),  # A1, A2, B1
    well_points_plan=GridRowsColumns(rows=1, columns=2),  # 3 FOV per well
)

# Create MDA sequence with the plate plan
seq = MDASequence(
    axis_order="ptc",
    stage_positions=plate_plan,
    time_plan={"interval": 0.1, "loops": 3},
    channels=["FITC"],
)

# Convert useq MDASequence to ome-writers Plate and Dimensions
plate = omew.plate_from_useq(seq)
dims = omew.dims_from_useq(
    seq, image_width=512, image_height=512, chunk_sizes={"y": 256, "x": 256}
)

# Create acquire-zarr stream
stream = omew.create_stream(
    path=output_path,
    dimensions=dims,
    dtype=np.uint16,
    backend="acquire-zarr",
    plate=plate,
    overwrite=True,
)

# create CMMCorePlus instance and load system configuration
core = CMMCorePlus()
core.loadSystemConfiguration()


# connect event handlers
@core.mda.events.frameReady.connect
def _on_frame_ready(
    frame: np.ndarray, event: useq.MDAEvent, frame_meta: FrameMetaV1
) -> None:
    stream.append(frame)


@core.mda.events.sequenceFinished.connect
def _on_sequence_finished(sequence: useq.MDASequence) -> None:
    stream.flush()

    # open zarr group and print structure
    gp = zarr.open_group(output_path, mode="r")
    print(gp.tree())

    # validate OME-NGFF JSON at root
    # TODO: figure out what is the validation error
    # zarr_json_path = output_path / "zarr.json"
    # assert zarr_json_path.exists(), "zarr.json should exist at root"
    # with open(zarr_json_path) as f:
    #     root_meta = json.load(f)
    #     validate_ome_json(json.dumps(root_meta))


# run the MDA sequence
core.mda.run(seq)
