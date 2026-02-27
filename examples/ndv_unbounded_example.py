# /// script
# requires-python = ">=3.11,<3.14"
# dependencies = [
#     "ome-writers[all]",
#     "ndv[vispy,pyqt]",
#     "numpy",
# ]
#
# [tool.uv.sources]
# ome-writers = { path = "../", editable = true }
# ///
"""Example of viewing an unbounded ome_writers stream in ndv.

Like ndv_viewer_example.py, but the time dimension has no predetermined count
(count=None). The actual number of timepoints (100) is only known at runtime.
"""

from __future__ import annotations

import sys
import time
from threading import Thread
from typing import TYPE_CHECKING

import ndv
import numpy as np

from ome_writers import AcquisitionSettings, Dimension, create_stream

if TYPE_CHECKING:
    from ome_writers import OMEStream

# Setup acquisition settings
# Derive format/backend from command line argument (default: auto)
FORMAT = "auto" if len(sys.argv) < 2 else sys.argv[1]

NUM_TIMEPOINTS = 100

settings = AcquisitionSettings(
    root_path="example_ndv_unbounded",
    dimensions=[
        Dimension(name="t", count=None, type="time"),  # unbounded!
        Dimension(name="c", count=4, type="channel"),
        Dimension(name="y", count=512, type="space"),
        Dimension(name="x", count=512, type="space"),
    ],
    dtype="uint16",
    overwrite=True,
    format=FORMAT,
)


def generate_frames(stream: OMEStream) -> None:
    """Generate gaussian blob frames that drift over time."""
    ny, nx = 512, 512
    nc = 4
    yy, xx = np.mgrid[:ny, :nx].astype(np.float32)
    total = NUM_TIMEPOINTS * nc

    for i in range(total):
        t = i // nc
        cy = ny / 2 + 80 * np.sin(t * 0.1)
        cx = nx / 2 + 80 * np.cos(t * 0.08)
        sigma2 = 1000 + 300 * (i % nc)
        blob = 2000 * np.exp(-((yy - cy) ** 2 + (xx - cx) ** 2) / sigma2)
        noise = np.random.randint(0, 200, size=(ny, nx), dtype=np.uint16)
        stream.append(blob.astype(np.uint16) + noise)
        time.sleep(0.01)

        if (i + 1) % 10 == 0:
            print(f"  Written {i + 1}/{total} frames")


# ===============================================================================

with create_stream(settings) as stream:
    view = stream.view()

    viewer = ndv.ArrayViewer(view)
    viewer.show()

    view.coords_changed.connect(viewer.data_wrapper.dims_changed)

    acquisition_thread = Thread(target=generate_frames, args=(stream,), daemon=True)
    acquisition_thread.start()

    ndv.run_app()


print("Done!")
