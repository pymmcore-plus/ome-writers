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
"""Example of viewing ome_writers stream in ndv during acquisition.

This example demonstrates using `live_shape=True` to get a StreamView
whose shape/coords dynamically reflect acquisition progress, and wiring
it to an ndv viewer for real-time display.
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

settings = AcquisitionSettings(
    root_path="example_ndv_viewer",
    dimensions=[
        Dimension(name="t", count=3, type="time"),
        Dimension(name="p", type="position", coords=["pos0", "pos1"]),
        Dimension(name="c", type="channel", coords=["DAPI", "GFP", "TRITC"]),
        Dimension(name="z", count=10, type="space"),
        Dimension(name="y", count=512, type="space"),
        Dimension(name="x", count=512, type="space"),
    ],
    dtype="uint16",
    overwrite=True,
    format=FORMAT,
)


def generate_frames(stream: OMEStream) -> None:
    """Generate gaussian blob frames that drift over time."""
    ny, nx = (dim.count or 1 for dim in settings.dimensions[-2:])
    nz = next((d.count for d in settings.dimensions if d.name == "z"), 1) or 1
    yy, xx = np.mgrid[:ny, :nx].astype(np.float32)

    for i in range(settings.num_frames):
        # Blob center drifts per (t,p,c) group; z only changes sigma
        g = i // nz  # group index (constant across z)
        cy, cx = ny / 2 + 80 * np.sin(g * 0.3), nx / 2 + 80 * np.cos(g * 0.2)
        sigma2 = 1000 + 500 * (i % nz)  # wider blob at higher z
        blob = 2000 * np.exp(-((yy - cy) ** 2 + (xx - cx) ** 2) / sigma2)
        noise = np.random.randint(0, 200, size=(ny, nx), dtype=np.uint16)
        stream.append(blob.astype(np.uint16) + noise)
        time.sleep(0.01)

        if (i + 1) % 10 == 0:
            print(f"  Written {i + 1}/{settings.num_frames} frames")


# ===============================================================================

with create_stream(settings) as stream:
    # Create a StreamView.  This is an array-like object with dims/coords
    # that reflect the stream's current state (by default, live_shape=True).
    view = stream.view()

    # ndv has built-in support for array-like objects that support the xarray
    # dims/coords convention.  So we can directly pass our StreamView to an ndv viewer
    viewer = ndv.ArrayViewer(view)
    viewer.show()

    # the only thing we need to connect is the StreamView's coords_changed signal,
    # so that the viewer updates when new frames are added and the shape/coords change.
    view.coords_changed.connect(viewer.data_wrapper.dims_changed)

    # Start frame generation in a background thread
    acquisition_thread = Thread(target=generate_frames, args=(stream,), daemon=True)
    acquisition_thread.start()

    # Start the ndv event loop (blocks until viewer is closed)
    ndv.run_app()


print("Done!")
