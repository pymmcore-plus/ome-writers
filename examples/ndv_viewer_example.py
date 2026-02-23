# /// script
# requires-python = ">=3.11,<3.14"
# dependencies = [
#     "ome-writers[all]",
#     "ndv[vispy,pyqt]",
#     "numpy",
# ]
#
# [tool.uv.sources]
# ome-writers = { path = "../" }
# ///
"""Example of viewing ome_writers stream in ndv during acquisition.

This example demonstrates using the coordinate tracking API to update
an ndv viewer in real-time as frames are generated and written.
"""

from __future__ import annotations

import sys
import time
from threading import Thread
from typing import TYPE_CHECKING, Any

import ndv
import numpy as np

from ome_writers import AcquisitionSettings, Dimension, create_stream
from ome_writers._array_view import AcquisitionView

if TYPE_CHECKING:
    from collections.abc import Hashable, Mapping, Sequence

    from ome_writers import OMEStream
    from ome_writers._coord_tracker import CoordUpdate

# Setup acquisition settings
# Derive format/backend from command line argument (default: zarr)
FORMAT = "zarr" if len(sys.argv) < 2 else sys.argv[1]

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

# ===============================================================================
# Function to generate and write frames in a background thread
# ===============================================================================


def generate_frames(stream: OMEStream) -> None:
    """Generate random noise frames."""
    rng = np.random.default_rng(42)
    frame_shape = [dim.count or 1 for dim in settings.dimensions[-2:]]

    print(f"Starting acquisition: {settings.num_frames} frames")
    for i in range(settings.num_frames):
        # Generate random noise
        frame = rng.integers(0, 2000, size=frame_shape, dtype=np.uint16)
        stream.append(frame)

        # Small delay to simulate acquisition
        time.sleep(0.01)

        if (i + 1) % 10 == 0:
            print(f"  Written {i + 1}/{settings.num_frames} frames")
    else:
        print("Acquisition complete!")


# ===============================================================================
# ndv-specific code: DataWrapper that dynamically manages coordinate ranges
# ===============================================================================


class CoordsAwareDataWrapper(ndv.DataWrapper):
    """DataWrapper that tracks acquisition progress via coordinate events.

    NDV updates the viewers sliders based on the coordinate ranges returned by `coords`,
    and you can trigger slider range updates by emitting `dims_changed`.
    """

    def __init__(self, view: AcquisitionView) -> None:
        super().__init__(view)
        self._view = view
        self._current_coords: Mapping[Hashable, Sequence] = {
            i: range(s) for i, s in zip(self.dims, self._data.shape, strict=False)
        }

    def update_coords(self, update: CoordUpdate) -> None:
        """Called when new dimensions become visible (high water marks)."""
        # Store the latest coordinate ranges
        self._current_coords = update.max_coords
        # Emit dims_changed to tell ndv to update its slider ranges
        self.dims_changed.emit()
        print("Updating ndv slider ranges to:\n", self._current_coords)

    @property
    def dims(self) -> tuple[Hashable, ...]:
        """Return dimension names."""
        # Get dimension names from the view
        return self._view.dims

    @property
    def coords(self) -> Mapping[Hashable, Sequence]:
        """Return current visible coordinate ranges."""
        return self._current_coords

    @classmethod
    def supports(cls, obj: Any) -> bool:
        """Check if this wrapper supports the given object."""
        return isinstance(obj, AcquisitionView)


# ===============================================================================
# Main script: create stream, viewer, and start acquisition
# ===============================================================================

stream = create_stream(settings)

# Create an acquisition view and pass it to ndv with our coords-tracking wrapper
view = AcquisitionView.from_stream(stream)
wrapper = CoordsAwareDataWrapper(view)
viewer = ndv.ArrayViewer(wrapper)
viewer.show()

# pass coordinate updates from the stream to the wrapper...
stream.on("coords_expanded", wrapper.update_coords)

# Start frame generation in a background thread
acquisition_thread = Thread(target=generate_frames, args=(stream,), daemon=True)
acquisition_thread.start()

# Start the ndv event loop (blocks until viewer is closed)
ndv.run_app()

# Cleanup
acquisition_thread.join(timeout=1.0)
stream.close()
print("Done!")
