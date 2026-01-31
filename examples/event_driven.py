"""Example of using ome_writers for event-driven acquisition.

This is our current recommendation where the higher-order structure of the dataset is
not known in advance.  In this pattern, we assume only that a series of 2D frames
will be acquired, and that each frame has some associated metadata.  This is the most
general case (though may lack convenience features of more structured sub-cases).
"""

import datetime
import sys
from collections.abc import Iterator
from itertools import count

import numpy as np

from ome_writers import AcquisitionSettings, Dimension, create_stream

# Derive backend from command line argument (default: auto)
BACKEND = "auto" if len(sys.argv) < 2 else sys.argv[1]
suffix = ".ome.tiff" if BACKEND == "tifffile" else ".ome.zarr"

# create acquisition settings
settings = AcquisitionSettings(
    root_path=f"example_event_driven{suffix}",
    dimensions=[
        Dimension(name="t", count=None, type="time"),
        Dimension(name="y", count=256, chunk_size=64, type="space"),
        Dimension(name="x", count=256, chunk_size=64, type="space"),
    ],
    dtype="uint16",
    overwrite=True,
    format=BACKEND,
)


frame_shape = tuple(d.count or 1 for d in settings.dimensions[-2:])
np.random.seed(0)


# Placeholder for event-driven frame generator
# In a real application, this would be replaced with code that interfaces
# with the microscope hardware to acquire frames and metadata.
# It's anything that pumps out frames, e.g., callbacks, async loops, threads, etc.
# and has some metadata associated with each frame.
def some_event_driven_frame_generator() -> Iterator[tuple[np.ndarray, dict]]:
    dtype = np.dtype(settings.dtype)
    iinfo = np.iinfo(dtype)
    low, high = iinfo.min, iinfo.max + 1
    counter = count()
    start_time = datetime.datetime.now()
    while True:
        frame_idx = next(counter)
        current_time = datetime.datetime.now()

        # Any key-value metadata can be provided per-frame.
        # Some keys have special format-specific meaning, and can be placed in the
        # proper OME-Tiff metadata, while Zarr just stores everything as-is.
        # - delta_t: time from start (seconds)
        # - exposure_time: exposure duration (seconds)
        # - position_x, position_y, position_z: stage positions (micrometers)
        metadata = {
            # Special keys:
            "delta_t": (current_time - start_time).total_seconds(),
            "exposure_time": 0.01,  # 10 ms exposure
            "position_x": 100.0 + frame_idx * 0.5,  # simulated stage drift
            "position_y": 200.0 + frame_idx * 0.3,
            "position_z": 50.0,
            # Custom keys are also preserved:
            "temperature": 37.0 + np.random.randn() * 0.1,
            "laser_power": 50.0,
        }
        print(
            f"Frame {frame_idx}: t={metadata['delta_t']:.3f}s, "
            f"pos=({metadata['position_x']:.1f}, {metadata['position_y']:.1f}, "
            f"{metadata['position_z']:.1f})"
        )
        yield (
            np.random.randint(low, high, size=frame_shape, dtype=dtype),
            metadata,
        )
        if np.random.rand() < 0.1:  # stop with 10% probability
            break


with create_stream(settings) as stream:
    for frame, meta in some_event_driven_frame_generator():
        stream.append(frame, frame_metadata=meta)


if settings.format == "tiff":
    from ome_types import from_tiff

    ome_obj = from_tiff(settings.root_path)
    print(ome_obj.to_xml())
