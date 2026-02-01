"""Example: Frame skipping during acquisition.

Demonstrates using OMEStream.skip() to handle failures during
multi-position timelapse acquisition.  Here, we simulate autofocus
failures at specific positions, and then skip all frames that would have been
acquired at that position/timepoint.
"""

import sys
from typing import cast

import numpy as np

from ome_writers import create_stream
from ome_writers._schema import AcquisitionSettings, dims_from_standard_axes

# Derive format/backend from command line argument (default: auto)
FORMAT = "auto" if len(sys.argv) < 2 else sys.argv[1]


settings = AcquisitionSettings(
    root_path="example_frame_skipping",
    dimensions=dims_from_standard_axes(
        {"t": 5, "p": 3, "c": 2, "z": 3, "y": 256, "x": 256}
    ),
    dtype="uint16",
    format=FORMAT,
    overwrite=True,
)

nt, npos, nc, nz, *_ = cast("tuple[int, ...]", settings.shape)


class AutofocusError(Exception):
    """Simulated autofocus failure."""


def attempt_autofocus(pos: int, t: int) -> None:
    """Simulate frame acquisition with occasional autofocus failures."""
    if pos == 1 and t in (1, 2, 4):
        raise AutofocusError(f"Autofocus failed at position {pos} at timepoint {t}")


with create_stream(settings) as stream:
    for t in range(nt):
        for p in range(npos):
            try:
                attempt_autofocus(p, t)
            except AutofocusError as e:
                n = nc * nz
                print(f"Warning: {e}, skipping {n} frames")
                stream.skip(frames=n)
            else:
                for _ in range(nc * nz):
                    frame = np.random.randint(0, 65536, size=(256, 256), dtype="uint16")
                    stream.append(frame)
