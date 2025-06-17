import os
from pathlib import Path

import numpy as np
import pytest
import useq

from ome_writers import Dimension, create_stream

try:
    from pymmcore_plus import CMMCorePlus
    from pymmcore_plus.metadata import FrameMetaV1
except ImportError:
    pytest.skip("pymmcore_plus is not installed", allow_module_level=True)


def _mda_seq_to_dims(seq: useq.MDASequence) -> list[Dimension]:
    dims: list[Dimension] = []
    for ax, size in seq.sizes.items():
        if not size:
            continue
        if ax == "t":
            dims.append(Dimension(label="t", size=size, unit=(1.0, "s")))
        elif ax == "c":
            dims.append(Dimension(label="c", size=size))
        elif ax in ("x", "y", "z", "p"):
            dims.append(Dimension(label=ax, size=size, unit=(1.0, "um")))
        else:
            raise ValueError(f"Unknown axis: {ax}")
    return dims


@pytest.mark.parametrize("backend", ["tensorstore", "acquire-zarr", "tiff"])
def test_pymmcore_plus_mda(tmp_path: Path, backend: str) -> None:
    seq = useq.MDASequence(
        time_plan=useq.TIntervalLoops(interval=0.001, loops=3),  # type: ignore
        z_plan=useq.ZRangeAround(range=2, step=1),
        channels=["DAPI", "FITC"],  # type: ignore
        stage_positions=[(0, 0), (0.1, 0.1)],  # type: ignore,
    )

    core = CMMCorePlus()
    core.loadSystemConfiguration()

    dims = [
        *_mda_seq_to_dims(seq),
        Dimension(label="y", size=core.getImageHeight()),
        Dimension(label="x", size=core.getImageWidth()),
    ]

    ext = ".ome.tiff" if backend == "tiff" else ".zarr"
    dest = tmp_path / f"test_pymmcore_plus_mda{ext}"
    stream = create_stream(
        dest, dimensions=dims, dtype=np.uint16, overwrite=True, backend=backend
    )

    @core.mda.events.frameReady.connect
    def _on_frame_ready(
        frame: np.ndarray, event: useq.MDAEvent, metadata: FrameMetaV1
    ) -> None:
        stream.append(frame)

    core.mda.run(seq)
    stream.flush()

    # make assertions
    if backend == "tiff":
        assert os.path.exists(str(dest).replace(".ome.tiff", "_p000.ome.tiff"))
    else:
        assert dest.exists()
