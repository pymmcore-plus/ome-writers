from __future__ import annotations

import os
from typing import TYPE_CHECKING

import numpy as np
import pytest

from ome_writers import create_stream, dims_from_useq

try:
    import useq
    from pymmcore_plus import CMMCorePlus
except ImportError:
    pytest.skip("pymmcore_plus is not installed", allow_module_level=True)


if TYPE_CHECKING:
    from pathlib import Path

    from pymmcore_plus.metadata import FrameMetaV1

    from .conftest import AvailableBackend


def test_pymmcore_plus_mda(tmp_path: Path, backend: AvailableBackend) -> None:
    seq = useq.MDASequence(
        time_plan=useq.TIntervalLoops(interval=0.001, loops=3),  # type: ignore
        z_plan=useq.ZRangeAround(range=2, step=1),
        channels=["DAPI", "FITC"],  # type: ignore
        stage_positions=[(0, 0), (0.1, 0.1)],  # type: ignore
    )

    core = CMMCorePlus()
    core.loadSystemConfiguration()

    dest = tmp_path / f"test_pymmcore_plus_mda{backend.file_ext}"
    stream = create_stream(
        dest,
        dimensions=dims_from_useq(seq, core.getImageWidth(), core.getImageHeight()),
        dtype=np.uint16,
        overwrite=True,
        backend=backend.name,
    )

    @core.mda.events.frameReady.connect
    def _on_frame_ready(
        frame: np.ndarray, event: useq.MDAEvent, metadata: FrameMetaV1
    ) -> None:
        stream.append(frame)

    core.mda.run(seq)
    stream.flush()

    # make assertions
    if backend.file_ext.endswith(".tiff"):
        assert os.path.exists(str(dest).replace(".ome.tiff", "_p000.ome.tiff"))
    else:
        assert dest.exists()
