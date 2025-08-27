import os
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pytest
import useq

from ome_writers import BackendName, create_stream
from ome_writers._util import dims_from_useq

try:
    from pymmcore_plus import CMMCorePlus
    from pymmcore_plus.metadata import FrameMetaV1
except ImportError:
    pytest.skip("pymmcore_plus is not installed", allow_module_level=True)

BACKENDS: Sequence[BackendName] = ["tensorstore", "acquire-zarr", "tiff"]


@pytest.mark.parametrize("backend", BACKENDS)
def test_pymmcore_plus_mda(tmp_path: Path, backend: BackendName) -> None:
    seq = useq.MDASequence(
        time_plan=useq.TIntervalLoops(interval=0.001, loops=3),  # type: ignore
        z_plan=useq.ZRangeAround(range=2, step=1),
        channels=["DAPI", "FITC"],  # type: ignore
        stage_positions=[(0, 0), (0.1, 0.1)],  # type: ignore
    )

    core = CMMCorePlus()
    core.loadSystemConfiguration()

    ext = ".ome.tiff" if backend == "tiff" else ".zarr"
    dest = tmp_path / f"test_pymmcore_plus_mda{ext}"
    stream = create_stream(
        dest,
        dimensions=dims_from_useq(seq, core.getImageWidth(), core.getImageHeight()),
        dtype=np.uint16,
        overwrite=True,
        backend=backend,
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
