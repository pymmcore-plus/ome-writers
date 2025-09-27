from __future__ import annotations

import os
from typing import TYPE_CHECKING

import numpy as np
import pytest

import ome_writers as omew

try:
    import useq
    from pymmcore_plus import CMMCorePlus
    from pymmcore_plus.metadata._ome import create_ome_metadata
except ImportError:
    pytest.skip("pymmcore_plus is not installed", allow_module_level=True)


if TYPE_CHECKING:
    from pathlib import Path

    from pymmcore_plus.metadata import FrameMetaV1, SummaryMetaV1

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
    stream = omew.create_stream(
        dest,
        dimensions=omew.dims_from_useq(
            seq, core.getImageWidth(), core.getImageHeight()
        ),
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


def test_pymmcore_plus_mda_tiff_metadata_update(tmp_path: Path) -> None:
    """Test pymmcore_plus MDA with metadata update after acquisition."""

    seq = useq.MDASequence(
        time_plan=useq.TIntervalLoops(interval=0.001, loops=2),  # type: ignore
        z_plan=useq.ZRangeAround(range=2, step=1),
        channels=["DAPI", "FITC"],  # type: ignore
        stage_positions=[(0, 0), (0.1, 0.1)],  # type: ignore
    )

    core = CMMCorePlus()
    core.loadSystemConfiguration()

    dest = tmp_path / "test_mda_tiff_metadata_update.ome.tiff"
    stream = omew.create_stream(
        dest,
        dimensions=omew.dims_from_useq(
            seq, core.getImageWidth(), core.getImageHeight()
        ),
        dtype=np.uint16,
        overwrite=True,
        backend="tiff",
    )

    summary_meta: SummaryMetaV1 | None = None
    frame_meta_list: list[FrameMetaV1] = []

    @core.mda.events.sequenceStarted.connect
    def _on_sequence_started(
        sequence: useq.MDASequence, summary_metadata: SummaryMetaV1
    ) -> None:
        nonlocal summary_meta
        summary_meta = summary_metadata

    @core.mda.events.frameReady.connect
    def _on_frame_ready(
        frame: np.ndarray, event: useq.MDAEvent, frame_meta: FrameMetaV1
    ) -> None:
        stream.append(frame)
        frame_meta_list.append(frame_meta)

    @core.mda.events.sequenceFinished.connect
    def _on_sequence_finished(sequence: useq.MDASequence) -> None:
        stream.flush()
        if summary_meta is not None:
            ome_metadata = create_ome_metadata(summary_meta, frame_meta_list)
            stream.update_metadata(dict(ome_metadata))

    core.mda.run(seq)

    assert len(frame_meta_list) > 0
    assert summary_meta is not None
