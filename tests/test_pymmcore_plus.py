from __future__ import annotations

import os
from contextlib import suppress
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


class PYMMCP:
    """A little example of how one might integrate pymmcore_plus MDA with ome-writers.

    This class listens to pymmcore_plus MDA events and writes data to an OME-TIFF
    file using ome-writers. It also collects metadata during the acquisition and
    updates the OME metadata at the end of the sequence.
    """

    def __init__(
        self,
        sequence: useq.MDASequence,
        core: CMMCorePlus,
        dest: Path,
        backend: str,  # "acquire-zarr", "tensorstore" or "tiff"
    ) -> None:
        self._seq = sequence
        self._core = core
        self._dest = dest

        self._summary_meta: SummaryMetaV1 = {}  # type: ignore
        self._frame_meta_list: list[FrameMetaV1] = []

        self._stream = omew.create_stream(
            self._dest,
            dimensions=omew.dims_from_useq(
                self._seq, core.getImageWidth(), core.getImageHeight()
            ),
            dtype=np.uint16,
            overwrite=True,
            backend=backend,
        )

        @core.mda.events.sequenceStarted.connect
        def _on_sequence_started(
            sequence: useq.MDASequence, summary_meta: SummaryMetaV1
        ) -> None:
            self._summary_meta = summary_meta

        @core.mda.events.frameReady.connect
        def _on_frame_ready(
            frame: np.ndarray, event: useq.MDAEvent, frame_meta: FrameMetaV1
        ) -> None:
            self._stream.append(frame)
            self._frame_meta_list.append(frame_meta)

        @core.mda.events.sequenceFinished.connect
        def _on_sequence_finished(sequence: useq.MDASequence) -> None:
            self._stream.flush()
            if hasattr(self._stream, "update_ome_metadata"):
                ome = create_ome_metadata(self._summary_meta, self._frame_meta_list)
                self._stream.update_ome_metadata(ome)

    def run(self) -> None:
        self._core.mda.run(self._seq)


def test_pymmcore_plus_mda_tiff_metadata_update(tmp_path: Path) -> None:
    """Test pymmcore_plus MDA with metadata update after acquisition."""

    # skip if tifffile or ome-types are not installed
    try:
        import tifffile
        from ome_types import from_xml
    except ImportError:
        pytest.skip("tifffile or ome-types are not installed")

    seq = useq.MDASequence(
        time_plan=useq.TIntervalLoops(interval=0.001, loops=2),  # type: ignore
        z_plan=useq.ZRangeAround(range=2, step=1),
        channels=["DAPI", "FITC"],  # type: ignore
        stage_positions=useq.WellPlatePlan(
            plate=useq.WellPlate.from_str("96-well"),
            a1_center_xy=(0, 0),
            selected_wells=((0, 0), (0, 1)),
        ),
    )

    core = CMMCorePlus()
    core.loadSystemConfiguration()

    dest = tmp_path / "test_mda_tiff_metadata_update.ome.tiff"

    pymm = PYMMCP(seq, core, dest, backend="tiff")
    pymm.run()

    # reopen the file and validate ome metadata
    for f in list(tmp_path.glob("*.ome.tiff")):
        with tifffile.TiffFile(f) as tif:
            ome_xml = tif.ome_metadata
            if ome_xml is not None:
                # validate by attempting to parse
                ome = from_xml(ome_xml)
                # assert there is plate information
                assert ome.plates


def test_pymmcore_plus_plate_zarr(tmp_path: Path) -> None:
    """Test pymmcore_plus MDA with WellPlatePlan and acquire-zarr backend."""
    output_path = tmp_path / "acq_z.zarr"

    plate_plan = useq.WellPlatePlan(
        plate="96-well",
        a1_center_xy=(0.0, 0.0),
        selected_wells=([0, 0, 1], [0, 1, 0]),  # A1, A2, B1
        well_points_plan=useq.GridRowsColumns(rows=2, columns=2),  # 4 FOV per well
    )

    seq = useq.MDASequence(
        axis_order="ptc",
        stage_positions=plate_plan,
        time_plan={"interval": 0.1, "loops": 3},
        channels=["DAPI", "FITC"],
    )

    plate = omew.plate_from_useq(seq)
    dims = omew.dims_from_useq(seq, image_width=512, image_height=512)

    stream = omew.create_stream(
        path=output_path,
        dimensions=dims,
        dtype=np.uint16,
        backend="acquire-zarr",
        plate=plate,
        overwrite=True,
    )

    def _on_frame_ready(
        frame: np.ndarray, event: useq.MDAEvent, frame_meta: FrameMetaV1
    ) -> None:
        stream.append(frame)

    def _on_sequence_finished(sequence: useq.MDASequence) -> None:
        stream.flush()

        # validate OME-NGFF JSON at root
        with suppress(ImportError):
            import json

            import zarr
            from yaozarrs import validate_ome_json

            zarr.open_group(output_path, mode="r")
            # TODO: figure out what is the validation error
            zarr_json_path = output_path / "zarr.json"
            assert zarr_json_path.exists(), "zarr.json should exist at root"
            with open(zarr_json_path) as f:
                root_meta = json.load(f)
                validate_ome_json(json.dumps(root_meta))

    core = CMMCorePlus()
    core.loadSystemConfiguration()
    core.mda.events.frameReady.connect(_on_frame_ready)
    core.mda.events.sequenceFinished.connect(_on_sequence_finished)

    core.mda.run(seq)
