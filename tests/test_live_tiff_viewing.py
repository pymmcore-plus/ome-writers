"""Tests for live TIFF viewing during acquisition."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest

from ome_writers import AcquisitionSettings, Dimension, create_stream

try:
    from ome_writers._backends import _tifffile  # noqa: F401
    from ome_writers._backends._tiff_array import FinalizedTiffArray, LiveTiffArray
except ImportError:
    pytest.skip(
        "LiveTiffArray tests require tifffile dependency",
        allow_module_level=True,
    )

from tests._utils import wait_for_frames, wait_for_pending_callbacks

if TYPE_CHECKING:
    from pathlib import Path


def test_live_tiff_viewing_basic(tmp_path: Path) -> None:
    """Test basic live viewing during TIFF acquisition."""
    settings = AcquisitionSettings(
        root_path=tmp_path / "test_live",
        dimensions=[
            Dimension(name="t", count=5, chunk_size=1, type="time"),
            Dimension(name="c", count=2, chunk_size=1, type="channel"),
            Dimension(name="y", count=32, chunk_size=32, type="space"),
            Dimension(name="x", count=32, chunk_size=32, type="space"),
        ],
        dtype="uint16",
        format="tifffile",
        overwrite=True,
    )

    with create_stream(settings) as stream:
        view = stream.view()
        # Underlying array should be a LiveTiffArray (no zarr dependency)
        array0 = view._arrays[0]
        assert isinstance(array0, LiveTiffArray)

        # Write all frames (5 time points * 2 channels = 10 frames)
        for i in range(10):
            frame = np.full((32, 32), i, dtype=np.uint16)
            stream.append(frame)

        # Wait for frames to be written by WriterThread
        wait_for_frames(stream._backend, expected_count=10)
        # Wait for async coord callbacks to update the view's shape
        wait_for_pending_callbacks(stream)

        # Should be able to read written frames
        assert view.shape == (5, 2, 32, 32)
        data = view[0, 0]
        assert data.shape == (32, 32)
        assert np.all(data == 0)  # First frame (t=0, c=0)

        data = view[2, 1]
        assert np.all(data == 5)  # Frame at t=2, c=1 (6th frame: t*2 + c = 2*2 + 1)


def test_live_viewing_returns_zeros_for_unwritten(tmp_path: Path) -> None:
    """Test that unwritten frames return zeros."""
    settings = AcquisitionSettings(
        root_path=tmp_path / "test_zeros",
        dimensions=[
            Dimension(name="t", count=10, chunk_size=1, type="time"),
            Dimension(name="y", count=16, chunk_size=16, type="space"),
            Dimension(name="x", count=16, chunk_size=16, type="space"),
        ],
        dtype="uint16",
        format="tifffile",
        overwrite=True,
    )

    with create_stream(settings) as stream:
        # Write only first 3 frames
        for i in range(3):
            frame = np.full((16, 16), i + 100, dtype=np.uint16)
            stream.append(frame)

        # Wait for frames to be written
        wait_for_frames(stream._backend, expected_count=3)

        # Get live view
        view = stream.view()

        # First 3 frames should have data
        assert np.all(view[0] == 100)
        assert np.all(view[1] == 101)
        assert np.all(view[2] == 102)

        # Remaining frames should be zeros (unwritten)
        assert np.all(view[3] == 0)
        assert np.all(view[9] == 0)


def test_finalized_tiff_uses_finalized_array(tmp_path: Path) -> None:
    """Test that finalized TIFF files use FinalizedTiffArray."""
    settings = AcquisitionSettings(
        root_path=tmp_path / "test_finalized",
        dimensions=[
            Dimension(name="t", count=3, chunk_size=1, type="time"),
            Dimension(name="y", count=16, chunk_size=16, type="space"),
            Dimension(name="x", count=16, chunk_size=16, type="space"),
        ],
        dtype="uint16",
        format="tifffile",
        overwrite=True,
    )

    # Write and close
    with create_stream(settings) as stream:
        for i in range(3):
            frame = np.full((16, 16), i, dtype=np.uint16)
            stream.append(frame)

    # After closing, should use FinalizedTiffArray (not zarr)
    view = stream.view(dynamic_shape=False)
    assert isinstance(view._arrays[0], FinalizedTiffArray)
    assert view.shape == (3, 16, 16)
    assert np.all(view[0] == 0)
    assert np.all(view[1] == 1)
    assert np.all(view[2] == 2)


def test_live_viewing_with_compression_raises_error(tmp_path: Path) -> None:
    """Test that live viewing with compression raises an error."""
    settings = AcquisitionSettings(
        root_path=tmp_path / "test_compression",
        dimensions=[
            Dimension(name="t", count=3, chunk_size=1, type="time"),
            Dimension(name="y", count=16, chunk_size=16, type="space"),
            Dimension(name="x", count=16, chunk_size=16, type="space"),
        ],
        dtype="uint16",
        format="tifffile",
        compression="lzw",  # Enable compression
        overwrite=True,
    )

    with create_stream(settings) as stream:
        for i in range(3):
            frame = np.full((16, 16), i, dtype=np.uint16)
            stream.append(frame)

        # Attempting live view with compression should raise error
        with pytest.raises(
            NotImplementedError,
            match="not supported with compression",
        ):
            stream.view()


def test_compute_index_strides() -> None:
    """Test stride computation for row-major ordering."""
    assert LiveTiffArray._compute_index_strides((10, 5, 2)) == (10, 2, 1)
    assert LiveTiffArray._compute_index_strides((3, 4)) == (4, 1)
    assert LiveTiffArray._compute_index_strides((100,)) == (1,)
    assert LiveTiffArray._compute_index_strides(()) == ()


def test_tiff_view_on_empty_closed_stream(tmp_path: Path) -> None:
    """View on a closed tiff stream with no frames written."""
    settings = AcquisitionSettings(
        root_path=tmp_path / "empty.ome.tiff",
        dimensions=[
            Dimension(name="t", count=3),
            Dimension(name="c", count=3),
            Dimension(name="z", count=3),
            Dimension(name="y", count=64),
            Dimension(name="x", count=64),
        ],
        dtype="uint16",
        overwrite=True,
        format="tifffile",
    )
    stream = create_stream(settings)
    stream.close()
    view = stream.view(dynamic_shape=False)
    assert view.shape == tuple(d.count for d in settings.dimensions)
    assert np.allclose(view[:], 0)


def test_tiff_view_on_single_written_closed_stream(tmp_path: Path) -> None:
    """Finalized partial uncompressed TIFF uses LiveTiffArray."""
    settings = AcquisitionSettings(
        root_path=tmp_path / "single_written.ome.tiff",
        dimensions=[
            Dimension(name="t", count=3),
            Dimension(name="y", count=64),
            Dimension(name="x", count=64),
        ],
        dtype="uint16",
        overwrite=True,
        format="tifffile",
    )
    stream = create_stream(settings)
    stream.append(np.ones((64, 64), dtype=np.uint16))
    stream.close()
    view = stream.view(dynamic_shape=False)
    assert view.shape == (3, 64, 64)
    assert np.all(view[0] == 1)  # Written frame
    assert np.all(view[1] == 0)  # Unwritten
    assert np.all(view[2] == 0)  # Unwritten


def test_tiff_view_on_finalized_compressed_stream(tmp_path: Path) -> None:
    """Finalized fully-written compressed TIFF is viewable via FinalizedTiffArray."""
    settings = AcquisitionSettings(
        root_path=tmp_path / "compressed_finalized.ome.tiff",
        dimensions=[
            Dimension(name="t", count=3),
            Dimension(name="y", count=64),
            Dimension(name="x", count=64),
        ],
        dtype="uint16",
        overwrite=True,
        format="tifffile",
        compression="lzw",
    )
    with create_stream(settings) as stream:
        stream.append(np.ones((64, 64), dtype=np.uint16))
        stream.append(np.ones((64, 64), dtype=np.uint16))
        stream.append(np.ones((64, 64), dtype=np.uint16))

    view = stream.view()
    assert view.shape == (3, 64, 64)
    assert np.all(view[:] == 1)


def test_tiff_view_on_partial_finalized_compressed_stream_raises(
    tmp_path: Path,
) -> None:
    """Finalized partial compressed TIFF view is not supported."""
    settings = AcquisitionSettings(
        root_path=tmp_path / "partial_compressed.ome.tiff",
        dimensions=[
            Dimension(name="t", count=3),
            Dimension(name="y", count=64),
            Dimension(name="x", count=64),
        ],
        dtype="uint16",
        overwrite=True,
        format="tifffile",
        compression="lzw",
    )
    stream = create_stream(settings)
    stream.append(np.ones((64, 64), dtype=np.uint16))
    stream.close()
    with pytest.raises(NotImplementedError, match="not supported with compression"):
        stream.view()


def test_tiff_view_on_empty_finalized_compressed_stream(tmp_path: Path) -> None:
    """Zero-frame finalized compressed TIFF raises (no raw byte access)."""
    settings = AcquisitionSettings(
        root_path=tmp_path / "empty_compressed.ome.tiff",
        dimensions=[
            Dimension(name="t", count=3),
            Dimension(name="y", count=64),
            Dimension(name="x", count=64),
        ],
        dtype="uint16",
        overwrite=True,
        format="tifffile",
        compression="lzw",
    )
    stream = create_stream(settings)
    stream.close()
    with pytest.raises(NotImplementedError, match="not supported with compression"):
        stream.view(dynamic_shape=False)


def test_unbounded_live_tiff_shape(tmp_path: Path) -> None:
    """Test that unbounded acquisitions report correct dynamic shape."""
    settings = AcquisitionSettings(
        root_path=tmp_path / "unbounded.ome.tiff",
        dimensions=[
            Dimension(name="t", count=None, chunk_size=1, type="time"),
            Dimension(name="c", count=2, chunk_size=1, type="channel"),
            Dimension(name="y", count=16, chunk_size=16, type="space"),
            Dimension(name="x", count=16, chunk_size=16, type="space"),
        ],
        dtype="uint16",
        format="tifffile",
        overwrite=True,
    )

    with create_stream(settings) as stream:
        # Before writing, shape should have 0 for unbounded dim
        arrays = stream._backend.get_arrays()
        assert isinstance(arrays[0], LiveTiffArray)
        assert arrays[0].shape == (0, 2, 16, 16)

        # Write 4 frames (2 complete time points)
        for i in range(4):
            frame = np.full((16, 16), i, dtype=np.uint16)
            stream.append(frame)

        wait_for_frames(stream._backend, expected_count=4)

        # Shape should reflect 2 time points
        assert arrays[0].shape == (2, 2, 16, 16)

        # Read the data
        assert np.all(arrays[0][0, 0] == 0)
        assert np.all(arrays[0][0, 1] == 1)
        assert np.all(arrays[0][1, 0] == 2)
        assert np.all(arrays[0][1, 1] == 3)

        # Write 2 more frames (1 more time point)
        for i in range(2):
            frame = np.full((16, 16), i + 10, dtype=np.uint16)
            stream.append(frame)

        wait_for_frames(stream._backend, expected_count=6)
        assert arrays[0].shape == (3, 2, 16, 16)
