"""Tests for the scratch array backend."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import numpy as np
import pytest

from ome_writers import AcquisitionSettings, Position, dims_from_standard_axes
from ome_writers._array_view import AcquisitionView
from ome_writers._stream import create_stream

if TYPE_CHECKING:
    from pathlib import Path


def _make_settings(**kwargs: object) -> AcquisitionSettings:
    defaults = {
        "format": "scratch",
        "dimensions": dims_from_standard_axes({"t": 3, "c": 2, "y": 8, "x": 8}),
        "dtype": "uint16",
    }
    defaults.update(kwargs)
    return AcquisitionSettings(**defaults)


def test_basic_write_and_read() -> None:
    settings = _make_settings()
    with create_stream(settings) as stream:
        arrays = stream._backend.get_arrays()
        assert len(arrays) == 1

        frame = np.ones((8, 8), dtype="uint16") * 42
        for _ in range(6):  # 3t * 2c
            stream.append(frame)

    arr = arrays[0]
    assert arr.shape == (3, 2, 8, 8)
    assert arr.dtype == np.uint16
    assert np.all(arr[0, 0] == 42)


def test_acquisition_view_compatibility() -> None:
    settings = _make_settings()
    with create_stream(settings) as stream:
        view = AcquisitionView.from_stream(stream)

        frame = np.ones((8, 8), dtype="uint16") * 7
        for _ in range(6):
            stream.append(frame)

        assert view.shape == (3, 2, 8, 8)
        assert view.dtype == np.uint16
        result = view[0, 0]
        assert result.shape == (8, 8)
        assert np.all(result == 7)


def test_multi_position() -> None:
    dims = dims_from_standard_axes(
        {
            "t": 2,
            "p": [Position(name="A"), Position(name="B")],
            "y": 8,
            "x": 8,
        }
    )
    settings = _make_settings(dimensions=dims)

    with create_stream(settings) as stream:
        arrays = stream._backend.get_arrays()
        assert len(arrays) == 2

        for i in range(4):  # 2t * 2p
            frame = np.full((8, 8), i, dtype="uint16")
            stream.append(frame)

    # Acquisition order: t0p0, t0p1, t1p0, t1p1
    assert np.all(arrays[0][0] == 0)  # t0, pos A
    assert np.all(arrays[1][0] == 1)  # t0, pos B
    assert np.all(arrays[0][1] == 2)  # t1, pos A
    assert np.all(arrays[1][1] == 3)  # t1, pos B


def test_unbounded_dimension_resize() -> None:
    dims = dims_from_standard_axes({"t": None, "c": 2, "y": 8, "x": 8})
    settings = _make_settings(dimensions=dims)

    with create_stream(settings) as stream:
        arrays = stream._backend.get_arrays()

        # Initially shape should have 1 for unbounded dim
        assert arrays[0].shape == (1, 2, 8, 8)

        # Write enough frames to trigger resize
        for i in range(20):  # 10t * 2c
            frame = np.full((8, 8), i, dtype="uint16")
            stream.append(frame)

        # Live view should see growth
        assert arrays[0].shape[0] == 10
        assert arrays[0].shape[1:] == (2, 8, 8)

        # Verify data integrity after resize
        assert np.all(arrays[0][0, 0] == 0)
        assert np.all(arrays[0][0, 1] == 1)


def test_skip_fills_zeros() -> None:
    settings = _make_settings()
    with create_stream(settings) as stream:
        arrays = stream._backend.get_arrays()

        frame = np.ones((8, 8), dtype="uint16") * 99
        stream.append(frame)  # t0c0
        stream.skip(frames=1)  # t0c1 skipped
        stream.append(frame)  # t1c0
        stream.skip(frames=3)  # t1c1, t2c0, t2c1 skipped

    assert np.all(arrays[0][0, 0] == 99)
    assert np.all(arrays[0][0, 1] == 0)
    assert np.all(arrays[0][1, 0] == 99)
    assert np.all(arrays[0][1, 1] == 0)
    assert np.all(arrays[0][2] == 0)


def test_tempdir_mode(tmp_path: Path) -> None:
    settings = _make_settings(root_path=str(tmp_path / "memback"))
    with create_stream(settings) as stream:
        frame = np.ones((8, 8), dtype="uint16") * 5
        for _ in range(6):
            stream.append(frame)

    root = tmp_path / "memback"
    assert root.exists()
    assert (root / "pos_0.dat").exists()
    assert (root / "manifest.json").exists()

    manifest = json.loads((root / "manifest.json").read_text())
    assert manifest["dtype"] == "uint16"
    assert [d["name"] for d in manifest["dimensions"]] == ["t", "c", "y", "x"]
    assert all("type" in d for d in manifest["dimensions"])
    assert manifest["dimensions"][0]["name"] == "t"
    assert manifest["dimensions"][0]["type"] == "time"
    assert manifest["dimensions"][0]["count"] == 3
    # Full settings fields are preserved
    assert "format" in manifest
    assert manifest["position_shapes"] == [[3, 2, 8, 8]]

    # Verify data is readable from memmap
    shape = tuple(manifest["position_shapes"][0])
    data = np.memmap(root / "pos_0.dat", dtype="uint16", mode="r", shape=shape)
    assert np.all(data[0, 0] == 5)


def test_tempdir_mode_multi_position(tmp_path: Path) -> None:
    dims = dims_from_standard_axes(
        {
            "p": [Position(name="A"), Position(name="B")],
            "y": 8,
            "x": 8,
        }
    )
    settings = _make_settings(root_path=str(tmp_path / "multi"), dimensions=dims)
    with create_stream(settings) as stream:
        for i in range(2):
            stream.append(np.full((8, 8), i, dtype="uint16"))

    root = tmp_path / "multi"
    assert (root / "pos_0.dat").exists()
    assert (root / "pos_1.dat").exists()
    manifest = json.loads((root / "manifest.json").read_text())
    # Position dimension should appear in the dimensions list
    dim_names = [d["name"] for d in manifest["dimensions"]]
    assert dim_names == ["p", "y", "x"]
    assert manifest["dimensions"][0]["type"] == "position"
    assert manifest["dimensions"][0]["count"] == 2
    # Position coords are preserved in the dimension
    coords = manifest["dimensions"][0]["coords"]
    assert coords[0]["name"] == "A"
    assert coords[1]["name"] == "B"
    assert manifest["position_shapes"] == [[8, 8], [8, 8]]


def test_context_manager_lifecycle() -> None:
    settings = _make_settings()
    stream = create_stream(settings)
    assert not stream.closed

    frame = np.zeros((8, 8), dtype="uint16")
    stream.append(frame)
    stream.close()
    assert stream.closed

    # Double close is safe
    stream.close()


def test_frame_metadata_jsonl(tmp_path: Path) -> None:
    settings = _make_settings(root_path=str(tmp_path / "meta"))
    with create_stream(settings) as stream:
        for i in range(6):
            stream.append(
                np.zeros((8, 8), dtype="uint16"),
                frame_metadata={"delta_t": i * 0.1, "exposure_time": 0.05},
            )

    lines = (tmp_path / "meta" / "frame_metadata.jsonl").read_text().splitlines()
    assert len(lines) == 6
    first = json.loads(lines[0])
    assert first["delta_t"] == 0.0
    assert first["exposure_time"] == 0.05
    assert "_pos" in first
    assert "_idx" in first

    last = json.loads(lines[5])
    assert last["delta_t"] == pytest.approx(0.5)
    assert last["exposure_time"] == 0.05


def test_frame_metadata_not_written_without_root_path() -> None:
    """No jsonl file when root_path is empty (pure memory mode)."""
    settings = _make_settings()
    with create_stream(settings) as stream:
        stream.append(
            np.zeros((8, 8), dtype="uint16"),
            frame_metadata={"delta_t": 0.0},
        )
    # Should not raise — metadata is silently ignored in pure memory mode
    assert stream.closed


def test_frame_metadata_skipped_when_none(tmp_path: Path) -> None:
    """No line written when frame_metadata is None."""
    settings = _make_settings(root_path=str(tmp_path / "sparse"))
    with create_stream(settings) as stream:
        stream.append(np.zeros((8, 8), dtype="uint16"))  # no metadata
        stream.append(
            np.zeros((8, 8), dtype="uint16"),
            frame_metadata={"delta_t": 1.0},
        )
        for _ in range(4):
            stream.append(np.zeros((8, 8), dtype="uint16"))  # no metadata

    lines = (tmp_path / "sparse" / "frame_metadata.jsonl").read_text().splitlines()
    assert len(lines) == 1
    assert json.loads(lines[0])["delta_t"] == 1.0


def test_event_callbacks() -> None:
    settings = _make_settings()
    events: list = []

    with create_stream(settings) as stream:
        stream.on("coords_changed", lambda u: events.append(u))
        for _ in range(6):
            stream.append(np.zeros((8, 8), dtype="uint16"))

    # Give the async executor a moment to process
    import time

    time.sleep(0.1)
    assert len(events) == 6


def test_no_root_path_required_for_scratch() -> None:
    """Scratch format should work without root_path."""
    settings = AcquisitionSettings(
        format="scratch",
        dimensions=dims_from_standard_axes({"t": 2, "y": 8, "x": 8}),
        dtype="uint16",
    )
    assert settings.root_path == ""
    assert settings.format.name == "scratch"


def test_root_path_required_for_disk_formats() -> None:
    """Disk-based formats should require root_path at stream creation."""
    settings = AcquisitionSettings(
        format="ome-zarr",
        dimensions=dims_from_standard_axes({"t": 2, "y": 8, "x": 8}),
        dtype="uint16",
    )
    with pytest.raises(ValueError, match="root_path is required"):
        create_stream(settings)


def test_scratch_format_object() -> None:
    from ome_writers import ScratchFormat

    fmt = ScratchFormat()
    assert fmt.name == "scratch"
    assert fmt.backend == "scratch"
    assert fmt.get_output_path("/tmp/foo") == "/tmp/foo"


def test_unbounded_with_tempdir(tmp_path: Path) -> None:
    dims = dims_from_standard_axes({"t": None, "y": 8, "x": 8})
    settings = _make_settings(
        root_path=str(tmp_path / "unbounded"),
        dimensions=dims,
    )
    with create_stream(settings) as stream:
        for i in range(10):
            stream.append(np.full((8, 8), i, dtype="uint16"))

    root = tmp_path / "unbounded"
    manifest = json.loads((root / "manifest.json").read_text())
    assert manifest["position_shapes"] == [[10, 8, 8]]

    shape = tuple(manifest["position_shapes"][0])
    data = np.memmap(root / "pos_0.dat", dtype="uint16", mode="r", shape=shape)
    assert np.all(data[0] == 0)
    assert np.all(data[9] == 9)


def test_acquisition_view_unbounded() -> None:
    """AcquisitionView.from_stream skips unbounded for now, but get_arrays works."""
    dims = dims_from_standard_axes({"t": None, "c": 2, "y": 8, "x": 8})
    settings = _make_settings(dimensions=dims)

    with create_stream(settings) as stream:
        arrays = stream._backend.get_arrays()
        for i in range(4):
            stream.append(np.full((8, 8), i, dtype="uint16"))

        # get_arrays works and shows live shape
        assert arrays[0].shape == (2, 2, 8, 8)
        assert np.all(arrays[0][0, 0] == 0)


def test_logical_bounds_guard() -> None:
    """Reading via get_arrays() should be bounded to logical shape."""
    dims = dims_from_standard_axes({"t": None, "c": 2, "y": 8, "x": 8})
    settings = _make_settings(dimensions=dims)
    with create_stream(settings) as stream:
        arrays = stream._backend.get_arrays()
        # Write 2 frames (t=0, c=0 and c=1)
        stream.append(np.ones((8, 8), dtype="uint16"))
        stream.append(np.ones((8, 8), dtype="uint16"))

        # Logical shape should be (1, 2, 8, 8)
        assert arrays[0].shape == (1, 2, 8, 8)
        # Full slice should return only the logical region
        result = arrays[0][:]
        assert result.shape == (1, 2, 8, 8)

        # Out-of-logical-range indexing should raise
        with pytest.raises(IndexError):
            arrays[0][1, 0]


def test_storage_order_scratch_uses_acquisition_order() -> None:
    """Scratch format with storage_order='ome' should use acquisition order."""
    settings = AcquisitionSettings(
        format="scratch",
        dimensions=dims_from_standard_axes({"c": 2, "t": 3, "y": 8, "x": 8}),
        dtype="uint16",
        storage_order="ome",
    )
    # For memory, storage order should match acquisition order (c, t)
    assert [d.name for d in settings.storage_index_dimensions] == ["c", "t"]
