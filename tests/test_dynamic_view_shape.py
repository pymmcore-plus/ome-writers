"""Tests for dynamic_shape mode on StreamView."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest

from ome_writers import AcquisitionSettings, Channel, Dimension, Position, create_stream
from ome_writers._stream_view import StreamView
from tests._utils import wait_for_pending_callbacks

if TYPE_CHECKING:
    from pathlib import Path


@pytest.fixture()
def first_backend(first_backend: str) -> str:
    """Skip view tests when tiff backend lacks zarr for live-viewing."""
    if first_backend == "tifffile":
        pytest.importorskip(
            "zarr.abc.store",
            reason="zarr v3 required for live-viewing tiff data",
        )
    return first_backend


FRAME = np.zeros((16, 16), dtype=np.uint16)
TC_DIMS = [
    Dimension(name="t", count=3, type="time"),
    Dimension(name="c", count=2, type="channel"),
    Dimension(name="y", count=16, type="space"),
    Dimension(name="x", count=16, type="space"),
]


def _settings(
    tmp_path: Path, backend: str, dims: list[Dimension]
) -> AcquisitionSettings:
    return AcquisitionSettings(
        root_path=tmp_path / "test",
        dimensions=dims,
        dtype="uint16",
        format=backend,
        overwrite=True,
    )


@pytest.mark.parametrize(
    "ch_dim, expected_c",
    [
        (
            Dimension(name="c", coords=[Channel(name="DAPI"), Channel(name="GFP")]),
            ["DAPI", "GFP"],
        ),
        (Dimension(name="c", count=2, type="channel"), range(2)),
    ],
    ids=["named_channels", "range_channels"],
)
def test_coords_non_live(
    tmp_path: Path, first_backend: str, ch_dim: Dimension, expected_c: object
) -> None:
    """Non-live view returns full coords from settings."""
    dims = [Dimension(name="t", count=3, type="time"), ch_dim, *TC_DIMS[2:]]
    with create_stream(_settings(tmp_path, first_backend, dims)) as stream:
        c = stream.view(dynamic_shape=False).coords
        assert c["t"] == range(3)
        assert c["c"] == expected_c
        assert c["y"] == range(16)


def test_coords_fallback_without_from_stream() -> None:
    """Direct __init__ (no from_stream) falls back to range-based coords."""

    view = StreamView(
        [np.zeros((4, 8, 8), dtype=np.uint16)],
        position_axis=None,
        dimension_labels=["t", "y", "x"],
    )
    assert view.coords == {"t": range(4), "y": range(8), "x": range(8)}


def test_dynamic_shape_and_coords(tmp_path: Path, first_backend: str) -> None:
    """Shape/coords start at zero and expand at each high water mark."""
    dims = [
        Dimension(name="t", count=3, type="time"),
        Dimension(name="c", coords=[Channel(name="DAPI"), Channel(name="GFP")]),
        *TC_DIMS[2:],
    ]
    with create_stream(_settings(tmp_path, first_backend, dims)) as stream:
        view = stream.view(dynamic_shape=True)

        # Initial: zero for non-frame dims
        assert view.shape == (0, 0, 16, 16)
        assert len(view) == 0
        assert view.coords["c"] == []

        expected = [
            # (shape_after, t_range, c_coords)
            ((1, 1, 16, 16), range(1), ["DAPI"]),
            ((1, 2, 16, 16), range(1), ["DAPI", "GFP"]),
            ((2, 2, 16, 16), range(2), ["DAPI", "GFP"]),
            ((2, 2, 16, 16), range(2), ["DAPI", "GFP"]),  # no HWM
            ((3, 2, 16, 16), range(3), ["DAPI", "GFP"]),
        ]
        for shape, t_range, c_coords in expected:
            stream.append(FRAME)
            wait_for_pending_callbacks(stream)
            assert view.shape == shape
            assert view.coords["t"] == t_range
            assert view.coords["c"] == c_coords


@pytest.mark.parametrize(
    "index, should_raise",
    [
        ((0, 0), False),  # within bounds
        ((1, 0), True),  # t out of bounds
        ((0, 1), True),  # c out of bounds
        ((-1, 0), False),  # -1 resolves to 0 (valid for size 1)
        ((-2, 0), True),  # -2 resolves to -1 (invalid for size 1)
    ],
)
def test_strict_bounds(
    tmp_path: Path,
    first_backend: str,
    index: tuple[int, int],
    should_raise: bool,
) -> None:
    """strict=True checks integer indices against live shape."""
    with create_stream(_settings(tmp_path, first_backend, TC_DIMS)) as stream:
        view = stream.view(dynamic_shape=True, strict=True)
        stream.append(FRAME)  # shape → (1, 1, 16, 16)
        wait_for_pending_callbacks(stream)

        if should_raise:
            with pytest.raises(IndexError, match="out of bounds"):
                view[index]
        else:
            assert view[index].shape == (16, 16)


def test_non_strict_allows_over_indexing(tmp_path: Path, first_backend: str) -> None:
    """dynamic_shape=True without strict returns zeros beyond live bounds."""
    with create_stream(_settings(tmp_path, first_backend, TC_DIMS)) as stream:
        view = stream.view(dynamic_shape=True)
        stream.append(FRAME)  # live shape (1, 1, 16, 16)
        wait_for_pending_callbacks(stream)
        assert np.allclose(view[2, 0], 0)


def test_position_dimension_tracking(tmp_path: Path, first_backend: str) -> None:
    """Position dim size reflects only visited positions."""
    dims = [
        Dimension(name="t", count=2, type="time"),
        Dimension(
            name="p",
            coords=[
                Position(name="pos0"),
                Position(name="pos1"),
                Position(name="pos2"),
            ],
        ),
        *TC_DIMS[2:],
    ]
    with create_stream(_settings(tmp_path, first_backend, dims)) as stream:
        view = stream.view(dynamic_shape=True)
        assert view.shape == (0, 0, 16, 16)

        stream.append(FRAME)  # t=0, p=0
        wait_for_pending_callbacks(stream)
        assert view.coords["p"] == ["pos0"]

        stream.append(FRAME)  # t=0, p=1
        wait_for_pending_callbacks(stream)
        assert view.coords["p"] == ["pos0", "pos1"]


def test_mid_acquisition_and_multiple_views(tmp_path: Path, first_backend: str) -> None:
    """Mid-acquisition view starts correct; second view also tracks."""
    with create_stream(_settings(tmp_path, first_backend, TC_DIMS)) as stream:
        for _ in range(3):  # t=0,c=0 / t=0,c=1 / t=1,c=0
            stream.append(FRAME)

        view1 = stream.view(dynamic_shape=True)
        assert view1.shape == (2, 2, 16, 16)

        view2 = stream.view(dynamic_shape=True)
        assert view2.shape == (2, 2, 16, 16)

        stream.append(FRAME)  # t=1,c=1 (no HWM)
        stream.append(FRAME)  # t=2,c=0 (HWM)
        wait_for_pending_callbacks(stream)
        assert view1.shape == (3, 2, 16, 16)
        assert view2.shape == (3, 2, 16, 16)


def test_non_live_full_shape(tmp_path: Path, first_backend: str) -> None:
    """Regression: dynamic_shape=False always returns full shape."""
    with create_stream(_settings(tmp_path, first_backend, TC_DIMS)) as stream:
        view = stream.view(dynamic_shape=False)
        assert view.shape == (3, 2, 16, 16)
        stream.append(FRAME)
        assert view.shape == (3, 2, 16, 16)


# ---- Unbounded stream tests ----

UNBOUNDED_TC_DIMS = [
    Dimension(name="t", count=None, type="time"),
    Dimension(name="c", count=2, type="channel"),
    Dimension(name="y", count=16, type="space"),
    Dimension(name="x", count=16, type="space"),
]


def test_unbounded_dynamic_shape(tmp_path: Path, first_backend: str) -> None:
    """Shape grows from (0,0,16,16) as frames are written with dynamic_shape."""
    with create_stream(_settings(tmp_path, first_backend, UNBOUNDED_TC_DIMS)) as stream:
        view = stream.view(dynamic_shape=True)

        # Initial: zero for non-frame dims
        assert view.shape == (0, 0, 16, 16)

        expected = [
            # (shape_after, t_coords, c_coords)
            ((1, 1, 16, 16), range(1), range(1)),  # t=0, c=0 -> HWM
            ((1, 2, 16, 16), range(1), range(2)),  # t=0, c=1 -> HWM (c grows)
            ((2, 2, 16, 16), range(2), range(2)),  # t=1, c=0 -> HWM (t grows)
            ((2, 2, 16, 16), range(2), range(2)),  # t=1, c=1 -> no HWM
            ((3, 2, 16, 16), range(3), range(2)),  # t=2, c=0 -> HWM (t grows)
        ]
        for shape, t_range, c_range in expected:
            stream.append(FRAME)
            wait_for_pending_callbacks(stream)
            assert view.shape == shape
            assert view.coords["t"] == t_range
            assert view.coords["c"] == c_range


def test_unbounded_non_dynamic_shape() -> None:
    """dynamic_shape=False uses current array extent for unbounded dims."""
    settings = AcquisitionSettings(
        format="scratch",
        dimensions=UNBOUNDED_TC_DIMS,
        dtype="uint16",
    )
    with create_stream(settings) as stream:
        for _ in range(6):  # 3 timepoints * 2 channels
            stream.append(FRAME)

        view = stream.view(dynamic_shape=False)
        assert view.shape == (3, 2, 16, 16)
        assert view.coords["t"] == range(3)
        assert view.coords["c"] == range(2)


def test_unbounded_single_dim_dynamic() -> None:
    """(None, y, x) — every frame is a HWM."""
    dims = [
        Dimension(name="t", count=None, type="time"),
        Dimension(name="y", count=16, type="space"),
        Dimension(name="x", count=16, type="space"),
    ]
    settings = AcquisitionSettings(format="scratch", dimensions=dims, dtype="uint16")

    with create_stream(settings) as stream:
        view = stream.view(dynamic_shape=True)
        assert view.shape == (0, 16, 16)

        for i in range(5):
            stream.append(FRAME)
            wait_for_pending_callbacks(stream)
            assert view.shape == (i + 1, 16, 16)
            assert view.coords["t"] == range(i + 1)


def test_unbounded_skip_no_inner_dims() -> None:
    """skip() on (None, y, x) — no inner dims, only outer dim matters."""
    dims = [
        Dimension(name="t", count=None, type="time"),
        Dimension(name="y", count=16, type="space"),
        Dimension(name="x", count=16, type="space"),
    ]
    settings = AcquisitionSettings(format="scratch", dimensions=dims, dtype="uint16")

    with create_stream(settings) as stream:
        view = stream.view(dynamic_shape=True)

        stream.append(FRAME)  # t=0
        wait_for_pending_callbacks(stream)
        assert view.shape == (1, 16, 16)

        stream.skip(frames=5)  # skip t=1..5
        wait_for_pending_callbacks(stream)
        assert view.shape == (6, 16, 16)
        assert view.coords["t"] == range(6)


def test_unbounded_skip_full_cycle() -> None:
    """skip() spanning >= inner_product hits all inner HWMs at once."""
    settings = AcquisitionSettings(
        format="scratch", dimensions=UNBOUNDED_TC_DIMS, dtype="uint16"
    )
    with create_stream(settings) as stream:
        view = stream.view(dynamic_shape=True)

        # Skip a full cycle (inner_product=2): should max out both t and c
        stream.skip(frames=2)
        wait_for_pending_callbacks(stream)
        assert view.shape == (1, 2, 16, 16)
        assert view.coords["t"] == range(1)
        assert view.coords["c"] == range(2)


def test_unbounded_skip_partial_cycle() -> None:
    """skip() spanning < inner_product scans individual frames."""
    settings = AcquisitionSettings(
        format="scratch", dimensions=UNBOUNDED_TC_DIMS, dtype="uint16"
    )
    with create_stream(settings) as stream:
        view = stream.view(dynamic_shape=True)

        # Skip 1 frame (< inner_product=2): only c=0 seen
        stream.skip(frames=1)
        wait_for_pending_callbacks(stream)
        assert view.shape == (1, 1, 16, 16)
        assert view.coords["c"] == range(1)

        # Skip 1 more: c=1 seen, still t=0
        stream.skip(frames=1)
        wait_for_pending_callbacks(stream)
        assert view.shape == (1, 2, 16, 16)


def test_unbounded_mid_acquisition_full_cycle() -> None:
    """Mid-acquisition view on unbounded stream after full inner cycle."""
    settings = AcquisitionSettings(
        format="scratch", dimensions=UNBOUNDED_TC_DIMS, dtype="uint16"
    )
    with create_stream(settings) as stream:
        # Write 6 frames (3t * 2c) before creating view
        for _ in range(6):
            stream.append(FRAME)

        # Creating view mid-acquisition triggers _init_unbounded_max_indices
        # with fc=6 >= inner_product=2 (full cycle branch)
        view = stream.view(dynamic_shape=True)
        assert view.shape == (3, 2, 16, 16)
        assert view.coords["t"] == range(3)
        assert view.coords["c"] == range(2)


def test_unbounded_mid_acquisition_partial_cycle() -> None:
    """Mid-acquisition view on unbounded stream before full inner cycle."""
    settings = AcquisitionSettings(
        format="scratch", dimensions=UNBOUNDED_TC_DIMS, dtype="uint16"
    )
    with create_stream(settings) as stream:
        # Write 1 frame (fc=1 < inner_product=2, partial cycle branch)
        stream.append(FRAME)

        view = stream.view(dynamic_shape=True)
        assert view.shape == (1, 1, 16, 16)
        assert view.coords["t"] == range(1)
        assert view.coords["c"] == range(1)


def test_unbounded_many_frames() -> None:
    """Verify works beyond old 10000 sentinel."""
    dims = [
        Dimension(name="t", count=None, type="time"),
        Dimension(name="y", count=16, type="space"),
        Dimension(name="x", count=16, type="space"),
    ]
    settings = AcquisitionSettings(format="scratch", dimensions=dims, dtype="uint16")
    n_frames = 100

    with create_stream(settings) as stream:
        view = stream.view(dynamic_shape=True)

        for _ in range(n_frames):
            stream.append(FRAME)

        wait_for_pending_callbacks(stream)
        assert view.shape == (n_frames, 16, 16)
        assert view.coords["t"] == range(n_frames)
