"""Tests for live_shape mode on StreamView."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest

from ome_writers import AcquisitionSettings, Channel, Dimension, Position, create_stream
from ome_writers._stream_view import StreamView
from tests._utils import wait_for_pending_callbacks

if TYPE_CHECKING:
    from pathlib import Path


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
        c = stream.view(live_shape=False).coords
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
        view = stream.view(live_shape=True)

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
        view = stream.view(live_shape=True, strict=True)
        stream.append(FRAME)  # shape → (1, 1, 16, 16)
        wait_for_pending_callbacks(stream)

        if should_raise:
            with pytest.raises(IndexError, match="out of bounds"):
                view[index]
        else:
            assert view[index].shape == (16, 16)


def test_non_strict_allows_over_indexing(tmp_path: Path, first_backend: str) -> None:
    """live_shape=True without strict returns zeros beyond live bounds."""
    with create_stream(_settings(tmp_path, first_backend, TC_DIMS)) as stream:
        view = stream.view(live_shape=True)
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
        view = stream.view(live_shape=True)
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

        view1 = stream.view(live_shape=True)
        assert view1.shape == (2, 2, 16, 16)

        view2 = stream.view(live_shape=True)
        assert view2.shape == (2, 2, 16, 16)

        stream.append(FRAME)  # t=1,c=1 (no HWM)
        stream.append(FRAME)  # t=2,c=0 (HWM)
        wait_for_pending_callbacks(stream)
        assert view1.shape == (3, 2, 16, 16)
        assert view2.shape == (3, 2, 16, 16)


def test_non_live_full_shape(tmp_path: Path, first_backend: str) -> None:
    """Regression: live_shape=False always returns full shape."""
    with create_stream(_settings(tmp_path, first_backend, TC_DIMS)) as stream:
        view = stream.view(live_shape=False)
        assert view.shape == (3, 2, 16, 16)
        stream.append(FRAME)
        assert view.shape == (3, 2, 16, 16)
