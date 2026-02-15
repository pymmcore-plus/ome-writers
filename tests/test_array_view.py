"""Test array view with all permutations of dimension orders."""

from __future__ import annotations

from itertools import permutations
from typing import TYPE_CHECKING

import numpy as np
import pytest

from ome_writers import AcquisitionSettings, Dimension, Position
from ome_writers._array_view import AcquisitionView
from ome_writers._frame_encoder import frame_generator, validate_encoded_frame_values
from ome_writers._stream import create_stream
from tests._utils import wait_for_frames

if TYPE_CHECKING:
    from pathlib import Path

# Dimension specifications
DIM_SPECS = {
    "p": {
        "type": "position",
        "coords": [
            Position(name="pos0", x_coord=0.0, y_coord=0.0),
            Position(name="pos1", x_coord=100.0, y_coord=0.0),
        ],
    },
    "t": {"count": 3, "chunk_size": 1, "type": "time"},
    "c": {"count": 2, "chunk_size": 1, "type": "channel"},
    "z": {"count": 4, "chunk_size": 1, "type": "space"},
    "y": {"count": 16, "chunk_size": 16, "type": "space"},
    "x": {"count": 16, "chunk_size": 16, "type": "space"},
}

NP = len(DIM_SPECS["p"]["coords"])
NC = DIM_SPECS["c"]["count"]
NT = DIM_SPECS["t"]["count"]
NZ = DIM_SPECS["z"]["count"]
NY = DIM_SPECS["y"]["count"]
NX = DIM_SPECS["x"]["count"]

DIM_ORDERS: list[str] = ["".join(p) + "yx" for p in permutations("tpcz")]
# add a few dim orders without one of tpc or z ...
DIM_ORDERS += ["pyx", "tpyx", "cpzyx", "ctyx"]


@pytest.mark.parametrize("dim_order", DIM_ORDERS)
def test_array_view(tmp_path: Path, dim_order: str, any_backend: str) -> None:
    """Test that array view works correctly for all dimension orderings.

    This tests all 24 permutations of (t, p, c, z) with y, x always at the end.
    """
    if any_backend == "tifffile":
        pytest.importorskip(
            "zarr", reason="tifffile backend requires zarr for array view"
        )
    if any_backend == "acquire-zarr":
        pytest.skip("acquire-zarr doesn't support read-only views")

    settings = AcquisitionSettings(
        root_path=tmp_path / f"test_{dim_order}",
        dimensions=[Dimension(name=dim, **DIM_SPECS[dim]) for dim in dim_order],
        dtype="uint16",
        overwrite=True,
        format=any_backend,
    )

    expected_frames = settings.num_frames or 1

    frames = frame_generator(settings)
    with create_stream(settings) as stream:
        view = AcquisitionView.from_stream(stream)
        for i, frame in enumerate(frames):
            stream.append(frame)

            # halfway through...
            if expected_frames > 20 and i == (expected_frames // 2):
                # we should be able to see early frames,
                wait_for_frames(stream._backend, expected_count=1)
                first_idx = (0,) * (view.ndim - 2)
                first_frame = view[first_idx]
                assert not np.allclose(first_frame, 0)

                # while later frames should still be zero (but not an error)
                last_idx = (-1,) * (view.ndim - 2)  # exercise negative indexing
                last_frame = view[last_idx]
                assert np.allclose(last_frame, 0)

    assert view.dims == list(dim_order)
    assert repr(view)
    assert len(view) == settings.shape[0]

    # Test basic indexing works
    non_xy_dims = len(view.shape) - 2  # All except y, x
    result = view[(0,) * non_xy_dims]
    assert result.shape == (NY, NX)

    # Test slicing works - get first slice of non-spatial dims
    result = view[(slice(0, 1),) * non_xy_dims]
    assert result.shape == (1,) * (non_xy_dims) + (NY, NX)

    arr = np.asarray(view)
    assert isinstance(arr, np.ndarray)
    assert arr.shape == view.shape == settings.shape
    assert arr.dtype == view.dtype == settings.dtype

    # Get dimension names in acquisition order (view order),
    # excluding position and spatial dims
    expected_names = [d.name for d in settings.index_dimensions]
    # test_position_slicing:
    if (pos_ax := settings.position_dimension_index) is not None:
        # Take first index of all non-xy dims, except slice all positions
        index = tuple(slice(None) if i == pos_ax else 0 for i in range(non_xy_dims))
        result = view[index]
        assert result.shape == (NP, NY, NX)

        for pos_idx in range(NP):
            ary = np.take(arr, pos_idx, axis=settings.position_dimension_index)
            validate_encoded_frame_values(ary, expected_names, pos_idx=pos_idx)


def test_array_view_on_closed_stream(tmp_path: Path, first_backend: str) -> None:
    """Test that array view works correctly for all dimension orderings.

    This tests all 24 permutations of (t, p, c, z) with y, x always at the end.
    """
    dim_order = "tzyx"
    settings = AcquisitionSettings(
        root_path=tmp_path / f"test_{dim_order}",
        dimensions=[Dimension(name=dim, **DIM_SPECS[dim]) for dim in dim_order],
        dtype="uint16",
        overwrite=True,
        format=first_backend,
    )

    stream = create_stream(settings)
    stream.close()
    assert stream.closed

    with pytest.raises(NotImplementedError, match="Creating a view on a closed stream"):
        AcquisitionView.from_stream(stream)


def test_norm_index() -> None:
    from ome_writers._array_view import _norm_index

    assert _norm_index(0, 10) == 0
    assert _norm_index(5, 10) == 5
    assert _norm_index(-1, 10) == 9
    assert _norm_index(-10, 10) == 0

    assert _norm_index(slice(0, 5), 10) == slice(0, 5)
    assert _norm_index(slice(-5, -1), 10) == slice(5, 9)
    assert _norm_index(slice(-5, None), 10) == slice(5, None)
    assert _norm_index(slice(None, -5), 10) == slice(None, 5)
