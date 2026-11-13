from __future__ import annotations

import json
from itertools import groupby, pairwise
from typing import TYPE_CHECKING, NamedTuple

import numpy as np
import pytest

from ome_writers._schema import AcquisitionSettings, Dimension, dims_from_standard_axes
from ome_writers._stream import create_stream
from tests._utils import read_array_data

if TYPE_CHECKING:
    from pathlib import Path

az = pytest.importorskip("acquire_zarr", reason="acquire-zarr not installed")


def test_acquire_zarr_full_streaming_support(tmp_path: Path) -> None:
    """Test that our backend abstraction doesn't break cool acquire-zarr features.

    One very nice thing about acquire-zarr's stream is that it makes no assumptions
    about shape of each buffer being passed to `stream.append()`,  C-contiguous
    buffers are simply concatenated according to the dimensionality declared in
    the settings.

    This test ensures that our backend abstraction preserves this behavior.
    """

    settings = AcquisitionSettings(
        root_path=str(tmp_path / "output.zarr"),
        dimensions=[
            Dimension(name="z", count=18, chunk_size=6, unit="um", scale=0.5),
            Dimension(name="y", count=128, chunk_size=64, unit="um", scale=0.1),
            Dimension(name="x", count=128, chunk_size=64, unit="um", scale=0.1),
        ],
        dtype="uint16",
        format="acquire-zarr",
    )

    shape = tuple(d.count or 1 for d in settings.dimensions)
    flat_data = np.arange(np.prod(shape), dtype=settings.dtype)
    # break the data into 10 arbitrary, non-frame/chunk-aligned, somewhat random pieces
    boundaries = [0, 1500, 3000, 5000, 7000, 9000, 12000, 15000, 18000, 22000, None]
    append_bits = [flat_data[start:stop] for start, stop in pairwise(boundaries)]

    with create_stream(settings) as stream:
        for bit in append_bits:
            stream.append(bit)

    output_data = read_array_data(settings.output_path)
    assert output_data.shape == (18, 128, 128)
    assert output_data.dtype == np.dtype(settings.dtype)
    assert np.array_equal(output_data.flatten(), flat_data)


# acquire-zarr's `ZarrStream.append` takes buffers of any size: each array (key) is a
# flat byte stream, filled in order.  ome-writers promises the same pass-through
# (https://github.com/pymmcore-plus/ome-writers/issues/41), taking the position (key)
# of each append from acquisition order, and splitting an append that spans several
# positions.  Each case writes the same data both ways (directly, appends are split
# at position boundaries), and both must give the same result.

Y, X = 4, 8
FRAME = Y * X
HALF = FRAME // 2
N_UNBOUNDED = 3  # frames written along an unbounded dimension


class _Case(NamedTuple):
    sizes: dict[str, int | None]  # non-frame dims, in acquisition order
    piece: float | str  # frames per append, or "all" (whole position / acquisition)
    span_positions: bool = False  # if True, appends may span several positions


PARTIAL_CASES = {
    "1pos-half-frames": _Case({"t": 3}, 0.5),
    "1pos-pixel-by-pixel": _Case({"t": 3}, 1 / FRAME),
    "1pos-0.75-frames": _Case({"t": 3}, 0.75),
    "1pos-tc-half-frames": _Case({"t": 3, "c": 2}, 0.5),
    "1pos-tzc-permuted-half-frames": _Case({"t": 2, "z": 2, "c": 2}, 0.5),
    "1pos-unbounded-half-frames": _Case({"t": None}, 0.5),  # works on main
    "p-inner-half-frames": _Case({"t": 3, "p": 2}, 0.5),
    "p-inner-unbounded-half-frames": _Case({"t": None, "p": 2}, 0.5),
    "p-outer-half-frames": _Case({"p": 2, "t": 3}, 0.5),
    "p-outer-1.5-frames": _Case({"p": 2, "t": 3}, 1.5),
    "p-outer-whole-position": _Case({"p": 2, "t": 3}, "all"),
    # appends spanning positions
    "p-inner-1.5-frames-spanning": _Case({"t": 3, "p": 2}, 1.5, True),
    "p-inner-whole-array": _Case({"t": 3, "p": 2}, "all", True),
    "p-inner-unbounded-whole-array": _Case({"t": None, "p": 2}, "all", True),
    "p-outer-2-frames-spanning": _Case({"p": 2, "t": 3}, 2, True),
    "p-outer-whole-array": _Case({"p": 2, "t": 3}, "all", True),
}


def _test_data(sizes: dict[str, int | None]) -> np.ndarray:
    """Unique values, shape (*acquisition_dims, Y, X)."""
    shape = [n or N_UNBOUNDED for n in sizes.values()]
    return np.arange(1, np.prod(shape) * FRAME + 1, dtype="uint16").reshape(
        *shape, Y, X
    )


def _appends(case: _Case, data: np.ndarray) -> list[list[tuple[int, np.ndarray]]]:
    """Split `data` into appends, each as [(position, part), ...] per position."""
    p_ax = list(case.sizes).index("p") if "p" in case.sizes else None
    frame_pos = [0 if p_ax is None else i[p_ax] for i in np.ndindex(data.shape[:-2])]
    # (position, start, stop) element ranges of same-position runs
    runs, start = [], 0
    for pos, frames in groupby(frame_pos):
        stop = start + len(list(frames)) * FRAME
        runs.append((pos, start, stop))
        start = stop

    spans = [(0, start)] if case.span_positions else [(a, b) for _, a, b in runs]
    ranges = []
    for a, b in spans:
        step = b - a if case.piece == "all" else int(case.piece * FRAME)
        ranges += [(i, min(i + step, b)) for i in range(a, b, step)]
    # split each append where it crosses into another position's run
    flat = data.ravel()
    return [
        [(p, flat[max(a, ra) : min(b, rb)]) for p, ra, rb in runs if ra < b and a < rb]
        for a, b in ranges
    ]


def _write_direct(root: Path, case: _Case, appends: list) -> None:
    """Write with acquire-zarr's own API, one array per position."""
    names = [k for k in case.sizes if k != "p"]
    storage = sorted(names, key="tcz".index)  # same OME order ome-writers uses
    dims = [(k, case.sizes[k] or 0, 1) for k in names] + [("y", Y, Y), ("x", X, X)]
    n_pos = case.sizes.get("p") or 1
    keys = [f"{i}/0" for i in range(n_pos)] if n_pos > 1 else ["0"]
    array = {
        "data_type": np.uint16,
        "dimensions": [
            az.Dimension(
                name=name,
                kind=az.DimensionType.SPACE,
                array_size_px=size,
                chunk_size_px=chunk,
                shard_size_chunks=1,
            )
            for name, size, chunk in dims
        ],
        "storage_dimension_order": None if storage == names else [*storage, "y", "x"],
    }
    stream = az.ZarrStream(
        az.StreamSettings(
            arrays=[az.ArraySettings(output_key=key, **array) for key in keys],
            store_path=str(root),
            version=az.ZarrVersion.V3,
        )
    )
    for parts in appends:
        for pos, part in parts:
            stream.append(part, key=keys[pos])
    stream.close()


def _settings(root: Path, sizes: dict[str, int | None]) -> AcquisitionSettings:
    return AcquisitionSettings(
        root_path=str(root),
        dimensions=dims_from_standard_axes({**sizes, "y": Y, "x": X}),
        dtype="uint16",
        format="acquire-zarr",
    )


def _write_ome_writers(root: Path, case: _Case, appends: list) -> None:
    """Write through ome-writers, which has no `key`: position is implied."""
    with create_stream(_settings(root, case.sizes)) as stream:
        for parts in appends:
            stream.append(np.concatenate([part for _, part in parts]))


def _assert_positions(root: Path, sizes: dict, data: np.ndarray) -> None:
    """Each position's array on disk is its part of `data`, in OME (TCZYX) order."""
    names = list(sizes)
    per_pos = list(np.moveaxis(data, names.index("p"), 0)) if "p" in names else [data]
    names = [n for n in names if n != "p"]
    perm = [names.index(n) for n in sorted(names, key="tcz".index)]
    for i, expected in enumerate(per_pos):
        actual = read_array_data(root / str(i) if len(per_pos) > 1 else root)
        np.testing.assert_array_equal(actual, expected.transpose(*perm, -2, -1))


@pytest.mark.parametrize("via", ["acquire-zarr", "ome-writers"])
@pytest.mark.parametrize("case", PARTIAL_CASES.values(), ids=PARTIAL_CASES.keys())
def test_acquire_zarr_partial_appends(tmp_path: Path, case: _Case, via: str) -> None:
    data = _test_data(case.sizes)
    root = tmp_path / "out.zarr"
    write = _write_direct if via == "acquire-zarr" else _write_ome_writers
    write(root, case, _appends(case, data))
    _assert_positions(root, case.sizes, data)


# Rules for appends that aren't exactly one frame (the stream tracks how many
# elements have been written, not how many appends were made):


def test_acquire_zarr_skip_mid_frame(tmp_path: Path) -> None:
    """`skip(frames=n)` goes to the start of the n-th next frame, from mid-frame."""
    root = tmp_path / "out.zarr"
    ones = np.ones(HALF, "uint16")
    # acquisition order: (t0,p0) (t0,p1) (t1,p0) (t1,p1) (t2,p0) (t2,p1)
    with create_stream(_settings(root, {"t": 3, "p": 2})) as stream:
        stream.append(ones)  # first half of (t0,p0)
        stream.skip(frames=2)  # rest of (t0,p0), all of (t0,p1)
        stream.append(ones * 2)  # (t1,p0), in two halves...
        stream.append(ones * 2)  # ...ending exactly on a frame boundary
        stream.skip(frames=1)  # so this skips all of (t1,p1)
        stream.append(ones * 3)  # first half of (t2,p0)
        stream.skip(frames=1)  # rest of (t2,p0)
        stream.append(np.full(FRAME, 4, "uint16"))  # (t2,p1)

    pos0, pos1 = np.zeros((2, 3, FRAME), "uint16")
    pos0[0, :HALF], pos0[1], pos0[2, :HALF] = 1, 2, 3
    pos1[2] = 4
    for i, expected in enumerate([pos0, pos1]):
        actual = read_array_data(root / str(i))
        np.testing.assert_array_equal(actual, expected.reshape(3, Y, X))


def test_acquire_zarr_partial_frame_metadata(tmp_path: Path) -> None:
    """`frame_metadata` belongs to the frame where the append's first element lands.

    A second `frame_metadata` for the same frame warns, and is discarded.
    """
    root = tmp_path / "out.zarr"
    sizes: dict[str, int | None] = {"t": 2, "p": 2}
    data = _test_data(sizes).ravel()
    # acquisition order: (t0,p0) (t0,p1) (t1,p0) (t1,p1)
    with create_stream(_settings(root, sizes)) as stream:
        stream.append(data[:HALF], frame_metadata={"i": "a"})  # (t0,p0)
        with pytest.warns(UserWarning):
            # starts in (t0,p0) too, which already has metadata: discarded
            stream.append(data[HALF : 3 * HALF], frame_metadata={"i": "dup"})
        # starts mid-(t0,p1)
        stream.append(data[3 * HALF : 4 * HALF], frame_metadata={"i": "b"})
        # starts exactly at (t1,p0), spans into (t1,p1)
        stream.append(data[4 * HALF :], frame_metadata={"i": "c"})

    def _meta(pos: int) -> list:
        zarr_json = json.loads((root / str(pos) / "zarr.json").read_text())
        return zarr_json["attributes"]["ome_writers"]["frame_metadata"]

    assert _meta(0) == [
        {"i": "a", "storage_index": [0]},
        {"i": "c", "storage_index": [1]},
    ]
    assert _meta(1) == [{"i": "b", "storage_index": [0]}]
    # only the metadata is discarded, never the data
    _assert_positions(root, sizes, data.reshape(2, 2, Y, X))


@pytest.mark.parametrize(
    ("sizes", "before", "too_much"),
    [({"t": 2}, 1.5, 1), ({"t": 2, "p": 2}, 2.5, 2)],
    ids=["1pos", "p-inner-spanning"],
)
def test_acquire_zarr_append_overflow(
    tmp_path: Path, sizes: dict, before: float, too_much: float
) -> None:
    """An append past the declared size raises before writing any of it.

    Like acquire-zarr, the stream stays usable: the next append that fits lands
    where the rejected one would have started.
    """
    root = tmp_path / "out.zarr"
    data = _test_data(sizes)
    flat, n = data.ravel(), int(before * FRAME)
    with create_stream(_settings(root, sizes)) as stream:
        stream.append(flat[:n])
        with pytest.raises(IndexError):
            stream.append(np.full(int(too_much * FRAME), 999, "uint16"))
        stream.append(flat[n:])
    _assert_positions(root, sizes, data)
