"""Tests for chunk/shard buffering, and for reading frames that are still buffered."""

from __future__ import annotations

import random
import threading
import time
from typing import TYPE_CHECKING, Any

import numpy as np
import pytest

from ome_writers import AcquisitionSettings, Dimension, create_stream
from ome_writers._memory import _bytes_per_chunk, _max_chunks
from ome_writers._router import FrameRouter
from tests._utils import read_array_data

if TYPE_CHECKING:
    from pathlib import Path

H, W = 8, 10
Y = Dimension(name="y", count=H, chunk_size=H, type="space")
X = Dimension(name="x", count=W, chunk_size=W, type="space")
P = Dimension(name="p", type="position", coords=["a", "b"])
UNBOUNDED = 10**9  # stand-in count, for computing indices of unbounded dimensions


def _dim(name: str, type_: str) -> Any:
    def make(count: int | None, chunk: int = 1, shard: int | None = None) -> Dimension:
        return Dimension(
            name=name,
            count=count,
            chunk_size=chunk,
            shard_size_chunks=shard,
            type=type_,
        )

    return make


T, C, Z = _dim("t", "time"), _dim("c", "channel"), _dim("z", "space")


def _settings(
    tmp_path: Path, backend: str, dims: list[Dimension], **kwargs: Any
) -> AcquisitionSettings:
    return AcquisitionSettings(
        root_path=tmp_path / "x.ome.zarr",
        dimensions=dims,
        dtype="uint16",
        format={"name": "ome-zarr", "backend": backend},
        overwrite=True,
        **kwargs,
    )


def _frame(i: int, shape: tuple[int, ...] = (H, W)) -> np.ndarray:
    """Return a frame unique to `i`, whose every pixel is `i + 1`."""
    return np.full(shape, i + 1, dtype="uint16")


def _textured(i: int) -> np.ndarray:
    """Return a frame unique to `i`, whose every pixel differs."""
    return np.arange(1, H * W + 1, dtype="uint16").reshape(H, W) + i


def _key(settings: AcquisitionSettings, i: int) -> tuple[int, ...]:
    """Return the acquisition-order index of frame `i`."""
    counts = [d.count or UNBOUNDED for d in settings.dimensions[:-2]]
    return tuple(int(v) for v in np.unravel_index(i, counts))


def _write_and_verify(settings: AcquisitionSettings, ops: list[int]) -> None:
    """Append (positive) or skip (negative) frames, close, then check all data."""
    router = iter(FrameRouter(settings))
    frame_shape = tuple(d.count for d in settings.array_storage_dimensions[-2:])
    written, seen, i = [], [], 0
    with create_stream(settings) as stream:
        for op in ops:
            if op < 0:
                stream.skip(frames=-op)
                seen.extend(next(router)[1] for _ in range(-op))
                i += -op
                continue
            for _ in range(op):
                pos, idx = next(router)
                stream.append(_frame(i, frame_shape))
                written.append((pos, idx, i + 1))
                seen.append(idx)
                i += 1

    index_shape = tuple(
        d.count if d.count is not None else max(idx[axis] for idx in seen) + 1
        for axis, d in enumerate(settings.array_storage_dimensions[:-2])
    )
    positions = settings.positions
    for p, position in enumerate(positions):
        want = np.zeros(index_shape + frame_shape, dtype="uint16")
        for pos, idx, value in written:
            if pos == p:
                want[idx] = value
        root = settings.output_path
        got = read_array_data(
            root if len(positions) == 1 else f"{root}/{position.name}"
        )
        assert got.shape == want.shape
        np.testing.assert_array_equal(got, want)


@pytest.fixture
def buffered_backend(zarr_backend: str) -> str:
    """The zarr backends that use `ChunkBuffer`."""
    if zarr_backend == "acquire-zarr":
        pytest.skip("acquire-zarr doesn't use ChunkBuffer")
    return zarr_backend


@pytest.mark.parametrize(
    ("dims", "n_frames", "kwargs"),
    [
        pytest.param([T(95, 1, 10), Y, X], 95, {}, id="short-last-shard"),
        pytest.param([T(None, 1, 10), Y, X], 95, {}, id="unbounded-shard-partial"),
        pytest.param([T(None, 1, 10), Y, X], 30, {}, id="unbounded-shard-exact"),
        pytest.param([T(None, 10), Y, X], 95, {}, id="unbounded-chunk"),
        pytest.param([T(100, 1, 10), Y, X], 95, {}, id="early-stop-shard"),
        pytest.param([T(100, 10), Y, X], 95, {}, id="early-stop-chunk"),
        pytest.param([T(4, 1, 2), C(2, 1, 2), Z(10, 1, 5), Y, X], 80, {}, id="3-dims"),
        pytest.param(
            [T(3), Z(4, 1, 2), C(2, 1, 2), Y, X],
            24,
            {"storage_order": "ome"},
            id="transposed",
        ),
        pytest.param([T(20, 1, 5), P, Y, X], 40, {}, id="t-p"),
        pytest.param([P, T(20, 1, 5), Y, X], 40, {}, id="p-t"),
        pytest.param([T(30, 2, 5), Y, X], 30, {}, id="chunk-2-shard-5"),
        pytest.param(
            [
                T(6),
                Dimension(
                    name="y",
                    count=100,
                    chunk_size=32,
                    shard_size_chunks=2,
                    type="space",
                ),
                Dimension(
                    name="x",
                    count=100,
                    chunk_size=32,
                    shard_size_chunks=2,
                    type="space",
                ),
            ],
            6,
            {},
            id="frame-not-multiple-of-spatial-shard",
        ),
    ],
)
def test_all_frames_reach_disk(
    tmp_path: Path, zarr_backend: str, dims: list, n_frames: int, kwargs: dict
) -> None:
    if zarr_backend == "acquire-zarr" and n_frames == 95 and dims[0].count == 100:
        pytest.skip("acquire-zarr's final shape stops at the frames written")
    settings = _settings(tmp_path, zarr_backend, dims, **kwargs)
    _write_and_verify(settings, [n_frames])


@pytest.mark.parametrize(
    ("count", "ops"),
    [
        pytest.param(30, [5, -1, 24], id="mid-shard"),
        pytest.param(30, [10, -10, 10], id="whole-shard"),
        pytest.param(30, [9, -1, 20], id="completes-shard"),
        pytest.param(None, [5, -1, 24], id="unbounded-mid-shard"),
        pytest.param(None, [10, -10, 10], id="unbounded-whole-shard"),
        pytest.param(None, [12, -3], id="unbounded-trailing"),
    ],
)
def test_skips(
    tmp_path: Path, buffered_backend: str, count: int | None, ops: list[int]
) -> None:
    settings = _settings(tmp_path, buffered_backend, [T(count, 1, 10), Y, X])
    _write_and_verify(settings, ops)


def test_one_write_per_shard(
    tmp_path: Path, buffered_backend: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    settings = _settings(tmp_path, buffered_backend, [T(95, 1, 10), Y, X])
    with create_stream(settings) as stream:
        write = stream._backend._write
        calls = []

        def counting_write(*args: Any) -> None:
            calls.append(args)
            write(*args)

        monkeypatch.setattr(stream._backend, "_write", counting_write)
        for i in range(95):
            stream.append(_frame(i))
    assert len(calls) == 10


def test_skips_release_shards(tmp_path: Path, buffered_backend: str) -> None:
    settings = _settings(tmp_path, buffered_backend, [T(30, 1, 10), Y, X])
    with create_stream(settings) as stream:
        buffer = stream._backend._chunk_buffers[0]
        for shard in range(3):
            stream.skip(frames=1)
            for i in range(9):
                stream.append(_frame(shard * 10 + i + 1))
            assert not buffer._active_chunks


def test_acquire_zarr_has_no_buffers(tmp_path: Path) -> None:
    pytest.importorskip("acquire_zarr")
    settings = _settings(tmp_path, "acquire-zarr", [T(20, 1, 5), Y, X])
    with create_stream(settings) as stream:
        assert stream._backend._chunk_buffers is None


LIVE_CASES = [
    pytest.param([T(30, 1, 10), Y, X], 30, {}, id="shard"),
    pytest.param([T(30, 1, 100), Y, X], 30, {}, id="shard-larger-than-count"),
    pytest.param([T(30, 10), Y, X], 30, {}, id="chunk"),
    pytest.param([T(None, 1, 10), Y, X], 25, {}, id="unbounded-shard"),
    pytest.param([T(None, 10), Y, X], 25, {}, id="unbounded-chunk"),
    pytest.param([T(3), Z(10, 1, 5), Y, X], 30, {}, id="t-z-shards"),
    pytest.param(
        [T(2), Z(6, 1, 2), C(3, 1, 3), Y, X],
        36,
        {"storage_order": "ome"},
        id="transposed",
    ),
    pytest.param([T(10, 1, 5), P, Y, X], 20, {}, id="t-p"),
    pytest.param([P, T(10, 1, 5), Y, X], 20, {}, id="p-t"),
]


@pytest.mark.parametrize(("dims", "n_frames", "kwargs"), LIVE_CASES)
def test_view_reads_each_frame_right_after_append(
    tmp_path: Path, buffered_backend: str, dims: list, n_frames: int, kwargs: dict
) -> None:
    settings = _settings(tmp_path, buffered_backend, dims, **kwargs)
    with create_stream(settings) as stream:
        view = stream.view(dynamic_shape=True, strict=False)
        for i in range(n_frames):
            stream.append(_frame(i))
            np.testing.assert_array_equal(view[_key(settings, i)], _frame(i))


@pytest.mark.parametrize(("dims", "n_frames", "kwargs"), LIVE_CASES)
def test_reader_thread_sees_every_finished_frame(
    tmp_path: Path, buffered_backend: str, dims: list, n_frames: int, kwargs: dict
) -> None:
    settings = _settings(tmp_path, buffered_backend, dims, **kwargs)
    latest = -1
    n_reads = 0
    wrong: list[Any] = []
    stop = threading.Event()

    def reader(view: Any) -> None:
        nonlocal n_reads
        while not stop.is_set():
            if (i := latest) < 0:
                continue
            try:
                if not np.array_equal(view[_key(settings, i)], _frame(i)):
                    wrong.append(i)
            except Exception as e:
                wrong.append(e)
            n_reads += 1

    with create_stream(settings) as stream:
        thread = threading.Thread(
            target=reader, args=(stream.view(dynamic_shape=True, strict=False),)
        )
        thread.start()
        try:
            for i in range(n_frames):
                stream.append(_frame(i))
                latest = i
            while not n_reads:
                time.sleep(0.001)
        finally:
            stop.set()
            thread.join()
    assert not wrong


def test_chunk_stays_readable_during_flush(
    tmp_path: Path, buffered_backend: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    settings = _settings(tmp_path, buffered_backend, [T(10, 1, 5), Y, X])
    seen = []
    with create_stream(settings) as stream:
        view = stream.view(dynamic_shape=True, strict=False)
        write = stream._backend._write

        def checking_write(*args: Any) -> None:
            seen.append(np.asarray(view[4]).max())  # not on disk yet
            write(*args)

        monkeypatch.setattr(stream._backend, "_write", checking_write)
        for i in range(5):
            stream.append(_frame(i))
    assert seen == [5]


def test_tensorstore_holds_chunk_until_commit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pytest.importorskip("tensorstore")

    class PendingWrite:
        callback: Any = None

        def add_done_callback(self, callback: Any) -> None:
            self.callback = callback

        def result(self) -> None: ...

    settings = _settings(tmp_path, "tensorstore", [T(10, 1, 5), Y, X])
    with create_stream(settings) as stream:
        pending = PendingWrite()
        backend = stream._backend
        monkeypatch.setattr(
            backend, "_write", lambda *_: backend._futures.append(pending)
        )
        view = stream.view(dynamic_shape=True, strict=False)
        for i in range(5):
            stream.append(_frame(i))

        buffer = backend._chunk_buffers[0]
        assert list(buffer._flushing) == [(0,)]
        assert not buffer._active_chunks
        np.testing.assert_array_equal(view[4], _frame(4))

        pending.callback(pending)
        assert not buffer._flushing


def _random_index(rng: random.Random, size: int, kind: str) -> int | slice:
    if kind == "int":
        return rng.randrange(-size, size)
    if kind == "all":
        return slice(None)
    start = rng.randrange(size)
    return slice(start, rng.randrange(start, size) + 1, rng.choice([1, 2, 3]))


def _random_key(
    rng: random.Random, shape: tuple[int, ...], uniform_index_kind: bool
) -> tuple[int | slice, ...]:
    kinds = ("int", "int", "all", "slice", "slice")
    n_index = len(shape) - 2
    if uniform_index_kind:
        # mixing the two on transposed storage hits an unrelated `StreamView` error
        first = rng.choice(("int", "slice"))
        index_kinds = ["int" if first == "int" else rng.choice(kinds[2:])] * n_index
    else:
        index_kinds = [rng.choice(kinds) for _ in range(n_index)]
    key = [
        _random_index(rng, size, kind)
        for size, kind in zip(shape, index_kinds, strict=False)
    ]
    n_frame_keys = rng.randint(0, 2) if len(key) == n_index else 0
    for size in shape[n_index:][:n_frame_keys]:
        key.append(_random_index(rng, size, rng.choice(kinds)))
    return tuple(key)


@pytest.mark.parametrize(
    ("dims", "n_frames", "kwargs"),
    [
        pytest.param([T(20, 1, 5), Y, X], 17, {}, id="shard"),
        pytest.param([T(20, 5), Y, X], 17, {}, id="chunk"),
        pytest.param([T(20, 1, 5), C(3, 1, 3), Y, X], 50, {}, id="t-c-shards"),
        pytest.param([T(20, 1, 5), P, Y, X], 33, {}, id="t-p"),
        pytest.param(
            [T(4, 1, 2), Z(4, 1, 2), C(3, 1, 3), Y, X],
            30,
            {"storage_order": "ome"},
            id="transposed",
        ),
    ],
)
def test_random_keys_match_numpy_mid_acquisition(
    tmp_path: Path, buffered_backend: str, dims: list, n_frames: int, kwargs: dict
) -> None:
    settings = _settings(tmp_path, buffered_backend, dims, **kwargs)
    shape = tuple(d.count or 0 for d in settings.dimensions)
    want = np.zeros(shape, dtype="uint16")
    rng = random.Random(0)
    with create_stream(settings) as stream:
        view = stream.view(dynamic_shape=False, strict=False)
        for i in range(n_frames):
            frame = _textured(i)
            stream.append(frame)
            want[_key(settings, i)] = frame
        for _ in range(100):
            key = _random_key(rng, shape, uniform_index_kind=bool(kwargs))
            np.testing.assert_array_equal(view[key], want[key], err_msg=str(key))


def test_view_keeps_working_after_close(tmp_path: Path, buffered_backend: str) -> None:
    settings = _settings(tmp_path, buffered_backend, [T(20, 1, 5), Y, X])
    stream = create_stream(settings)
    view = stream.view(dynamic_shape=True, strict=False)
    want = np.zeros((20, H, W), dtype="uint16")
    for i in range(13):
        want[i] = _textured(i)
        stream.append(want[i])
    stream.close()
    for i in range(13):
        np.testing.assert_array_equal(view[i], want[i])
    np.testing.assert_array_equal(view[:], want)


UNBOUNDED_KEYS = [
    pytest.param(lambda n: slice(None), id="all"),
    pytest.param(lambda n: slice(0, n), id="to-extent"),
    pytest.param(lambda n: slice(n // 2, n), id="sub-range"),
    pytest.param(lambda n: slice(0, n, 2), id="step"),
    pytest.param(lambda n: slice(1, n, 3), id="offset-step"),
    pytest.param(lambda n: (slice(None), 2), id="all-with-frame-int"),
    pytest.param(lambda n: (slice(0, n), slice(1, 5), slice(2, 8, 2)), id="sub-region"),
]


@pytest.mark.parametrize("n_frames", [3, 10, 13, 25])
@pytest.mark.parametrize(("chunk", "shard"), [(1, 10), (10, None)])
@pytest.mark.parametrize("make_key", UNBOUNDED_KEYS)
def test_slices_on_unbounded_dimension(
    tmp_path: Path,
    buffered_backend: str,
    chunk: int,
    shard: int | None,
    n_frames: int,
    make_key: Any,
) -> None:
    settings = _settings(tmp_path, buffered_backend, [T(None, chunk, shard), Y, X])
    with create_stream(settings) as stream:
        view = stream.view(dynamic_shape=True, strict=False)
        want = np.empty((n_frames, H, W), dtype="uint16")
        for i in range(n_frames):
            want[i] = _textured(i)
            stream.append(want[i])
        key = make_key(n_frames)
        np.testing.assert_array_equal(view[key], want[key])


def test_memory_estimate_counts_shards(tmp_path: Path) -> None:
    dims = [T(10, 2, 5), Z(4, 1, 2), Y, X]
    settings = _settings(tmp_path, "tensorstore", dims)
    assert _bytes_per_chunk(settings) == 10 * 2 * H * W * 2
    assert _max_chunks(settings) == 2  # one t shard is open across 2 z shards
