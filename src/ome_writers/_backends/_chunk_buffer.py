"""Chunk buffer for managing in-memory data before flushing to storage."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from ome_writers._backends._backend import ArrayLike


class ChunkBuffer:
    """Manages in-memory chunk buffers for a single position's array.

    Most backends will immediately flush incoming frames to storage.  However, if the
    user has requested chunking or sharding along any non-frame dimensions (e.g. 3D ZYX
    chunks), then frames must be buffered until a full chunk (or shard, if sharded) is
    available for writing. Otherwise, backends like zarr-python and tensorstore will be
    forced to read-modify-write chunks or shards on every frame write, which is very
    inefficient.

    This class serves to buffer incoming frames until a full chunk or shard is
    available, at which point it can be flushed to storage in one operation.  It
    supports multiple active chunks to handle all acquisition patterns, including
    transposed storage order and non-contiguous chunked dimensions.

    Reads can run on other threads while frames are written, and must find every frame
    either here or on disk (see `_BufferedArray`).  So a chunk moves from
    `_active_chunks` to `_flushing` with `begin_flush`, and only leaves `_flushing` with
    `end_flush`, once its write is visible on disk.

    Parameters
    ----------
    index_shape : tuple[int | None, ...]
        Shape of index dimensions (excluding frame dimensions).
        None values indicate unbounded dimensions (in practice, only the first
        dimension can be unbounded).
    chunk_shape : tuple[int, ...]
        Number of frames written at once along each index dimension: the chunk size, or
        the shard size for sharded dimensions.
    frame_shape : tuple[int, int]
        Shape of frame dimensions (Y, X).
    dtype : str | np.dtype[Any]
        Data type for the array.
    """

    __slots__ = (
        "_active_chunks",
        "_extent",
        "_filled_count",
        "_flushing",
        "chunk_shape",
        "dtype",
        "frame_shape",
        "index_shape",
    )

    def __init__(
        self,
        index_shape: tuple[int | None, ...],
        chunk_shape: tuple[int, ...],
        frame_shape: tuple[int, int],
        dtype: str | np.dtype[Any],
    ) -> None:
        self.index_shape = index_shape
        self.chunk_shape = chunk_shape
        self.frame_shape = frame_shape
        self.dtype = dtype

        self._active_chunks: dict[tuple[int, ...], np.ndarray] = {}
        self._filled_count: dict[tuple[int, ...], int] = {}
        self._flushing: dict[tuple[int, ...], np.ndarray] = {}
        self._extent = [0] * len(index_shape)  # high-water mark of added frames

    def add_frame(
        self,
        storage_index: tuple[int, ...],
        frame: np.ndarray,
    ) -> tuple[int, ...] | None:
        """Add frame to buffer. Returns chunk_coords if chunk is complete."""
        self._extent = [
            max(e, i + 1) for e, i in zip(self._extent, storage_index, strict=False)
        ]

        # Single divmod pass for both chunk coords and position within chunk
        chunk_coords, frame_within_chunk = [], []
        for idx, cs in zip(storage_index, self.chunk_shape, strict=False):
            cc, fc = divmod(idx, cs)
            chunk_coords.append(cc)
            frame_within_chunk.append(fc)
        chunk_coords = tuple(chunk_coords)

        if chunk_coords not in self._active_chunks:
            self._allocate_chunk(chunk_coords)

        self._active_chunks[chunk_coords][tuple(frame_within_chunk)] = frame
        self._filled_count[chunk_coords] += 1

        if self._is_chunk_complete(chunk_coords):
            return chunk_coords
        return None

    def skip_frame(self, storage_index: tuple[int, ...]) -> tuple[int, ...] | None:
        """Count a skipped frame. Returns chunk_coords if that chunk needs flushing."""
        coords = tuple(
            i // cs for i, cs in zip(storage_index, self.chunk_shape, strict=False)
        )
        self._filled_count[coords] = self._filled_count.get(coords, 0) + 1
        if not self._is_chunk_complete(coords):
            return None
        if coords in self._active_chunks:
            return coords
        del self._filled_count[coords]  # every frame skipped: nothing to write
        return None

    def begin_flush(
        self, chunk_coords: tuple[int, ...], *, trim: bool = False
    ) -> tuple[tuple[int, ...], np.ndarray]:
        """Move chunk to `_flushing`, and return its storage location and data.

        If `trim`, the returned data stops at the frames added so far.
        """
        data = self._active_chunks[chunk_coords]
        self._flushing[chunk_coords] = data  # must happen before removal from active
        del self._active_chunks[chunk_coords]
        self._filled_count.pop(chunk_coords, None)
        start = tuple(
            cc * cs for cc, cs in zip(chunk_coords, self.chunk_shape, strict=False)
        )
        if trim:
            data = data[
                tuple(
                    slice(0, e - s) for e, s in zip(self._extent, start, strict=False)
                )
            ]
        return start, data

    def end_flush(self, chunk_coords: tuple[int, ...]) -> None:
        """Forget a flushed chunk. Call once its write is visible on disk."""
        self._flushing.pop(chunk_coords, None)

    def snapshot(self) -> list[tuple[tuple[int, ...], np.ndarray]]:
        """Return all chunks that may not be readable from disk yet."""
        # Order matters: chunks move from active to flushing, so check active first.
        active = list(self._active_chunks.items())
        return active + list(self._flushing.items())

    # ------------------------

    def _allocate_chunk(self, chunk_coords: tuple[int, ...]) -> None:
        """Allocate buffer for a new chunk."""
        actual_chunk_shape = self._get_actual_chunk_shape(chunk_coords)
        full_shape = actual_chunk_shape + self.frame_shape
        self._active_chunks[chunk_coords] = np.zeros(
            full_shape, dtype=np.dtype(self.dtype)
        )
        self._filled_count.setdefault(chunk_coords, 0)  # keep already-skipped frames

    def _get_actual_chunk_shape(self, chunk_coords: tuple[int, ...]) -> tuple[int, ...]:
        """Compute actual chunk shape (handles partial chunks at boundaries)."""
        actual_shape = []
        for i, (cc, cs) in enumerate(zip(chunk_coords, self.chunk_shape, strict=False)):
            dim_size = self.index_shape[i]
            if dim_size is None:
                actual_shape.append(cs)
            else:
                start = cc * cs
                actual_shape.append(min(start + cs, dim_size) - start)
        return tuple(actual_shape)

    def _is_chunk_complete(self, chunk_coords: tuple[int, ...]) -> bool:
        """Check if all expected frames in chunk have been written."""
        actual_shape = self._get_actual_chunk_shape(chunk_coords)
        expected_count = math.prod(actual_shape)
        return self._filled_count[chunk_coords] == expected_count


class _BufferedArray:
    """Array whose reads include frames still waiting in a `ChunkBuffer`."""

    __slots__ = ("_array", "_buffer")

    def __init__(self, array: ArrayLike, buffer: ChunkBuffer) -> None:
        self._array = array
        self._buffer = buffer

    @property
    def shape(self) -> tuple[int, ...]:
        return self._array.shape

    @property
    def dtype(self) -> Any:
        return self._array.dtype

    def __getitem__(self, key: Any) -> np.ndarray:
        # Snapshot BEFORE reading disk: a chunk missing from the snapshot was fully
        # written before the disk read began, so that read is guaranteed to see it.
        chunks = self._buffer.snapshot()
        result = np.asarray(self._array[key])
        if not chunks:
            return result

        key = key if isinstance(key, tuple) else (key,)
        key = key + (slice(None),) * (len(self.shape) - len(key))
        copied = False
        for coords, data in chunks:
            start = tuple(
                c * s for c, s in zip(coords, self._buffer.chunk_shape, strict=False)
            )
            placed = _selection(key, self.shape, result.shape, start, data.shape)
            if placed is None:
                continue
            if not copied:
                result = np.array(result)
                copied = True
            out, src, drop = placed
            values = data[(Ellipsis, *key[len(start) :])][np.ix_(*src)]
            values = np.squeeze(values, axis=drop) if drop else values
            result[np.ix_(*out) if out else ()] = values
        return result


def _selection(
    key: tuple[Any, ...],
    shape: tuple[int, ...],
    read_shape: tuple[int, ...],
    start: tuple[int, ...],
    chunk_shape: tuple[int, ...],
) -> tuple[list[np.ndarray], list[np.ndarray], tuple[int, ...]] | None:
    """Map index-dim selections onto a chunk, or return None if they don't overlap."""
    out, src, drop = [], [], []
    for axis, s in enumerate(start):
        k, size, n = key[axis], shape[axis], chunk_shape[axis]
        if isinstance(k, slice):
            # `size` can lag the read (a tensorstore handle's shape is stale after a
            # resize) and unbounded chunks extend past the array, so stop at the length
            # that was actually read.  Only negative starts need the true `size`.
            bound = size if (k.start or 0) < 0 else max(size, s + n)
            positions = np.arange(*k.indices(bound))[: read_shape[len(out)]]
            mask = (positions >= s) & (positions < s + n)
            if not mask.any():
                return None
            out.append(np.nonzero(mask)[0])
            src.append(positions[mask] - s)
        else:
            k = k + size if k < 0 else k
            if not s <= k < s + n:
                return None
            src.append(np.array([k - s]))
            drop.append(axis)
    return out, src, tuple(drop)
