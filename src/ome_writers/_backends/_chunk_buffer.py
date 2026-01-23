"""Chunk buffer for managing in-memory data before flushing to storage."""

from __future__ import annotations

import math
from typing import Any

import numpy as np


class ChunkBuffer:
    """Manages in-memory chunk buffers for a single position's array.

    Most backends will immediately flush incoming frames to storage.  However, if the
    user has requested chunking along any non-frame dimensions (e.g. 3D ZYX chunks),
    then frames must be buffered until a full chunk is available for writing. Otherwise,
    backends like zarr-python and tensorstore will be forced to read-modify-write chunks
    on every frame write, which is very inefficient.

    This class serves to buffer incoming frames until a full chunk is available, at
    which point the chunk can be flushed to storage in one operation.  It supports
    multiple active chunks to handle all acquisition patterns, including transposed
    storage order and non-contiguous chunked dimensions.

    Parameters
    ----------
    index_shape : tuple[int | None, ...]
        Shape of index dimensions (excluding frame dimensions).
        None values indicate unbounded dimensions (in practice, only the first
        dimension can be unbounded).
    chunk_shape : tuple[int, ...]
        Chunk sizes for each index dimension.
    frame_shape : tuple[int, int]
        Shape of frame dimensions (Y, X).
    dtype : str | np.dtype[Any]
        Data type for the array.
    """

    __slots__ = (
        "_active_chunks",
        "_filled_count",
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
        self.index_shape = list(index_shape)  # Mutable for resize updates
        self.chunk_shape = chunk_shape
        self.frame_shape = frame_shape
        self.dtype = dtype

        self._active_chunks: dict[tuple[int, ...], np.ndarray] = {}
        self._filled_count: dict[tuple[int, ...], int] = {}

    def add_frame(
        self,
        storage_index: tuple[int, ...],
        frame: np.ndarray,
    ) -> tuple[int, ...] | None:
        """Add frame to buffer. Returns chunk_coords if chunk is complete."""
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

    def get_chunk_for_flush(
        self, chunk_coords: tuple[int, ...]
    ) -> tuple[tuple[int, ...], np.ndarray]:
        """Extract chunk buffer and compute storage location for writing."""
        buffer = self._active_chunks.pop(chunk_coords)
        self._filled_count.pop(chunk_coords)
        storage_start = tuple(
            cc * cs for cc, cs in zip(chunk_coords, self.chunk_shape, strict=False)
        )
        return storage_start, buffer

    def flush_all_partial(self) -> list[tuple[tuple[int, ...], np.ndarray]]:
        """Flush all remaining chunks (may be incomplete)."""
        return [
            self.get_chunk_for_flush(coords)
            for coords in list(self._active_chunks.keys())
        ]

    # ------------------------

    def _allocate_chunk(self, chunk_coords: tuple[int, ...]) -> None:
        """Allocate buffer for a new chunk."""
        actual_chunk_shape = self._get_actual_chunk_shape(chunk_coords)
        full_shape = actual_chunk_shape + self.frame_shape
        self._active_chunks[chunk_coords] = np.zeros(full_shape, dtype=self.dtype)
        self._filled_count[chunk_coords] = 0

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
