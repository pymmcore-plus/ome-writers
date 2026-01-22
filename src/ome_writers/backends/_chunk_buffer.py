"""Yaozarrs-based backends for OME-Zarr v0.5."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import numpy as np


class ChunkBuffer:
    """Manages in-memory chunk buffers for a single position's array.

    Supports multiple active chunks to handle all acquisition patterns,
    including transposed storage order and non-contiguous chunked dimensions.
    """

    def __init__(
        self,
        index_shape: tuple[int | None, ...],  # Shape of index dims only
        chunk_shape: tuple[int, ...],  # Chunk sizes for index dims
        frame_shape: tuple[int, int],  # Y, X dimensions
        dtype: str | np.dtype[Any],
    ) -> None:
        """Initialize chunk buffer.

        Parameters
        ----------
        index_shape : tuple[int | None, ...]
            Shape of index dimensions (excluding frame dimensions).
            None values indicate unlimited dimensions.
        chunk_shape : tuple[int, ...]
            Chunk sizes for each index dimension.
        frame_shape : tuple[int, int]
            Shape of frame dimensions (Y, X).
        dtype : str | np.dtype[Any]
            Data type for the array.
        """
        self.index_shape = list(index_shape)  # Make mutable for updates
        self.chunk_shape = chunk_shape
        self.frame_shape = frame_shape
        self.dtype = dtype

        # Active chunk buffers: {chunk_coords: ndarray}
        # chunk_coords = tuple of chunk indices in storage space
        # buffer shape = chunk_shape + frame_shape
        self._active_chunks: dict[tuple[int, ...], np.ndarray] = {}

        # Track filled frames per chunk: {chunk_coords: set of frame_indices}
        # frame_indices are positions within the chunk (not global)
        self._filled_frames: dict[tuple[int, ...], set[tuple[int, ...]]] = {}

    def add_frame(
        self,
        storage_index: tuple[int, ...],  # Index in storage space (no frame dims)
        frame: np.ndarray,  # 2D frame (Y, X)
    ) -> tuple[int, ...] | None:
        """Add frame to buffer. Returns chunk_coords if chunk is complete."""
        chunk_coords = self._get_chunk_coords(storage_index)
        frame_within_chunk = self._get_frame_within_chunk(storage_index)

        # Initialize chunk buffer if needed
        if chunk_coords not in self._active_chunks:
            self._allocate_chunk(chunk_coords)

        # Store frame in chunk buffer
        buffer = self._active_chunks[chunk_coords]
        buffer[frame_within_chunk] = frame
        self._filled_frames[chunk_coords].add(frame_within_chunk)

        # Check if chunk is complete
        if self._is_chunk_complete(chunk_coords):
            return chunk_coords
        return None

    def _get_chunk_coords(self, storage_index: tuple[int, ...]) -> tuple[int, ...]:
        """Convert storage index to chunk coordinates."""
        return tuple(
            idx // chunk_size
            for idx, chunk_size in zip(storage_index, self.chunk_shape, strict=False)
        )

    def _get_frame_within_chunk(
        self, storage_index: tuple[int, ...]
    ) -> tuple[int, ...]:
        """Get frame's position within its chunk."""
        return tuple(
            idx % chunk_size
            for idx, chunk_size in zip(storage_index, self.chunk_shape, strict=False)
        )

    def _allocate_chunk(self, chunk_coords: tuple[int, ...]) -> None:
        """Allocate buffer for a new chunk."""
        import numpy as np

        # Determine actual chunk shape (may be partial at array boundaries)
        actual_chunk_shape = self._get_actual_chunk_shape(chunk_coords)
        full_shape = actual_chunk_shape + self.frame_shape

        self._active_chunks[chunk_coords] = np.zeros(full_shape, dtype=self.dtype)
        self._filled_frames[chunk_coords] = set()

    def _get_actual_chunk_shape(self, chunk_coords: tuple[int, ...]) -> tuple[int, ...]:
        """Compute actual chunk shape (handles partial chunks at boundaries)."""
        actual_shape = []
        for i, (cc, cs) in enumerate(zip(chunk_coords, self.chunk_shape, strict=False)):
            start = cc * cs
            if self.index_shape[i] is None:
                # Unlimited dimension - use full chunk size
                actual_shape.append(cs)
            else:
                # Limited dimension - may be partial at boundary
                end = min(start + cs, self.index_shape[i])
                actual_shape.append(end - start)
        return tuple(actual_shape)

    def _is_chunk_complete(self, chunk_coords: tuple[int, ...]) -> bool:
        """Check if all expected frames in chunk have been written."""
        import numpy as np

        actual_shape = self._get_actual_chunk_shape(chunk_coords)
        expected_count = int(np.prod(actual_shape))
        return len(self._filled_frames[chunk_coords]) == expected_count

    def get_chunk_for_flush(
        self, chunk_coords: tuple[int, ...]
    ) -> tuple[tuple[int, ...], np.ndarray]:
        """Extract chunk buffer and compute storage location for writing."""
        buffer = self._active_chunks.pop(chunk_coords)
        self._filled_frames.pop(chunk_coords)

        # Calculate starting storage index for this chunk
        storage_start = tuple(
            cc * cs for cc, cs in zip(chunk_coords, self.chunk_shape, strict=False)
        )

        return storage_start, buffer

    def flush_all_partial(self) -> list[tuple[tuple[int, ...], np.ndarray]]:
        """Flush all remaining chunks during finalize (may be incomplete)."""
        chunks_to_flush = []
        for chunk_coords in list(self._active_chunks.keys()):
            storage_start, buffer = self.get_chunk_for_flush(chunk_coords)
            chunks_to_flush.append((storage_start, buffer))
        return chunks_to_flush

    def estimate_memory_usage(self) -> int:
        """Estimate current memory usage in bytes."""
        return sum(buf.nbytes for buf in self._active_chunks.values())
