"""Unit tests for ChunkBuffer class."""

from __future__ import annotations

import numpy as np
import pytest

from ome_writers._backends._chunk_buffer import ChunkBuffer


@pytest.mark.parametrize(
    ("index_shape", "chunk_shape", "n_frames", "expected_chunks"),
    [
        # Single 2x2x2 chunk
        ((2, 2, 2), (2, 2, 2), 8, [(0, 0, 0)]),
        # Four 2x2 chunks in 4x4 space
        ((4, 4), (2, 2), 16, [(0, 0), (0, 1), (1, 0), (1, 1)]),
        # Eight 16-frame chunks in 128-frame 1D array
        ((128,), (16,), 128, [(i,) for i in range(8)]),
    ],
)
def test_complete_chunks(
    index_shape: tuple[int, ...],
    chunk_shape: tuple[int, ...],
    n_frames: int,
    expected_chunks: list[tuple[int, ...]],
) -> None:
    """Test filling complete chunks returns correct chunk coordinates."""
    buffer = ChunkBuffer(index_shape, chunk_shape, (2, 2), np.uint16)
    completed = []

    # Generate indices for all frames
    indices = [
        tuple(
            (i // np.prod(index_shape[j + 1 :], dtype=int)) % index_shape[j]
            for j in range(len(index_shape))
        )
        for i in range(n_frames)
    ]

    for idx in indices:
        frame = np.ones((2, 2), dtype=np.uint16) * sum(idx)
        if result := buffer.add_frame(idx, frame):
            completed.append(result)
            _, chunk_data = buffer.get_chunk_for_flush(result)
            assert chunk_data.dtype == np.uint16
            assert chunk_data.shape[: len(chunk_shape)] == chunk_shape

    assert set(completed) == set(expected_chunks)


@pytest.mark.parametrize(
    ("index_shape", "chunk_shape", "indices", "expected_coords", "expected_shape"),
    [
        # Corner partial (1x1)
        ((5, 5), (2, 2), [(4, 4)], (2, 2), (1, 1, 2, 2)),
        # Edge partial (2x1)
        ((6, 5), (2, 2), [(0, 4), (1, 4)], (0, 2), (2, 1, 2, 2)),
        # Row partial (1x2)
        ((5, 6), (2, 2), [(4, 0), (4, 1)], (2, 0), (1, 2, 2, 2)),
    ],
)
def test_partial_chunks(
    index_shape: tuple[int, ...],
    chunk_shape: tuple[int, ...],
    indices: list[tuple[int, ...]],
    expected_coords: tuple[int, ...],
    expected_shape: tuple[int, ...],
) -> None:
    """Test partial chunks at array boundaries have correct shapes."""
    buffer = ChunkBuffer(index_shape, chunk_shape, (2, 2), np.uint16)

    result = None
    for idx in indices:
        result = buffer.add_frame(idx, np.ones((2, 2), dtype=np.uint16))

    assert result == expected_coords
    _, chunk_data = buffer.get_chunk_for_flush(result)
    assert chunk_data.shape == expected_shape


def test_unlimited_dimension() -> None:
    """Test unlimited dimension grows chunks indefinitely."""
    buffer = ChunkBuffer((None, 4), (2, 2), (2, 2), np.uint16)

    completed = []
    for i in range(10):
        for j in range(4):
            if result := buffer.add_frame((i, j), np.ones((2, 2), dtype=np.uint16)):
                completed.append(result)
                buffer.get_chunk_for_flush(result)

    # 10x4 frames with 2x2 chunks = 5x2 = 10 chunks
    assert len(completed) == 10


def test_flush_all_partial() -> None:
    """Test flushing incomplete chunks during finalize."""
    buffer = ChunkBuffer((6, 6), (2, 2), (2, 2), np.uint16)

    # Add one frame to three different chunks
    for i, pos in enumerate([(0, 0), (2, 2), (0, 4)]):
        buffer.add_frame(pos, np.ones((2, 2), dtype=np.uint16) * i)

    assert len(buffer._active_chunks) == 3

    chunks = buffer.flush_all_partial()
    assert len(chunks) == 3
    assert {start for start, _ in chunks} == {(0, 0), (2, 2), (0, 4)}
    assert len(buffer._active_chunks) == 0
