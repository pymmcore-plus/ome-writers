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
            _, chunk_data = buffer.begin_flush(result)
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
    _, chunk_data = buffer.begin_flush(result)
    assert chunk_data.shape == expected_shape


def test_unlimited_dimension() -> None:
    """Test unlimited dimension grows chunks indefinitely."""
    buffer = ChunkBuffer((None, 4), (2, 2), (2, 2), np.uint16)

    completed = []
    for i in range(10):
        for j in range(4):
            if result := buffer.add_frame((i, j), np.ones((2, 2), dtype=np.uint16)):
                completed.append(result)
                buffer.begin_flush(result)

    # 10x4 frames with 2x2 chunks = 5x2 = 10 chunks
    assert len(completed) == 10


def test_trimmed_flush() -> None:
    """Test partial flushes stop at the frames added so far."""
    buffer = ChunkBuffer((None, 4), (10, 4), (2, 2), np.uint16)
    for idx in [(0, 0), (0, 1), (1, 0), (2, 1)]:
        buffer.add_frame(idx, np.ones((2, 2), dtype=np.uint16))
    assert buffer._active_chunks[(0, 0)].shape == (10, 4, 2, 2)

    start, data = buffer.begin_flush((0, 0), trim=True)
    assert start == (0, 0)
    assert data.shape == (3, 2, 2, 2)
    assert not buffer._active_chunks


def test_trim_is_relative_to_chunk_start() -> None:
    buffer = ChunkBuffer((None,), (10,), (2, 2), np.uint16)
    for i in range(10):
        buffer.add_frame((i,), np.ones((2, 2), dtype=np.uint16))
    buffer.begin_flush((0,))
    for i in range(10, 13):
        buffer.add_frame((i,), np.ones((2, 2), dtype=np.uint16))

    start, data = buffer.begin_flush((1,), trim=True)
    assert start == (10,)
    assert data.shape == (3, 2, 2)


def test_flush_keeps_chunk_in_snapshot_until_end_flush() -> None:
    buffer = ChunkBuffer((2,), (2,), (2, 2), np.uint16)
    assert buffer.snapshot() == []
    buffer.add_frame((0,), np.ones((2, 2), dtype=np.uint16))
    assert [coords for coords, _ in buffer.snapshot()] == [(0,)]

    buffer.begin_flush((0,))
    assert not buffer._active_chunks
    assert [coords for coords, _ in buffer.snapshot()] == [(0,)]

    buffer.end_flush((0,))
    assert buffer.snapshot() == []


def test_skip_completes_chunk() -> None:
    """Test a skip that fills the rest of an active chunk asks for a flush."""
    buffer = ChunkBuffer((4,), (2,), (2, 2), np.uint16)
    frame = np.ones((2, 2), dtype=np.uint16)
    assert buffer.add_frame((0,), frame) is None
    assert buffer.skip_frame((1,)) == (0,)
    start, data = buffer.begin_flush((0,))
    assert start == (0,)
    assert data.shape == (2, 2, 2)


def test_skip_before_frames_counts_toward_completion() -> None:
    """Test skips recorded before a chunk is allocated aren't forgotten."""
    buffer = ChunkBuffer((4,), (2,), (2, 2), np.uint16)
    assert buffer.skip_frame((0,)) is None
    assert not buffer._active_chunks
    assert buffer.add_frame((1,), np.ones((2, 2), dtype=np.uint16)) == (0,)


def test_chunk_of_only_skips_needs_no_flush() -> None:
    buffer = ChunkBuffer((4,), (2,), (2, 2), np.uint16)
    assert buffer.skip_frame((0,)) is None
    assert buffer.skip_frame((1,)) is None
    assert not buffer._active_chunks
    assert not buffer._filled_count
