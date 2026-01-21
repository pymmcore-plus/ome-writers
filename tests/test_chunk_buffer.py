"""Unit tests for ChunkBuffer class."""

from __future__ import annotations

import numpy as np

from ome_writers.backends._yaozarrs import ChunkBuffer


def test_chunk_buffer_single_chunk() -> None:
    """Test buffering frames into a single chunk."""
    # Single chunk: 2x2x2 frames with 2x2 frame dims
    index_shape = (2, 2, 2)
    chunk_shape = (2, 2, 2)
    frame_shape = (2, 2)
    dtype = np.uint16

    buffer = ChunkBuffer(index_shape, chunk_shape, frame_shape, dtype)

    # Add all frames to fill one chunk
    frames_added = 0
    chunk_coords = None
    for z in range(2):
        for c in range(2):
            for t in range(2):
                frame = np.ones(frame_shape, dtype=dtype) * (z * 4 + c * 2 + t)
                result = buffer.add_frame((z, c, t), frame)
                frames_added += 1

                # Should return chunk coords only when complete
                if frames_added == 8:
                    assert result == (0, 0, 0)
                    chunk_coords = result
                else:
                    assert result is None

    # Extract the chunk
    assert chunk_coords is not None
    storage_start, chunk_data = buffer.get_chunk_for_flush(chunk_coords)

    assert storage_start == (0, 0, 0)
    assert chunk_data.shape == (2, 2, 2, 2, 2)
    assert chunk_data.dtype == dtype

    # Verify data is correct
    for z in range(2):
        for c in range(2):
            for t in range(2):
                expected = z * 4 + c * 2 + t
                assert np.all(chunk_data[z, c, t] == expected)


def test_chunk_buffer_multiple_chunks() -> None:
    """Test multiple simultaneous active chunks."""
    # 4x4 index space with 2x2 chunks -> 2x2 = 4 chunks
    index_shape = (4, 4)
    chunk_shape = (2, 2)
    frame_shape = (2, 2)
    dtype = np.uint16

    buffer = ChunkBuffer(index_shape, chunk_shape, frame_shape, dtype)

    # Add frames in interleaved order to activate multiple chunks
    completed_chunks = []

    # Interleave: (0,0), (2,2), (1,1), (3,3) to activate all 4 chunks
    indices = [(0, 0), (2, 2), (1, 1), (3, 3), (0, 1), (2, 3), (1, 0), (3, 2)]

    for idx in indices:
        frame = np.ones(frame_shape, dtype=dtype) * (idx[0] * 4 + idx[1])
        result = buffer.add_frame(idx, frame)
        if result is not None:
            completed_chunks.append(result)

    # Should have 2 completed chunks so far (4 frames each)
    assert len(completed_chunks) == 2

    # Complete remaining frames
    remaining = [(0, 2), (0, 3), (2, 0), (2, 1), (1, 2), (1, 3), (3, 0), (3, 1)]
    for idx in remaining:
        frame = np.ones(frame_shape, dtype=dtype) * (idx[0] * 4 + idx[1])
        result = buffer.add_frame(idx, frame)
        if result is not None:
            completed_chunks.append(result)

    # All 4 chunks should be complete now
    assert len(completed_chunks) == 4
    assert set(completed_chunks) == {(0, 0), (0, 1), (1, 0), (1, 1)}


def test_chunk_buffer_partial_chunk() -> None:
    """Test handling of partial chunks at array boundaries."""
    # 5x5 index space with 2x2 chunks -> last chunk is 1x1
    index_shape = (5, 5)
    chunk_shape = (2, 2)
    frame_shape = (2, 2)
    dtype = np.uint16

    buffer = ChunkBuffer(index_shape, chunk_shape, frame_shape, dtype)

    # Fill the partial chunk at (4, 4) -> chunk coords (2, 2)
    # This chunk should only have 1 frame
    frame = np.ones(frame_shape, dtype=dtype) * 42
    result = buffer.add_frame((4, 4), frame)

    # Should complete immediately since it only expects 1 frame
    assert result == (2, 2)

    # Extract and verify
    storage_start, chunk_data = buffer.get_chunk_for_flush(result)
    assert storage_start == (4, 4)
    assert chunk_data.shape == (1, 1, 2, 2)  # Partial chunk
    assert np.all(chunk_data == 42)


def test_chunk_buffer_partial_chunk_edge() -> None:
    """Test partial chunk at edge when some dims are full."""
    # 6x5 index space with 2x2 chunks -> last column has 6x1 partial chunks
    index_shape = (6, 5)
    chunk_shape = (2, 2)
    frame_shape = (2, 2)
    dtype = np.uint16

    buffer = ChunkBuffer(index_shape, chunk_shape, frame_shape, dtype)

    # Fill a partial chunk at column edge: (0, 4) and (1, 4) -> chunk coords (0, 2)
    # This chunk should have 2 frames (full in first dim, partial in second)
    frame1 = np.ones(frame_shape, dtype=dtype) * 1
    frame2 = np.ones(frame_shape, dtype=dtype) * 2

    result1 = buffer.add_frame((0, 4), frame1)
    assert result1 is None  # Not complete yet

    result2 = buffer.add_frame((1, 4), frame2)
    assert result2 == (0, 2)  # Complete now

    # Extract and verify
    storage_start, chunk_data = buffer.get_chunk_for_flush(result2)
    assert storage_start == (0, 4)
    assert chunk_data.shape == (2, 1, 2, 2)  # 2x1 partial chunk
    assert np.all(chunk_data[0, 0] == 1)
    assert np.all(chunk_data[1, 0] == 2)


def test_chunk_buffer_memory_estimate() -> None:
    """Test memory usage estimation."""
    index_shape = (4, 4)
    chunk_shape = (2, 2)
    frame_shape = (100, 100)
    dtype = np.uint16

    buffer = ChunkBuffer(index_shape, chunk_shape, frame_shape, dtype)

    # No active chunks initially
    assert buffer.estimate_memory_usage() == 0

    # Add one frame to activate one chunk
    frame = np.zeros(frame_shape, dtype=dtype)
    buffer.add_frame((0, 0), frame)

    # One chunk should be active now
    expected_bytes = 2 * 2 * 100 * 100 * 2  # chunk_shape * frame_shape * dtype size
    assert buffer.estimate_memory_usage() == expected_bytes

    # Add frame to a different chunk
    buffer.add_frame((2, 2), frame)

    # Two chunks should be active now
    assert buffer.estimate_memory_usage() == expected_bytes * 2


def test_chunk_buffer_unlimited_dimension() -> None:
    """Test buffering with unlimited dimension."""
    # First dimension is unlimited (None)
    index_shape = (None, 4)
    chunk_shape = (2, 2)
    frame_shape = (2, 2)
    dtype = np.uint16

    buffer = ChunkBuffer(index_shape, chunk_shape, frame_shape, dtype)

    # Add frames beyond initial size
    # Need to fill complete chunks: each chunk is 2x2 in index space
    completed_chunks = []
    frame_num = 0
    for i in range(10):
        for j in range(2):
            frame = np.ones(frame_shape, dtype=dtype) * frame_num
            result = buffer.add_frame((i, j), frame)
            frame_num += 1
            if result is not None:
                completed_chunks.append(result)
                # Extract and discard the chunk
                buffer.get_chunk_for_flush(result)

    # Update shape after resize
    buffer.index_shape[0] = 10

    # Should have completed 5 chunks (each chunk is 2x2 frames)
    # i=0,1 j=0,1 -> chunk (0,0)
    # i=2,3 j=0,1 -> chunk (1,0)
    # i=4,5 j=0,1 -> chunk (2,0)
    # i=6,7 j=0,1 -> chunk (3,0)
    # i=8,9 j=0,1 -> chunk (4,0)
    assert len(completed_chunks) == 5


def test_chunk_buffer_flush_all_partial() -> None:
    """Test flushing all partial chunks during finalize."""
    index_shape = (6, 6)
    chunk_shape = (2, 2)
    frame_shape = (2, 2)
    dtype = np.uint16

    buffer = ChunkBuffer(index_shape, chunk_shape, frame_shape, dtype)

    # Add some frames in different chunks (one frame each, so none complete)
    # (0, 0) -> chunk (0, 0)
    # (2, 2) -> chunk (1, 1)
    # (0, 4) -> chunk (0, 2)
    buffer.add_frame((0, 0), np.ones(frame_shape, dtype=dtype))
    buffer.add_frame((2, 2), np.ones(frame_shape, dtype=dtype) * 2)
    buffer.add_frame((0, 4), np.ones(frame_shape, dtype=dtype) * 3)

    # Should have 3 active partial chunks
    assert len(buffer._active_chunks) == 3

    # Flush all partial chunks
    chunks = buffer.flush_all_partial()

    # Should return all 3 chunks
    assert len(chunks) == 3

    # Verify storage starts
    storage_starts = [start for start, _ in chunks]
    assert set(storage_starts) == {(0, 0), (2, 2), (0, 4)}

    # Buffers should be cleared
    assert len(buffer._active_chunks) == 0


def test_chunk_buffer_3d_chunking() -> None:
    """Test realistic 3D chunking scenario from benchmark."""
    # Simulate z=128, chunk_size=16 with 1024x1024 frames
    # (using smaller frames for test speed)
    index_shape = (128,)
    chunk_shape = (16,)
    frame_shape = (32, 32)  # Smaller for testing
    dtype = np.uint16

    buffer = ChunkBuffer(index_shape, chunk_shape, frame_shape, dtype)

    completed_chunks = []

    # Write all 128 frames
    for z in range(128):
        frame = np.ones(frame_shape, dtype=dtype) * z
        result = buffer.add_frame((z,), frame)
        if result is not None:
            completed_chunks.append(result)
            # Extract and discard the chunk
            buffer.get_chunk_for_flush(result)

    # Should have completed 8 chunks (128 / 16 = 8)
    assert len(completed_chunks) == 8
    assert completed_chunks == [
        (0,),
        (1,),
        (2,),
        (3,),
        (4,),
        (5,),
        (6,),
        (7,),
    ]

    # No partial chunks remaining (all were flushed)
    assert len(buffer._active_chunks) == 0


def test_chunk_buffer_order_independence() -> None:
    """Test that chunk completion works regardless of frame order."""
    index_shape = (4, 4)
    chunk_shape = (2, 2)
    frame_shape = (2, 2)
    dtype = np.uint16

    buffer = ChunkBuffer(index_shape, chunk_shape, frame_shape, dtype)

    # Fill first chunk in reverse order
    indices = [(1, 1), (1, 0), (0, 1), (0, 0)]

    for i, idx in enumerate(indices):
        frame = np.ones(frame_shape, dtype=dtype) * i
        result = buffer.add_frame(idx, frame)

        if i < 3:
            assert result is None
        else:
            assert result == (0, 0)  # Chunk complete

    # Verify chunk data
    storage_start, chunk_data = buffer.get_chunk_for_flush((0, 0))
    assert storage_start == (0, 0)
    assert chunk_data.shape == (2, 2, 2, 2)

    # Verify frame values are in correct positions
    assert np.all(chunk_data[0, 0] == 3)  # Last frame added
    assert np.all(chunk_data[0, 1] == 2)
    assert np.all(chunk_data[1, 0] == 1)
    assert np.all(chunk_data[1, 1] == 0)  # First frame added
