"""Tests for coordinate tracking system."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from ome_writers import AcquisitionSettings, Dimension, create_stream
from ome_writers._stream import OMEStream, get_format_for_backend

if TYPE_CHECKING:
    from pathlib import Path

    from ome_writers._coord_tracker import CoordUpdate


# Number of barrier tasks to submit when waiting for callbacks (for CI consistency)
BARRIERS = 20


def _wait_for_pending_callbacks(stream: OMEStream, timeout: float = 1.0) -> None:
    """Wait for all pending async callbacks to complete (for testing).

    Submits barrier tasks serially to ensure all prior work completes tests.
    """
    if executor := stream._callback_executor:
        for _ in range(BARRIERS):
            executor.submit(lambda: None).result(timeout=timeout)


def test_coord_events(tmp_path: Path, first_backend: str) -> None:
    """Test coordinate tracking event system."""
    settings = AcquisitionSettings(
        root_path=tmp_path / "test",
        dimensions=[
            Dimension(name="t", count=3, type="time"),
            Dimension(name="c", count=2, type="channel"),
            Dimension(name="y", count=16, type="space"),
            Dimension(name="x", count=16, type="space"),
        ],
        dtype="uint16",
        format=first_backend,
        overwrite=True,
    )

    frame = np.zeros((16, 16), dtype=np.uint16)
    expanded_history: list[CoordUpdate] = []
    changed_history: list[CoordUpdate] = []

    def track_expanded(update: CoordUpdate) -> None:
        expanded_history.append(update)

    def track_changed(update: CoordUpdate) -> None:
        changed_history.append(update)

    with create_stream(settings) as stream:
        # No tracker created yet
        assert stream._coord_tracker is None
        assert stream._event_handlers == {}

        # Register handlers - should create tracker
        stream.on("coords_expanded", track_expanded)
        stream.on("coords_changed", track_changed)
        assert stream._coord_tracker is not None
        assert len(stream._event_handlers["coords_expanded"]) == 1
        assert len(stream._event_handlers["coords_changed"]) == 1

        # Write first frame (t=0, c=0) - high water mark
        stream.append(frame)
        _wait_for_pending_callbacks(stream)
        assert len(expanded_history) == 1
        assert len(changed_history) == 1
        assert expanded_history[0].max_coords["t"] == range(1)
        assert expanded_history[0].max_coords["c"] == range(1)
        assert expanded_history[0].current_indices["t"] == 0
        assert expanded_history[0].current_indices["c"] == 0
        assert expanded_history[0].is_high_water_mark

        # Write second frame (t=0, c=1) - new channel, high water mark
        stream.append(frame)
        _wait_for_pending_callbacks(stream)
        assert len(expanded_history) == 2
        assert len(changed_history) == 2
        assert expanded_history[1].max_coords["c"] == range(2)
        assert expanded_history[1].current_indices["c"] == 1

        # Write third frame (t=1, c=0) - new timepoint, high water mark
        stream.append(frame)
        _wait_for_pending_callbacks(stream)
        # Hi dev!  See an error here in CI logs?? re-run, or increase BARRIERS!
        assert len(expanded_history) == 3
        assert len(changed_history) == 3
        assert expanded_history[2].max_coords["t"] == range(2)

        # Write fourth frame (t=1, c=1) - no new high water mark
        stream.append(frame)
        _wait_for_pending_callbacks(stream)
        assert len(expanded_history) == 3  # No new expanded event
        assert len(changed_history) == 4  # But coords_changed fires

        # Test skip crossing high water marks
        stream.skip(frames=2)  # Skip to end
        _wait_for_pending_callbacks(stream)
        # Hi dev!  See an error here in CI logs?? re-run, or increase BARRIERS!
        assert len(expanded_history) == 4  # New high water mark at t=2
        assert expanded_history[3].max_coords["t"] == range(3)


def test_coord_tracking_zero_overhead(tmp_path: Path, first_backend: str) -> None:
    """Test that coord tracking has zero overhead when no handlers registered."""
    suffix = get_format_for_backend(first_backend)
    settings = AcquisitionSettings(
        root_path=tmp_path / f"test.ome.{suffix}",
        dimensions=[
            Dimension(name="t", count=2, type="time"),
            Dimension(name="y", count=16, type="space"),
            Dimension(name="x", count=16, type="space"),
        ],
        dtype="uint16",
        format=first_backend,
        overwrite=True,
    )

    frame = np.zeros((16, 16), dtype=np.uint16)

    with create_stream(settings) as stream:
        # Write frames without any event handlers
        stream.append(frame)
        stream.append(frame)

        # Tracker should never be created
        assert stream._coord_tracker is None
        assert stream._frames_written == 2  # Frame count still tracked


def test_coord_tracking_mid_acquisition(tmp_path: Path, first_backend: str) -> None:
    """Test registering handler mid-acquisition correctly initializes tracker."""
    settings = AcquisitionSettings(
        root_path=tmp_path / "test",
        dimensions=[
            Dimension(name="t", count=3, type="time"),
            Dimension(name="c", count=2, type="channel"),
            Dimension(name="y", count=16, type="space"),
            Dimension(name="x", count=16, type="space"),
        ],
        dtype="uint16",
        format=first_backend,
        overwrite=True,
    )

    frame = np.zeros((16, 16), dtype=np.uint16)
    expanded_history: list[CoordUpdate] = []
    changed_history: list[CoordUpdate] = []

    def track_expanded(update: CoordUpdate) -> None:
        expanded_history.append(update)

    def track_changed(update: CoordUpdate) -> None:
        changed_history.append(update)

    with create_stream(settings) as stream:
        # Write several frames before registering handlers
        stream.append(frame)  # t=0, c=0
        stream.append(frame)  # t=0, c=1
        stream.append(frame)  # t=1, c=0
        assert stream._frames_written == 3

        # Now register handlers mid-acquisition
        stream.on("coords_expanded", track_expanded)
        stream.on("coords_changed", track_changed)

        # Tracker should be initialized with current frame count
        assert stream._coord_tracker is not None
        assert stream._coord_tracker._frames_written == 3

        # Should have correct initial coords based on frames already written
        initial_coords = stream._coord_tracker.get_coords()
        assert initial_coords["t"] == range(2)  # Seen t=0,1
        assert initial_coords["c"] == range(2)  # Seen c=0,1

        # Next frame doesn't cross high water mark
        stream.append(frame)  # t=1, c=1
        _wait_for_pending_callbacks(stream)
        # Hi dev!  See an error here in CI logs?? re-run, or increase BARRIERS!
        assert len(expanded_history) == 0  # No expanded event
        assert len(changed_history) == 1  # But coords_changed fires

        # This frame crosses to new timepoint - both events triggered
        stream.append(frame)  # t=2, c=0
        _wait_for_pending_callbacks(stream)
        assert len(expanded_history) == 1
        assert len(changed_history) == 2
        assert expanded_history[0].max_coords["t"] == range(3)
