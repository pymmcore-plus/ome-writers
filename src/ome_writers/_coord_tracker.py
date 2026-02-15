"""Coordinate tracking for monitoring acquisition progress."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, cast

import numpy as np

if TYPE_CHECKING:
    from collections.abc import Hashable, Mapping, Sequence

    from ome_writers._schema import AcquisitionSettings


@dataclass(frozen=True, slots=True)
class CoordUpdate:
    """Coordinate state information passed to event handlers."""

    max_coords: Mapping[str, Sequence]
    """Mapping of dimension names to maximum visited coordinate ranges."""
    current_indices: Mapping[str, int]
    """Mapping of dimension names to last written position."""
    frame_number: int
    """Global frame number."""
    is_high_water_mark: bool
    """True if this frame is a new "high water mark" (e.g. `max_coords` has expanded)."""  # noqa: E501


def high_water_marks(shape: tuple[range | int, ...]) -> dict[int, list[int]]:
    """Return frame indices where max dimension index increases."""
    if not shape:
        return {}

    first, *rest = shape

    if isinstance(first, range):
        a0_lo, a0_hi = first.start, first.stop
    else:
        a0_lo, a0_hi = 0, first

    strides = []
    stride = 1
    for s in reversed([a0_hi, *rest]):
        strides.append((stride, s - 1))  # type:ignore
        stride *= s  # type:ignore
    strides.reverse()

    lo = a0_lo * strides[0][0]
    hi = a0_hi * strides[0][0]

    arrays = []
    for st, mx in strides:
        v_lo = -(-lo // st)
        v_hi = min((hi - 1) // st, mx)
        if v_lo <= v_hi:
            arrays.append(np.arange(v_lo, v_hi + 1) * st)

    if not arrays:
        return {}

    bump_indices = np.unique(np.concatenate(arrays))
    values = np.column_stack([np.minimum(bump_indices // st, mx) for st, mx in strides])
    a, b = bump_indices.tolist(), values.tolist()
    return dict(zip(a, b, strict=False))


class _CoordTracker:
    """Tracks coordinate visibility and detects high water marks."""

    def __init__(
        self, settings: AcquisitionSettings, initial_frame_count: int = 0
    ) -> None:
        dims = settings.dimensions
        self._settings = settings
        self._non_frame_dims = dims[:-2]

        # Compute high water marks for non-frame dimensions
        non_frame_shape = tuple(d.count or 10000 for d in dims[:-2])
        self._high_water_marks = high_water_marks(non_frame_shape)
        self._frames_written = initial_frame_count
        self._needs_current_indices = False  # Optimization flag

        # Compute dimension strides for index calculations
        self._strides = []
        stride = 1
        for dim in reversed(self._non_frame_dims):
            self._strides.append(stride)
            stride *= dim.count or 10000
        self._strides.reverse()

        # Initialize current max indices based on initial frame count
        self._current_max_indices = [-1] * len(non_frame_shape)
        if initial_frame_count > 0:
            # Find the highest water mark we've already passed
            for frame_num in sorted(self._high_water_marks.keys()):
                if frame_num < initial_frame_count:
                    self._current_max_indices = self._high_water_marks[frame_num]
                else:
                    break

        # Base coords (frame dims always full range)
        self._base_coords = {d.name: range(1) for d in dims[:-2]}
        self._base_coords.update({d.name: range(d.count) for d in dims[-2:]})

    def _frame_to_indices(self, frame_num: int) -> list[int]:
        """Convert linear frame index to multi-dimensional indices."""
        indices = []
        remaining = frame_num
        # Iterate in reverse order (last dimension varies fastest)
        for dim in reversed(self._non_frame_dims):
            size = dim.count or 10000
            indices.append(remaining % size)
            remaining //= size
        # Reverse to get original dimension order
        return list(reversed(indices))

    def _indices_to_dict(self, indices: list[int]) -> dict[Hashable, int]:
        """Convert indices list to dict mapping dimension names to values."""
        result: dict[Hashable, int] = {}
        for i, dim in enumerate(self._non_frame_dims):
            result[dim.name] = indices[i]
        # Frame dims always at 0 (single plane)
        for dim in self._settings.dimensions[-2:]:
            result[dim.name] = 0
        return result

    def set_needs_current_indices(self, value: bool) -> None:
        """Configure whether current indices need to be computed."""
        self._needs_current_indices = value

    def update(self) -> CoordUpdate | None:
        """Increment frame count and return coordinate state if needed.

        Returns None on non-HWM frames if needs_current_indices=False
        (optimization for COORDS_EXPANDED-only listeners).
        """
        frame_num = self._frames_written
        self._frames_written += 1

        # Check for high water mark
        is_hwm = frame_num in self._high_water_marks
        if is_hwm:
            self._current_max_indices = self._high_water_marks[frame_num]

        # Skip computation if not needed (only COORDS_EXPANDED and not HWM)
        if not self._needs_current_indices and not is_hwm:
            return None

        # Compute full update (only when needed)
        current_indices = self._frame_to_indices(frame_num)
        return CoordUpdate(
            max_coords=self.get_coords(),
            current_indices=self._indices_to_dict(current_indices),
            frame_number=frame_num,
            is_high_water_mark=is_hwm,
        )

    def skip(self, frames: int) -> CoordUpdate | None:
        """Update frame count for skipped frames, return update if HWM crossed."""
        start_idx = self._frames_written
        end_idx = start_idx + frames
        self._frames_written = end_idx

        # Check if we crossed any high water marks
        crossed_marks = [
            (idx, marks)
            for idx, marks in self._high_water_marks.items()
            if start_idx <= idx < end_idx
        ]

        if crossed_marks:
            # Update to the highest mark we crossed
            _, self._current_max_indices = max(crossed_marks, key=lambda x: x[0])

            # Use the end position as "current"
            current_indices = self._frame_to_indices(end_idx - 1)

            return CoordUpdate(
                max_coords=self.get_coords(),
                current_indices=self._indices_to_dict(current_indices),
                frame_number=end_idx - 1,
                is_high_water_mark=True,
            )
        return None

    def get_coords(self) -> Mapping[Hashable, Sequence]:
        """Get mapping of dimension names to visible coordinate ranges."""
        result = dict(self._base_coords)

        for i, dim in enumerate(self._non_frame_dims):
            max_idx = self._current_max_indices[i] + 1
            if dim.coords:
                result[dim.name] = [c.name for c in dim.coords[:max_idx]]
            else:
                result[dim.name] = range(max_idx)

        return cast("Mapping[Hashable, Sequence]", result)
