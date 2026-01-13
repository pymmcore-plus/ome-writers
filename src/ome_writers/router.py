"""FrameRouter: maps streaming frame indices to storage locations.

The FrameRouter is the stateful iterator at the heart of ome-writers. It bridges
acquisition order (how frames arrive from the camera) and storage order (how
arrays are organized on disk).

Key Responsibilities
--------------------
1. **Iterate in acquisition order** - respects dimension order in the schema
2. **Emit storage-order indices** - applies permutation when storage differs
3. **Handle position as meta-dimension** - positions become separate arrays, not an axis

Design Notes
------------
The router is the *only* component that understands both orderings:

- **Acquisition order**: The order dimensions appear in `ArraySettings.dimensions`.
  This is how frames arrive from the microscope. For dimensions [T, Z, C, Y, X],
  frames arrive in nested-loop order where the last dimension varies fastest.

- **Storage order**: Controlled by `ArraySettings.storage_order`. Can be:
  - "acquisition" - same as acquisition order
  - "ngff" - canonical NGFF order (time, channel, space)
  - list[str] - explicit axis names

The router yields `(position_key, storage_index)` tuples where:
- `position_key` identifies which array/file to write to
- `storage_index` is the N-dimensional index in storage order

Backends receive storage-order indices directly and don't need to know about
acquisition order.

Examples
--------
Basic iteration with storage order matching acquisition order:

>>> from ome_writers.schema import ArraySettings, Dimension
>>> import numpy as np
>>> settings = ArraySettings(
...     dimensions=[
...         Dimension(name="t", count=2),
...         Dimension(name="c", count=3),
...         Dimension(name="y", count=64),
...         Dimension(name="x", count=64),
...     ],
...     dtype=np.uint16,
... )
>>> router = FrameRouter(settings)
>>> for pos_key, idx in router:
...     print(f"pos={pos_key}, idx={idx}")
pos=0, idx=(0, 0)
pos=0, idx=(0, 1)
pos=0, idx=(0, 2)
pos=0, idx=(1, 0)
pos=0, idx=(1, 1)
pos=0, idx=(1, 2)

With reordering (acquisition TZC, storage TCZ for NGFF):

>>> settings = ArraySettings(
...     dimensions=[
...         Dimension(name="t", count=2),
...         Dimension(name="z", count=2),
...         Dimension(name="c", count=2),
...         Dimension(name="y", count=64),
...         Dimension(name="x", count=64),
...     ],
...     dtype=np.uint16,
...     storage_order="ngff",
... )
>>> router = FrameRouter(settings)
>>> for pos_key, idx in router:
...     print(f"pos={pos_key}, idx={idx}")
pos=0, idx=(0, 0, 0)
pos=0, idx=(0, 1, 0)
pos=0, idx=(0, 0, 1)
pos=0, idx=(0, 1, 1)
pos=0, idx=(1, 0, 0)
pos=0, idx=(1, 1, 0)
pos=0, idx=(1, 0, 1)
pos=0, idx=(1, 1, 1)

Multi-position with position interleaved (time-lapse across positions):

>>> from ome_writers.schema import Position, PositionDimension, ArraySettings, Dimension
>>> from ome_writers.router import FrameRouter
>>> settings = ArraySettings(
...     dimensions=[
...         Dimension(name="t", count=2),
...         PositionDimension(
...             positions=[
...                 Position(name="A1"),
...                 Position(name="B2"),
...             ]
...         ),
...         Dimension(name="c", count=2),
...         Dimension(name="y", count=64),
...         Dimension(name="x", count=64),
...     ],
...     dtype=np.uint16,
... )
>>> router = FrameRouter(settings)
>>> for pos_key, idx in router:
...     print(f"pos={pos_key}, idx={idx}")
pos=A1, idx=(0, 0)
pos=A1, idx=(0, 1)
pos=B2, idx=(0, 0)
pos=B2, idx=(0, 1)
pos=A1, idx=(1, 0)
pos=A1, idx=(1, 1)
pos=B2, idx=(1, 0)
pos=B2, idx=(1, 1)
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from .schema_pydantic import ArraySettings, Dimension, PositionDimension

if TYPE_CHECKING:
    from collections.abc import Iterator


class FrameRouter:
    """Stateful iterator mapping acquisition frames to storage locations.

    Parameters
    ----------
    settings : ArraySettings
        Schema describing dimensions, dtype, and storage order.

    Yields
    ------
    tuple[str, tuple[int, ...]]
        (position_key, storage_index) for each frame.
        - position_key: string name identifying the array
        - storage_index: N-dimensional index in storage order (excludes Y/X)
    """

    def __init__(self, settings: ArraySettings) -> None:
        self._settings = settings

        # Parse dimensions, tracking where position appears in the order.
        # all_sizes includes position; index_dim_names excludes it.
        all_sizes: list[int] = []
        index_dim_names: list[str] = []
        position_axis: int | None = None
        position_names: list[str] = ["0"]  # default single position

        for dim in settings.dimensions:
            if isinstance(dim, PositionDimension):
                if position_axis is not None:
                    raise ValueError("Only one PositionDimension is allowed")
                position_axis = len(all_sizes)
                all_sizes.append(dim.count)
                position_names = dim.names
            elif isinstance(dim, Dimension) and dim.name not in ("y", "x"):
                all_sizes.append(dim.count)
                index_dim_names.append(dim.name)

        self._all_sizes = tuple(all_sizes)
        self._position_axis = position_axis
        self._position_names = position_names
        self._index_dim_names = index_dim_names

        # Total frames across all positions
        self._total_frames = int(np.prod(self._all_sizes)) if self._all_sizes else 1

        # Compute storage order permutation (for non-position dimensions only)
        self._storage_dim_names = _resolve_storage_order(
            index_dim_names, settings.storage_order
        )
        self._permutation = _compute_permutation(
            index_dim_names, self._storage_dim_names
        )

        # Iteration state
        self._current_frame = 0

    def __iter__(self) -> Iterator[tuple[str, tuple[int, ...]]]:
        """Return iterator, resetting to first frame."""
        self._current_frame = 0
        return self

    def __next__(self) -> tuple[str, tuple[int, ...]]:
        """Yield next (position_key, storage_index) tuple."""
        if self._current_frame >= self._total_frames:
            raise StopIteration

        # Compute full N-dimensional index (including position if present)
        full_idx = self._linear_to_nd(self._current_frame, self._all_sizes)

        # Extract position and build storage index
        if self._position_axis is not None:
            pos_idx = full_idx[self._position_axis]
            # Remove position from index
            acq_idx = (
                full_idx[: self._position_axis] + full_idx[self._position_axis + 1 :]
            )
        else:
            pos_idx = 0
            acq_idx = full_idx

        # Apply permutation: acquisition order → storage order
        storage_idx = tuple(acq_idx[p] for p in self._permutation)
        pos_key = self._position_names[pos_idx]

        self._current_frame += 1
        return pos_key, storage_idx

    @staticmethod
    def _linear_to_nd(linear_idx: int, sizes: tuple[int, ...]) -> tuple[int, ...]:
        """Convert linear index to N-dimensional index (row-major order)."""
        if not sizes:
            return ()
        indices: list[int] = []
        remaining = linear_idx
        for size in reversed(sizes):
            indices.append(remaining % size)
            remaining //= size
        return tuple(reversed(indices))

    @property
    def position_keys(self) -> list[str]:
        """Position names in acquisition order.

        Backends use this to create arrays/files for each position before
        iteration begins.
        """
        return list(self._position_names)


def _ngff_sort_key(name: str) -> tuple[int, int]:
    """Sort key for NGFF canonical order: time, channel, space (z, y, x)."""
    if name == "t":
        return (0, 0)
    if name == "c":
        return (1, 0)
    if name in ("z", "y", "x"):
        return (2, {"z": 0, "y": 1, "x": 2}[name])
    return (3, 0)


def _resolve_storage_order(
    acq_dim_names: list[str],
    storage_order: str | list[str],
) -> list[str]:
    """Resolve storage_order setting to explicit list of dimension names."""
    if storage_order == "acquisition":
        return list(acq_dim_names)
    elif storage_order == "ngff":
        return sorted(acq_dim_names, key=_ngff_sort_key)
    elif isinstance(storage_order, list):
        if set(storage_order) != set(acq_dim_names):
            raise ValueError(
                f"storage_order {storage_order} doesn't match "
                f"acquisition dimensions {acq_dim_names}"
            )
        return list(storage_order)
    else:
        raise ValueError(f"Invalid storage_order: {storage_order!r}")


def _compute_permutation(
    acq_names: list[str], storage_names: list[str]
) -> tuple[int, ...]:
    """Compute permutation to convert acquisition indices to storage indices."""
    return tuple(acq_names.index(name) for name in storage_names)
