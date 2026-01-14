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

The router yields `(position_info, storage_index)` tuples where:
- `position_info` is a tuple of (position_index, Position) identifying the position
- `storage_index` is the N-dimensional index in storage order

Backends receive storage-order indices directly and don't need to know about
acquisition order. They can use position_info to access the position index
(for array lookup) and Position metadata (name, row, column for path building).

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
>>> for (pos_idx, pos), idx in router:
...     print(f"pos={pos_idx}:{pos.name}, idx={idx}")
pos=0:0, idx=(0, 0)
pos=0:0, idx=(0, 1)
pos=0:0, idx=(0, 2)
pos=0:0, idx=(1, 0)
pos=0:0, idx=(1, 1)
pos=0:0, idx=(1, 2)

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
>>> for (pos_idx, pos), idx in router:
...     print(f"pos={pos_idx}:{pos.name}, idx={idx}")
pos=0:A1, idx=(0, 0)
pos=0:A1, idx=(0, 1)
pos=1:B2, idx=(0, 0)
pos=1:B2, idx=(0, 1)
pos=0:A1, idx=(1, 0)
pos=0:A1, idx=(1, 1)
pos=1:B2, idx=(1, 0)
pos=1:B2, idx=(1, 1)
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from .schema import ArraySettings, Dimension, Position, PositionDimension

if TYPE_CHECKING:
    from collections.abc import Iterator

# Type alias for position info: (position_index, Position)
PositionInfo = tuple[int, Position]


class FrameRouter:
    """Stateful iterator mapping acquisition frames to storage locations.

    Parameters
    ----------
    settings : ArraySettings
        Schema describing dimensions, dtype, and storage order.

    Yields
    ------
    tuple[PositionInfo, tuple[int, ...]]
        (position_info, storage_index) for each frame.
        - position_info: tuple of (position_index, Position) with index and metadata
        - storage_index: N-dimensional index in storage order (excludes Y/X)
    """

    def __init__(self, settings: ArraySettings) -> None:
        self._settings = settings

        # Parse dimensions, tracking where position appears in the order.
        # all_sizes includes position; index_dim_names excludes it.
        all_sizes: list[int | None] = []
        index_dims: list[Dimension] = []
        position_axis: int | None = None
        # Default single position with name "0"
        positions: list[Position] = [Position(name="0")]

        for dim in settings.dimensions:
            if isinstance(dim, PositionDimension):
                if position_axis is not None:
                    raise ValueError("Only one PositionDimension is allowed")
                position_axis = len(all_sizes)
                all_sizes.append(dim.count)
                positions = dim.positions
            # FIXME!!! don't hardcode names "Y" and "X"
            elif isinstance(dim, Dimension) and dim.name.lower() not in ("y", "x"):
                all_sizes.append(dim.count)
                index_dims.append(dim)

        self._all_sizes = tuple(all_sizes)
        self._position_axis = position_axis
        self._positions = positions
        self._index_dim_names = index_dims

        # Compute storage order permutation (for non-position dimensions only)
        self._storage_dim_names = _resolve_storage_order(
            index_dims, settings.storage_order
        )
        self._permutation = _compute_permutation(index_dims, self._storage_dim_names)

        self.reset()

    def __iter__(self) -> Iterator[tuple[PositionInfo, tuple[int, ...]]]:
        """Return iterator, resetting to first frame."""
        self.reset()
        return self

    def reset(self) -> None:
        self._dim_indices = [0] * len(self._all_sizes)
        self._finished = False

    def __next__(self) -> tuple[PositionInfo, tuple[int, ...]]:
        """Yield next (position_info, storage_index) tuple.

        For finite dimensions, iteration stops after all frames.
        For unlimited dimensions, iteration never stops - caller must break.
        """
        if self._finished or not self._all_sizes:
            raise StopIteration

        # Use current dimension indices
        full_idx = tuple(self._dim_indices)

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
        position_info: PositionInfo = (pos_idx, self._positions[pos_idx])

        # Increment indices for next iteration
        self._increment_indices()

        return position_info, storage_idx

    def _increment_indices(self) -> None:
        """Increment dimension indices like nested loops (rightmost varies fastest).

        Sets _finished flag when all finite dimensions have been exhausted.
        For unlimited dimensions, never sets _finished.
        """
        # Start from rightmost dimension (varies fastest)
        for i in range(len(self._dim_indices) - 1, -1, -1):
            self._dim_indices[i] += 1

            # Check if we've exceeded the limit for this dimension
            size_limit = self._all_sizes[i]
            if size_limit is None:
                # Unlimited dimension - never wraps, iteration continues indefinitely
                return
            elif self._dim_indices[i] < size_limit:
                # Still within bounds for this dimension
                return

            # Exceeded this dimension's limit - reset and carry to next dimension
            self._dim_indices[i] = 0

        # Wrapped all dimensions - iteration is complete
        self._finished = True

    @property
    def positions(self) -> list[Position]:
        """Position objects in acquisition order.

        Backends use this to create arrays/files for each position before
        iteration begins. Each Position contains name, row, and column metadata.
        """
        return list(self._positions)

    @property
    def storage_dimension_names(self) -> list[str]:
        """Dimension names in storage order (excluding frame dims like XY)."""
        return list(self._storage_dim_names)


def _ngff_sort_key(dim: Dimension) -> tuple[int, int]:
    """Sort key for NGFF canonical order: time, channel, space (z, y, x)."""
    if dim.type == "time":
        return (0, 0)
    if dim.type == "channel":
        return (1, 0)
    if dim.type == "space":
        return (2, {"z": 0, "y": 1, "x": 2}.get(dim.name, -1))
    return (3, 0)


def _resolve_storage_order(
    acq_dims: list[Dimension],
    storage_order: str | list[str],
) -> list[str]:
    """Resolve storage_order setting to explicit list of dimension names."""
    if storage_order == "acquisition":
        return [dim.name for dim in acq_dims]
    elif storage_order == "ngff":
        return [x.name for x in sorted(acq_dims, key=_ngff_sort_key)]
    elif isinstance(storage_order, list):
        acq_dim_names = {dim.name for dim in acq_dims}
        if set(storage_order) != acq_dim_names:
            raise ValueError(
                f"storage_order {storage_order} doesn't match "
                f"acquisition dimensions {acq_dim_names}"
            )
        return list(storage_order)
    else:
        raise ValueError(f"Invalid storage_order: {storage_order!r}")


def _compute_permutation(
    acq_dims: list[Dimension], storage_names: list[str]
) -> tuple[int, ...]:
    """Compute permutation to convert acquisition indices to storage indices."""
    dim_names = [dim.name for dim in acq_dims]
    return tuple(dim_names.index(name) for name in storage_names)
