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

>>> from ome_writers import ArraySettings, Dimension
>>> from ome_writers._router import FrameRouter
>>> settings = ArraySettings(
...     dimensions=[
...         Dimension(name="t", count=2),
...         Dimension(name="c", count=3),
...         Dimension(name="y", count=64, type="space"),
...         Dimension(name="x", count=64, type="space"),
...     ],
...     dtype="uint16",
... )
>>> router = FrameRouter(settings)
>>> for pos_info, idx in router:
...     print(f"pos={pos_info}, idx={idx}")
pos=(0, Position(name='0', row=None, column=None)), idx=(0, 0)
pos=(0, Position(name='0', row=None, column=None)), idx=(0, 1)
pos=(0, Position(name='0', row=None, column=None)), idx=(0, 2)
pos=(0, Position(name='0', row=None, column=None)), idx=(1, 0)
pos=(0, Position(name='0', row=None, column=None)), idx=(1, 1)
pos=(0, Position(name='0', row=None, column=None)), idx=(1, 2)

Multi-position with position interleaved (time-lapse across positions):

>>> from ome_writers import Position, PositionDimension, ArraySettings, Dimension
>>> from ome_writers._router import FrameRouter
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
...         Dimension(name="y", count=64, type="space"),
...         Dimension(name="x", count=64, type="space"),
...     ],
...     dtype="uint16",
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

from typing import TYPE_CHECKING, cast

from .schema import ArraySettings, Dimension, Position, PositionDimension

if TYPE_CHECKING:
    from collections.abc import Iterator

# Type alias for position info: (position_index, Position)
PositionInfo = tuple[int, Position]


class FrameRouter:
    """Stateful iterator mapping acquisition frames to storage locations.

    We differentiate between three sets of dimensions:
    1. In-frame Dimensions (usually the last two dimensions, often called Y and X and
       of type='space').
    2. Index Dimensions: all non-frame Dimensions excluding PositionDimension (used
       for indexing within each position when appending new frames)
       (we also track the permutation of index-dimensions into "storage order", if the
       user has specified a storage order that differs from acquisition order).
    3. Position Dimension: PositionDimension, if present (used to select which position
       array to write to).

    Routers *may* be iterated over multiple times; each iteration resets to the
    beginning.

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
        dims = settings.dimensions

        # we have already validated that last two dims are spatial in
        # schema._validate_dims_list
        self._frame_dims = cast("list[Dimension]", dims[-2:])

        non_frame_dims = dims[:-2]

        # sizes of all non-frame dimensions, *including* position dimension if present
        self._non_frame_sizes: tuple[int | None, ...] = tuple(
            d.count for d in non_frame_dims
        )

        # Index Dimensions (all non-frame dims excluding PositionDimension)
        self._index_dims: list[Dimension] = []

        # Positions list from the PositionDimension
        # Defaults single position with name "0"
        self._positions: list[Position] = [Position(name="0")]
        # index of PositionDimension in dimensions, if present
        self._position_index: int | None = None

        for idx, dim in enumerate(non_frame_dims):
            if isinstance(dim, PositionDimension):
                self._position_index = idx
                self._positions = dim.positions
            else:
                self._index_dims.append(dim)

        # Compute storage order permutation (for non-position "index" dimensions only)
        self._storage_index_dims = _sort_dims_to_storage_order(
            self._index_dims, settings.storage_order
        )
        self._permutation = _compute_permutation(
            self._index_dims, self._storage_index_dims
        )

        self._reset()

    @property
    def positions(self) -> tuple[Position, ...]:
        """Position objects in acquisition order.

        Backends use this to create arrays/files for each position before
        iteration begins. Each Position contains name, row, and column metadata.
        """
        return tuple(self._positions)

    @property
    def array_storage_dimensions(self) -> tuple[Dimension, ...]:
        """Full array dims in storage order, including spatial dims (Y, X) at end.

        Returns the full Dimension objects with all metadata (count, scale, type, etc.)
        reordered according to storage_order, with spatial dimensions appended.
        Backends use this to build arrays with the correct shape and metadata.
        """
        return tuple(self._storage_index_dims) + tuple(self._frame_dims)

    @property
    def num_frames(self) -> int | None:
        """Return total number of frames, or None if unlimited dimension present."""
        total = 1
        for size in self._non_frame_sizes:
            if size is None:
                return None
            total *= size
        return total

    def __iter__(self) -> Iterator[tuple[PositionInfo, tuple[int, ...]]]:
        """Return iterator, resetting to first frame."""
        self._reset()
        return self

    def __next__(self) -> tuple[PositionInfo, tuple[int, ...]]:
        """Yield next (position_info, storage_index) tuple.

        This is the primary function of the FrameRouter, mapping acquisition
        order to storage order.

        For finite dimensions, iteration stops after all frames.
        For unlimited dimensions, iteration never stops - caller must break.
        """
        if self._finished or not self._non_frame_sizes:
            raise StopIteration

        # Use current dimension indices
        full_idx = tuple(self._dim_indices)

        # Extract position and build storage index
        if self._position_index is not None:
            pos_idx = full_idx[self._position_index]
            # Remove position from index
            acq_idx = (
                full_idx[: self._position_index] + full_idx[self._position_index + 1 :]
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

    def _reset(self) -> None:
        self._dim_indices = [0] * len(self._non_frame_sizes)
        self._finished = False

    def _increment_indices(self) -> None:
        """Increment dimension indices like nested loops (rightmost varies fastest).

        Sets _finished flag when all finite dimensions have been exhausted.
        For unlimited dimensions, never sets _finished.
        """
        # Start from rightmost dimension (varies fastest)
        for i in range(len(self._dim_indices) - 1, -1, -1):
            self._dim_indices[i] += 1

            # Check if we've exceeded the limit for this dimension
            size_limit = self._non_frame_sizes[i]
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


def _ngff_sort_key(dim: Dimension) -> tuple[int, int]:
    """Sort key for NGFF canonical order: time, channel, space (z, y, x)."""
    if dim.type == "time":
        return (0, 0)
    if dim.type == "channel":
        return (1, 0)
    if dim.type == "space":
        return (2, {"z": 0, "y": 1, "x": 2}.get(dim.name, -1))
    return (3, 0)


def _sort_dims_to_storage_order(
    acq_dims: list[Dimension],
    storage_order: str | list[str],
) -> list[Dimension]:
    """Resolve storage_order setting to explicit list of dimension names."""
    if storage_order == "acquisition":
        return list(acq_dims)
    elif storage_order == "ngff":
        return sorted(acq_dims, key=_ngff_sort_key)
    elif isinstance(storage_order, list):
        dims_map = {dim.name: dim for dim in acq_dims}
        if set(storage_order) != set(dims_map):
            raise ValueError(
                f"storage_order names {storage_order!r} don't match "
                f"acquisition dimension names {list(dims_map)}"
            )
        return [dims_map[name] for name in storage_order]
    else:
        raise ValueError(
            f"Invalid storage_order: {storage_order!r}. Must be 'acquisition', 'ngff', "
            "or list of names."
        )


def _compute_permutation(
    acq_dims: list[Dimension], storage_names: list[Dimension]
) -> tuple[int, ...]:
    """Compute permutation to convert acquisition indices to storage indices."""
    dim_names = [dim.name for dim in acq_dims]
    return tuple(dim_names.index(dim.name) for dim in storage_names)
