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

- **Acquisition order**: The order dimensions appear in
  `AcquisitionSettings.dimensions`. This is how frames arrive from the microscope. For
  dimensions [T, Z, C, Y, X], frames arrive in nested-loop order where the last
  dimension varies fastest.

- **Storage order**: Controlled by `AcquisitionSettings.storage_order`. Can be:
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

>>> from ome_writers import AcquisitionSettings, Dimension
>>> from ome_writers._router import FrameRouter
>>> settings = AcquisitionSettings(
...     root_path="test.zarr",
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

>>> from ome_writers import Position, PositionDimension, AcquisitionSettings, Dimension
>>> from ome_writers._router import FrameRouter
>>> settings = AcquisitionSettings(
...     root_path="test.zarr",
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

from typing import TYPE_CHECKING

from .schema import AcquisitionSettings, Dimension, Position

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
    settings : AcquisitionSettings
        Schema describing dimensions, dtype, and storage order.

    Yields
    ------
    tuple[PositionInfo, tuple[int, ...]]
        (position_info, storage_index) for each frame.
        - position_info: tuple of (position_index, Position) with index and metadata
        - storage_index: N-dimensional index in storage order (excludes Y/X)
    """

    def __init__(self, settings: AcquisitionSettings) -> None:
        self._positions = settings.positions
        self._position_index: int | None = settings.position_dimension_index
        self._non_frame_sizes = tuple(d.count for d in settings.dimensions[:-2])
        self._num_non_frame_dims = len(self._non_frame_sizes)

        # Compute storage order permutation (for non-position "index" dimensions only)
        self._permutation = _compute_permutation(
            settings.index_dimensions, settings.storage_index_dimensions
        )
        self._is_permuted = self._permutation != tuple(range(len(self._permutation)))

        # Precompute which indices to extract for acq_idx (all except position)
        if self._position_index is not None:
            self._acq_idx_positions = tuple(
                i for i in range(self._num_non_frame_dims) if i != self._position_index
            )
        else:
            self._acq_idx_positions = tuple(range(self._num_non_frame_dims))

        self._reset()

    def __iter__(self) -> Iterator[tuple[PositionInfo, tuple[int, ...]]]:
        """Return iterator, resetting to first frame."""
        self._reset()
        return self

    def _reset(self) -> None:
        self._dim_indices = [0] * self._num_non_frame_dims
        self._finished = False

    def __next__(self) -> tuple[PositionInfo, tuple[int, ...]]:
        """Yield next (position_info, storage_index) tuple.

        This is the primary function of the FrameRouter, mapping acquisition
        order to storage order.

        For finite dimensions, iteration stops after all frames.
        For unlimited dimensions, iteration never stops - caller must break.
        """
        if self._finished:
            raise StopIteration

        # Extract position index directly from current indices
        if self._position_index is not None:
            pos_idx = self._dim_indices[self._position_index]
        else:
            pos_idx = 0

        # Build storage index by extracting relevant indices and applying permutation
        storage_idx = tuple(self._dim_indices[i] for i in self._acq_idx_positions)
        if self._is_permuted:
            storage_idx = tuple(storage_idx[p] for p in self._permutation)

        # Increment indices for next iteration
        self._increment_indices()
        return (pos_idx, self._positions[pos_idx]), storage_idx

    def _increment_indices(self) -> None:
        """Increment dimension indices like nested loops (rightmost varies fastest).

        Sets _finished flag when all finite dimensions have been exhausted.
        For unlimited dimensions, never sets _finished.
        """
        if not self._num_non_frame_dims:
            self._finished = True
            return

        # Fast path: increment rightmost dimension (99%+ of calls)
        last_idx = self._num_non_frame_dims - 1
        self._dim_indices[last_idx] += 1

        size_limit = self._non_frame_sizes[last_idx]
        if size_limit is None:
            return  # Unlimited dimension
        if self._dim_indices[last_idx] < size_limit:
            return  # Still within bounds - common case

        # Slow path: wrap rightmost and carry to other dimensions (rare)
        self._dim_indices[last_idx] = 0

        # Handle remaining dimensions (only when wrapping occurs)
        for i in range(last_idx - 1, -1, -1):
            self._dim_indices[i] += 1

            size_limit = self._non_frame_sizes[i]
            if size_limit is None:
                return  # Unlimited dimension
            elif self._dim_indices[i] < size_limit:
                return  # Still within bounds

            # Exceeded this dimension's limit - reset and carry to next dimension
            self._dim_indices[i] = 0

        # Wrapped all dimensions - iteration is complete
        self._finished = True


def _compute_permutation(
    acq_dims: list[Dimension], storage_names: list[Dimension]
) -> tuple[int, ...]:
    """Compute permutation to convert acquisition indices to storage indices."""
    dim_names = [dim.name for dim in acq_dims]
    return tuple(dim_names.index(dim.name) for dim in storage_names)
