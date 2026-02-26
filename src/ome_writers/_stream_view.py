from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from collections.abc import Callable, Hashable, Iterable, Mapping, Sequence
    from typing import Any, SupportsIndex, TypeAlias, TypeVar

    from typing_extensions import Self

    from ome_writers._backends._backend import ArrayLike
    from ome_writers._coord_tracker import CoordUpdate
    from ome_writers._stream import OMEStream

    Index: TypeAlias = SupportsIndex | slice
    F = TypeVar("F", bound=Callable[..., Any])


class _SimpleSignalInstance:
    """Minimal signal with connect/disconnect/emit (no dependencies)."""

    __slots__ = ("_slots",)

    def __init__(self) -> None:
        self._slots: list[Callable[[], None]] = []

    def connect(self, callback: F) -> F:
        """Connect a callback to the signal. The callback must take NO arguments.

        Note: a direct (strong) reference to the callback is stored.  The same
        object must be used when calling `disconnect`.

        Returns
        -------
        Callable
            The same callback that was passed in, for convenience in decorator usage.
        """
        self._slots.append(callback)
        return callback

    def disconnect(self, callback: Callable[[], None]) -> None:
        """Remove a previously connected callback."""
        self._slots.remove(callback)

    def emit(self) -> None:
        """Emit the signal, calling all connected callbacks."""
        for cb in self._slots:
            cb()


class StreamView:
    """Read-only view presenting multiple position arrays as a single array.

    This class provides a read-only view onto an ongoing acquisition stream.  It
    combines all underlying arrays back into a single higher-dimensional array with the
    same shape *and order* as the original `AcquisitionSettings.dimensions`.  Use it as
    you would a normal numpy array, but note that it only supports:

    - `__getitem__` with integer and slice indexing (no advanced indexing)
    - `__array__` for conversion to a numpy array (with optional dtype conversion)
    - `shape`, `dtype`, and `ndim` properties
    - `__len__` returning the number of positions (size of position dimension)
    - `dims` property returning dimension labels (xarray-style)

    Accessing not-yet-written positions or frames should return zeros (or the
    fill-value of the underlying arrays, if specified).

    !!! warning
        It is *strongly discouraged* to materialize the entire view as a single array
        (e.g., via  `view[:]` or `np.asarray(view)`).  It is intended as a preview
        onto individual frames at a time.
    """

    __slots__ = (
        "_acq_perm",
        "_arrays",
        "_coords_changed",
        "_coords_data",
        "_dims",
        "_dtype",
        "_inv_perm",
        "_n_perm",
        "_position_axis",
        "_shape_override",
        "_strict_bounds",
    )

    @classmethod
    def from_stream(
        cls,
        stream: OMEStream,
        *,
        dynamic_shape: bool = True,
        strict: bool = False,
    ) -> Self:
        """Create view directly from OMEStream.

        Prefer using [`OMEStream.view`][ome_writers.OMEStream.view], which calls
        this method internally (and may have additional logic).

        Parameters
        ----------
        stream : OMEStream
            The stream to create a view from.
        dynamic_shape : bool
            If True, `shape`/`coords` dynamically reflect only what has been acquired.
            `dims` will always reflect the full acquisition dimensions, but `shape` will
            start with zeros for non-frame dims and grow as new frames/positions are
            written, and `coords` will represent the coords of the currently acquired
            frames.  If False, `shape`/`coords` reflect the full/expected acquisition
            settings from the start.
        strict : bool
            If True (and dynamic_shape=True), raise IndexError on integer indices
            outside the live bounds.
        """
        settings = stream._settings
        if settings.is_unbounded:  # pragma: no cover
            raise NotImplementedError(
                "Creating a view on an unbounded stream is not currently supported."
            )

        # Compute acquisition order permutation from settings
        # this is the permutation required to convert the storage_order arrays
        # *back* to acquisition order.
        if (sp := settings.storage_index_permutation) is not None:
            acquisition_perm = tuple(sp.index(i) for i in range(len(sp)))
        else:
            acquisition_perm = None

        view = cls(
            stream._backend.get_arrays(),
            position_axis=settings.position_dimension_index,
            acquisition_order_perm=acquisition_perm,
            dimension_labels=[d.name for d in settings.dimensions],
        )

        # Compute full coords from settings
        full_coords: dict[Hashable, Sequence] = {}
        for dim in settings.dimensions:
            if dim.coords:
                full_coords[dim.name] = [getattr(c, "name", c) for c in dim.coords]
            else:
                full_coords[dim.name] = range(dim.count)
        view._coords_data = full_coords

        if dynamic_shape:
            view._strict_bounds = strict
            # Register callback (lazily creates coord tracker)
            stream.on("coords_expanded", view._on_coords_expanded)
            # read initial state from coord tracker
            assert stream._coord_tracker is not None  # should be created by on()
            coords = stream._coord_tracker.get_coords()
            view._coords_data = dict(coords)
            view._shape_override = tuple(len(coords[d]) for d in view._dims)

        return view

    def __init__(
        self,
        arrays: Sequence[ArrayLike],
        *,
        position_axis: int | None = 0,
        acquisition_order_perm: tuple[int, ...] | None = None,
        dimension_labels: Sequence[str] = (),
    ) -> None:
        if not arrays:  # pragma: no cover
            raise ValueError("Arrays list cannot be empty")

        # NB:
        # we could potentially apply the acquisition_order_perm HERE, and remove
        # logic in __getitem__.  But that puts more demands on the ArrayLike object
        # ... let's explore that later.
        self._arrays = list(arrays)
        a0 = arrays[0]
        dtype = getattr(a0.dtype, "name", a0.dtype)  # handle non-numpy arrays
        self._dtype = np.dtype(dtype)
        self._acq_perm = acquisition_order_perm
        self._dims: tuple[str, ...] = tuple(dimension_labels)

        # Validate shapes match
        first, *others = self._arrays
        ref = first.shape
        for i, arr in enumerate(others, 1):
            if arr.shape != ref:  # pragma: no cover
                raise ValueError(f"Array {i} shape {arr.shape} != {ref}")

        # Store inverse permutation and cached length for indexing
        if acquisition_order_perm:
            self._n_perm = n = len(acquisition_order_perm)
            self._inv_perm = tuple(acquisition_order_perm.index(i) for i in range(n))
        else:
            self._n_perm = 0
            self._inv_perm = None

        # Validate position_axis (None means no position dimension)
        if position_axis is not None:
            acq_shape = self._acq_shape(ref)
            if not 0 <= position_axis <= len(acq_shape):  # pragma: no cover
                raise ValueError(f"position_axis {position_axis} out of range")
        self._position_axis = position_axis

        # Live-shape tracking (set by from_stream when dynamic_shape=True)
        self._coords_data: Mapping[Hashable, Sequence] | None = None
        self._shape_override: tuple[int, ...] | None = None
        self._strict_bounds: bool = False
        self._coords_changed = _SimpleSignalInstance()

    @property
    def coords_changed(self) -> _SimpleSignalInstance:
        """Signal emitted when coords/shape change in `dynamic_shape` mode.

        This is a simple API, with no dependencies on psygnal/Qt.  The returned object
        provides `connect` and `disconnect` methods for registering callbacks, and an
        `emit` method for emitting the signal.  Callbacks must take no arguments.
        """
        return self._coords_changed

    @property
    def coords(self) -> Mapping[Hashable, Sequence]:
        """Coordinate labels for each dimension.

        In practice this will be `Mapping[str, Sequence]` but for consistency with
        the broader xarray semantics, consumers should prepare for any hashable keys.
        """
        if self._coords_data is not None:
            return self._coords_data
        return {d: range(s) for d, s in zip(self.dims, self.shape, strict=False)}

    @property
    def dims(self) -> tuple[str, ...]:
        """Return dimension labels."""
        return self._dims or tuple(f"d{i}" for i in range(self.ndim))

    @property
    def shape(self) -> tuple[int, ...]:
        """Return the full shape of the view, combining all position arrays."""
        if (override := self._shape_override) is not None:
            return override

        acq_shape = self._acq_shape(self._arrays[0].shape)

        # If no position dimension, return array shape directly
        if (pos_ax := self._position_axis) is None:
            return acq_shape

        # Otherwise insert position dimension at specified axis
        return (*acq_shape[:pos_ax], len(self._arrays), *acq_shape[pos_ax:])

    @property
    def dtype(self) -> np.dtype:
        """Return the dtype of the underlying arrays."""
        return self._dtype

    @property
    def ndim(self) -> int:
        """Return the number of dimensions in the view."""
        return len(self.shape)

    def _on_coords_expanded(self, update: CoordUpdate) -> None:
        """Update live coords and shape from a high water mark event."""
        mc = update.max_coords
        new_shape = tuple(len(mc[d]) for d in self._dims)
        # Events may arrive out of order (async executor); only grow shape.
        if (old := self._shape_override) is not None and new_shape <= old:
            return
        self._coords_data = dict(mc)
        self._shape_override = new_shape
        self._coords_changed.emit()

    def __getitem__(self, key: Index | tuple[Index, ...]) -> np.ndarray:
        """Get item(s) from the view using integer and slice indexing."""
        # Normalize key to full tuple with slices
        keys = key if isinstance(key, tuple) else (key,)
        keys = keys + (slice(None),) * (self.ndim - len(keys))

        # Strict bounds check for dynamic_shape mode
        if self._strict_bounds and (live := self._shape_override) is not None:
            for i, (k, s) in enumerate(zip(keys, live, strict=False)):
                if isinstance(k, int):
                    resolved = k if k >= 0 else s + k
                    if resolved < 0 or resolved >= s:
                        raise IndexError(
                            f"Index {k} is out of bounds for live "
                            f"dimension {self._dims[i]!r} with size {s}"
                        )

        # Extract position index and array indices
        if (pos_axis := self._position_axis) is None:
            # No position dimension: single position, all indices for array
            pos_idx = 0
            array_indices = keys
        else:
            # Has position dimension: extract it from key
            pos_idx = keys[pos_axis]
            array_indices = keys[:pos_axis] + keys[pos_axis + 1 :]

        # Convert array indices to storage order
        if inv_perm := self._inv_perm:
            n_permuted = n = len(inv_perm)
            storage_idx = tuple(array_indices[i] for i in inv_perm) + array_indices[n:]
        else:
            n_permuted = len(array_indices) - 2
            storage_idx = array_indices

        # Track which non-spatial dims survived integer indexing (for transpose)
        kept_dims = tuple(not isinstance(k, int) for k in array_indices[:n_permuted])

        # Single position: return directly
        if not isinstance(pos_idx, slice):
            return self._get_from_position(pos_idx, storage_idx, kept_dims)

        # Multiple positions: stack results at adjusted position axis
        # Axis adjusts for collapsed dims (e.g., pos at 1 but dim 0 collapsed)
        results = [
            self._get_from_position(int(i), storage_idx, kept_dims)
            for i in self._resolve_position_indices(pos_idx)
        ]
        assert pos_axis is not None  # narrowed by earlier check
        stack_ax = pos_axis - sum(
            1 for i in range(pos_axis) if isinstance(keys[i], int)
        )
        return np.stack(results, axis=stack_ax)

    def _get_from_position(
        self, pos: SupportsIndex, key: tuple[Index, ...], kept: tuple[bool, ...]
    ) -> np.ndarray:
        # Normalize negative indices for backends like TensorStore
        arr = self._arrays[pos]
        normed_key = tuple(_norm_index(*x) for x in zip(key, arr.shape, strict=False))
        result = np.asarray(arr[normed_key] if normed_key else arr[:])

        if (acq_perm := self._acq_perm) and any(kept):
            # Build permutation for kept dims only
            kept_acq = tuple(acq_perm[i] for i, k in enumerate(kept) if k)
            perm = kept_acq + tuple(range(len(kept_acq), result.ndim))
            if perm != tuple(range(result.ndim)):
                result = np.transpose(result, perm)

        return result

    def _acq_shape(self, storage_shape: tuple[int, ...]) -> tuple[int, ...]:
        """Compute shape in acquisition order from storage shape."""
        if acq_perm := self._acq_perm:
            n = self._n_perm
            return tuple(storage_shape[i] for i in acq_perm) + storage_shape[n:]
        return storage_shape

    def _resolve_position_indices(self, position_index: Any) -> Iterable[int]:
        """Convert position slice or array to iterable of indices."""
        try:
            if isinstance(position_index, slice):
                return range(*position_index.indices(len(self._arrays)))
            return np.asarray(position_index).flat
        except (TypeError, ValueError) as e:  # pragma: no cover
            raise IndexError(f"Invalid position index: {position_index}") from e

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(n_positions={len(self._arrays)}, "
            f"shape={self.shape}, dtype={self.dtype})"
        )

    def __len__(self) -> int:
        """Return the length of the first dimension."""
        return self.shape[0]

    def __array__(self, dtype: Any = None, copy: bool | None = None) -> np.ndarray:
        """Present this object as a numpy array, with optional dtype conversion."""
        if copy is False:  # pragma: no cover
            raise ValueError("Zero-copy array conversion is not supported.")

        result = self[:]
        return result.astype(dtype) if dtype else result


def _norm_index(idx: Index, dim_size: int) -> Index:
    """Normalize negative indices to positive for all backends."""
    if isinstance(idx, int):
        # Convert negative to positive
        return idx if idx >= 0 else dim_size + idx

    if isinstance(idx, slice):
        # Normalize slice indices
        start, stop, step = idx.start, idx.stop, idx.step
        if start is not None and start < 0:
            start = dim_size + start
        if stop is not None and stop < 0:
            stop = dim_size + stop
        return slice(start, stop, step)
    raise TypeError(f"Invalid index type: {type(idx)}")  # pragma: no cover
