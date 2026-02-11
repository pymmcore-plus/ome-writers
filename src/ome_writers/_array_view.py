"""Experimental multi-position array view."""

from __future__ import annotations

from concurrent.futures import Executor, Future
from typing import TYPE_CHECKING, Any, cast

import numpy as np

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence

    from typing_extensions import Self

    from ome_writers._backends._backend import ArrayLike
    from ome_writers._stream import OMEStream


class AcquisitionView:
    """Read-only view presenting multiple position arrays as a single array.

    This class provides an async, read-only view onto an ongoing
    acquisition stream. It combines all underlying arrays back into a single
    higher-dimensional array with the same shape *and order* as the original
    `AcquisitionSettings.dimensions`. Use it as you would a normal numpy array,
    but note that it only supports:

    - `__getitem__` with integer and slice indexing (returns `Future[np.ndarray]`,
      call `.result()` to block and retrieve data)
    - `__array__` for conversion to a numpy array (blocks internally)
    - `shape`, `dtype`, and `ndim` properties

    NOTE:
    - Caller provides and manages the `Executor` (e.g., `ThreadPoolExecutor`)
    - If no executor provided, requests are synchronous (wrapped futures)

    Accessing not-yet-written positions or frames should return zeros (or the
    fill-value of the underlying arrays, if specified).

    Parameters
    ----------
    arrays : Sequence[ArrayLike]
        Sequence of array-like objects to view.
    position_axis : int | None
        Axis index for the position dimension. None means no position dimension.
    acquisition_order_perm : tuple[int, ...] | None
        Permutation to convert storage order to acquisition order.
    dimension_labels : tuple[str, ...]
        Labels for each dimension.
    executor : Executor | None
        Optional executor for async reads. If None, reads are synchronous
        (wrapped in completed futures). Caller is responsible for executor
        lifecycle management.

    !!! warning
        It is *strongly discouraged* to materialize the entire view as a single array
        (e.g., via `view[:]` or `np.asarray(view)`).  It is intended as a preview
        onto individual frames at a time.
    """

    __slots__ = (
        "_acq_perm",
        "_arrays",
        "_dims",
        "_executor",
        "_inv_perm",
        "_n_perm",
        "_position_axis",
    )

    @classmethod
    def from_stream(
        cls, stream: OMEStream, *, executor: Executor | None = None
    ) -> Self:
        """Create view directly from OMEStream.

        Parameters
        ----------
        stream : OMEStream
            The stream to create a view from.
        executor : Executor | None
            Optional executor for async reads. If None, reads are synchronous
            (wrapped in completed futures). Caller manages executor lifecycle.
        """
        if stream.closed:
            raise NotImplementedError(
                "Creating a view on a closed stream is not currently supported."
            )

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

        return cls(
            stream._backend.get_arrays(),
            position_axis=settings.position_dimension_index,
            acquisition_order_perm=acquisition_perm,
            dimension_labels=[d.name for d in settings.dimensions],
            executor=executor,
        )

    def __init__(
        self,
        arrays: Sequence[ArrayLike],
        *,
        position_axis: int | None = 0,
        acquisition_order_perm: tuple[int, ...] | None = None,
        dimension_labels: tuple[str, ...] = (),
        executor: Executor | None = None,
    ) -> None:
        if not arrays:  # pragma: no cover
            raise ValueError("Arrays list cannot be empty")

        # NB:
        # we could potentially apply the acquisition_order_perm HERE, and remove
        # logic in __getitem__.  But that puts more demands on the ArrayLike object
        # ... let's explore that later.
        self._arrays = list(arrays)
        self._acq_perm = acquisition_order_perm
        self._dims = dimension_labels
        self._executor = executor

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

    @property
    def dims(self) -> tuple[str, ...]:
        """Return dimension labels."""
        return self._dims or tuple(f"d{i}" for i in range(self.ndim))

    @property
    def shape(self) -> tuple[int, ...]:
        acq_shape = self._acq_shape(self._arrays[0].shape)

        # If no position dimension, return array shape directly
        if (pos_ax := self._position_axis) is None:
            return acq_shape

        # Otherwise insert position dimension at specified axis
        return (*acq_shape[:pos_ax], len(self._arrays), *acq_shape[pos_ax:])

    @property
    def dtype(self) -> Any:
        return getattr(self._arrays[0], "dtype", None)

    @property
    def ndim(self) -> int:
        return len(self.shape)

    def __getitem__(self, key: Any) -> Future[np.ndarray]:
        """Get array slice asynchronously.

        Returns a Future that will contain the requested array slice. Call
        `.result()` to block and retrieve the data, or `.cancel()` to cancel
        the request before it completes.

        Parameters
        ----------
        key : Any
            Index or slice specification.

        Returns
        -------
        Future[np.ndarray]
            Future containing the requested array slice.
        """
        if self._executor is None:
            # Synchronous fallback: execute immediately and wrap in completed future
            result = self._getitem_sync(key)
            future: Future[np.ndarray] = Future()
            future.set_result(result)
            return future

        # Async: submit to executor
        return self._executor.submit(self._getitem_sync, key)

    def _getitem_sync(self, key: Any) -> np.ndarray:
        """Synchronous implementation of array slicing."""
        # Normalize key to full tuple with slices
        if not isinstance(key, tuple):
            key = (key,)
        key = key + (slice(None),) * (self.ndim - len(key))

        # Extract position index and array indices
        if (pos_axis := self._position_axis) is None:
            # No position dimension: single position, all indices for array
            pos_idx = 0
            array_indices = key
        else:
            # Has position dimension: extract it from key
            pos_idx = key[pos_axis]
            array_indices = key[:pos_axis] + key[pos_axis + 1 :]

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
        if isinstance(pos_idx, int):
            return self._get(pos_idx, storage_idx, kept_dims)

        # Multiple positions: stack results at adjusted position axis
        # Axis adjusts for collapsed dims (e.g., pos at 1 but dim 0 collapsed)
        pos_axis = cast("int", pos_axis)
        position_range = self._resolve_position_indices(pos_idx)
        results = [self._get(int(i), storage_idx, kept_dims) for i in position_range]
        stack_ax = pos_axis - sum(1 for i in range(pos_axis) if isinstance(key[i], int))
        return np.stack(results, axis=stack_ax)

    def _get(self, pos: int, key: tuple, kept: tuple[bool, ...]) -> np.ndarray:
        # Normalize negative indices for backends like TensorStore
        arr = self._arrays[pos]
        normed_key = tuple(
            _norm_index(idx, dim_size)
            for idx, dim_size in zip(key, arr.shape, strict=False)
        )
        result = np.asarray(arr[normed_key] if normed_key else arr[:])

        if (acq_perm := self._acq_perm) and any(kept):
            # Build permutation for kept dims only
            kept_idx = [i for i, k in enumerate(kept) if k]
            kept_acq = tuple(acq_perm[i] for i in kept_idx)
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
        return self.shape[0]

    def __array__(self, dtype: Any = None, copy: bool | None = None) -> np.ndarray:
        """Present this object as a numpy array, with optional dtype conversion.

        This method blocks until the entire array is loaded. Use with caution on
        large datasets.
        """
        if copy is False:  # pragma: no cover
            raise ValueError("Zero-copy array conversion is not supported.")

        result = self[:].result()  # Block for the future
        return result.astype(dtype) if dtype else result


def _norm_index(idx: int | slice, dim_size: int) -> int | slice:
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
