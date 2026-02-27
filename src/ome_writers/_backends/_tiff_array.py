"""TIFF-backed ArrayLike objects for live and finalized viewing.

These classes implement the ArrayLike protocol (shape, dtype, __getitem__)
and replace the previous zarr-based approach (LiveTiffStore + zarr.Array)
for reading TIFF data during and after acquisition.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from typing import Any

    import tifffile

    from ome_writers._backends._tifffile import WriterThread
    from ome_writers._schema import Dimension


def _compute_strides(dims: tuple[int, ...]) -> tuple[int, ...]:
    """Compute strides for row-major (C) ordering.

    Examples
    --------
    >>> _compute_strides((10, 5, 2))
    (10, 2, 1)
    """
    if not dims:
        return ()

    strides = [1]
    for i in reversed(range(len(dims) - 1)):
        strides.append(strides[-1] * dims[i + 1])

    return tuple(reversed(strides))


def _tiff_getitem(
    shape: tuple[int, ...],
    n_index: int,
    frame_shape: tuple[int, ...],
    dtype: np.dtype,
    read_frame: Any,
    key: Any,
) -> np.ndarray:
    """Shared indexing implementation for TIFF-backed array-like objects.

    Maps numpy-style integer/slice indexing to individual frame reads, then
    assembles the result.

    Parameters
    ----------
    shape
        Full array shape (index dims + spatial dims).
    n_index
        Number of non-spatial (index) dimensions.
    frame_shape
        Shape of a single frame (Y, X).
    dtype
        NumPy dtype.
    read_frame
        Callable(flat_idx: int) -> np.ndarray of *frame_shape*.
    key
        Index key from ``__getitem__``.
    """
    ndim = len(shape)

    if not isinstance(key, tuple):
        key = (key,)
    key = key + (slice(None),) * (ndim - len(key))

    # Expand non-spatial index dimensions into ranges
    idx_ranges: list[list[int]] = []
    for i in range(n_index):
        k = key[i]
        if isinstance(k, int):
            v = k if k >= 0 else shape[i] + k
            idx_ranges.append([v])
        elif isinstance(k, slice):
            idx_ranges.append(list(range(*k.indices(shape[i]))))
        else:
            raise TypeError(f"Unsupported index type: {type(k)}")  # pragma: no cover

    strides = _compute_strides(shape[:n_index]) if n_index else ()

    # Allocate buffer: selected index dims + full spatial dims
    buf_shape = tuple(len(r) for r in idx_ranges) + frame_shape
    buf = np.zeros(buf_shape, dtype=dtype)

    idx_out_shape = tuple(len(r) for r in idx_ranges)
    for out_idx in np.ndindex(idx_out_shape) if idx_out_shape else [()]:
        flat = sum(idx_ranges[i][out_idx[i]] * strides[i] for i in range(n_index))
        buf[out_idx] = read_frame(flat)

    # Build final key: squeeze integer-indexed dims + apply spatial slicing
    buf_key: list[Any] = []
    for i in range(n_index):
        buf_key.append(0 if isinstance(key[i], int) else slice(None))
    for i in range(n_index, ndim):
        buf_key.append(key[i])

    return buf[tuple(buf_key)]


class LiveTiffArray:
    """ArrayLike backed by a TIFF file being written (or finalized uncompressed).

    Reads raw frame data at calculated byte offsets.  Only works for
    ``contiguous=True`` (uncompressed) TIFF writes.

    The ``shape`` property is dynamic: for unbounded first dimensions it
    derives the outer extent from ``WriterThread.frames_written``, eliminating
    the previous hard-coded magic number (1000).

    Parameters
    ----------
    writer_thread
        Reference to the ``WriterThread`` writing (or that wrote) this file.
    file_path
        Path to the TIFF file.
    storage_dims
        Dimension descriptors in storage order (including Y, X).
    dtype
        NumPy dtype string.
    """

    __slots__ = (
        "_dtype_obj",
        "_frame_nbytes",
        "_frame_shape",
        "_index_counts",
        "_inner_product",
        "_n_index",
        "_path",
        "_thread",
    )

    def __init__(
        self,
        writer_thread: WriterThread,
        file_path: str,
        storage_dims: tuple[Dimension, ...],
        dtype: str,
    ) -> None:
        self._thread = writer_thread
        self._path = file_path
        self._index_counts = tuple(d.count for d in storage_dims[:-2])
        self._frame_shape = tuple(d.count or 1 for d in storage_dims[-2:])
        self._dtype_obj = np.dtype(dtype)
        self._n_index = len(storage_dims) - 2
        self._frame_nbytes = math.prod(self._frame_shape) * self._dtype_obj.itemsize

        # Pre-compute inner product for unbounded shape calculation
        inner = self._index_counts[1:] if len(self._index_counts) > 1 else ()
        self._inner_product = math.prod(c for c in inner if c is not None) or 1

    @property
    def shape(self) -> tuple[int, ...]:
        """Full array shape, dynamically derived for unbounded dims."""
        counts = list(self._index_counts)
        if counts and counts[0] is None:
            with self._thread.state_lock:
                nf = self._thread.frames_written
            # Ceiling division: how many complete outer slices?
            counts[0] = -(-nf // self._inner_product) if nf > 0 else 0
        return tuple(c if c is not None else 1 for c in counts) + self._frame_shape

    @property
    def dtype(self) -> np.dtype:
        return self._dtype_obj

    def _read_frame(self, flat_idx: int) -> np.ndarray:
        """Read a single frame by its flat index."""
        with self._thread.state_lock:
            nf = self._thread.frames_written
            data_offset = self._thread.data_offset

        if data_offset is None or flat_idx >= nf:
            return np.zeros(self._frame_shape, dtype=self._dtype_obj)

        pos = data_offset + flat_idx * self._frame_nbytes
        with open(self._path, "rb") as f:
            f.seek(pos)
            data = f.read(self._frame_nbytes)

        if len(data) < self._frame_nbytes:  # pragma: no cover
            return np.zeros(self._frame_shape, dtype=self._dtype_obj)

        return np.frombuffer(data, dtype=self._dtype_obj).reshape(self._frame_shape)

    def __getitem__(self, key: Any) -> np.ndarray:
        return _tiff_getitem(
            self.shape,
            self._n_index,
            self._frame_shape,
            self._dtype_obj,
            self._read_frame,
            key,
        )


class FinalizedTiffArray:
    """ArrayLike backed by a completed TIFF file, read via tifffile.

    Supports compressed TIFFs by reading through tifffile's page-level API.

    Parameters
    ----------
    tiff_file
        An open ``tifffile.TiffFile`` instance.  The caller is responsible for
        closing it (typically via ``weakref.finalize``).
    shape
        Declared full shape of the array.
    dtype
        NumPy dtype string.
    """

    __slots__ = (
        "__weakref__",
        "_dtype_obj",
        "_frame_shape",
        "_n_index",
        "_shape",
        "_tf",
    )

    def __init__(
        self,
        tiff_file: tifffile.TiffFile,
        shape: tuple[int, ...],
        dtype: str,
    ) -> None:
        self._tf = tiff_file
        self._shape = shape
        self._dtype_obj = np.dtype(dtype)
        self._frame_shape = shape[-2:]
        self._n_index = len(shape) - 2

    @property
    def shape(self) -> tuple[int, ...]:
        return self._shape

    @property
    def dtype(self) -> np.dtype:
        return self._dtype_obj

    def _read_frame(self, flat_idx: int) -> np.ndarray:
        """Read a single frame by its flat index via tifffile pages."""
        pages = self._tf.pages
        if flat_idx < len(pages):
            return pages[flat_idx].asarray()
        return np.zeros(self._frame_shape, dtype=self._dtype_obj)  # pragma: no cover

    def __getitem__(self, key: Any) -> np.ndarray:
        return _tiff_getitem(
            self._shape,
            self._n_index,
            self._frame_shape,
            self._dtype_obj,
            self._read_frame,
            key,
        )
