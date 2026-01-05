from __future__ import annotations

import abc
from abc import abstractmethod
from pathlib import Path
from typing import TYPE_CHECKING

from typing_extensions import Self

from ome_writers._util import DimensionIndexIterator

from ._dimensions import Dimension

if TYPE_CHECKING:
    from collections.abc import Iterator, Sequence
    from types import TracebackType

    import numpy as np

    from ._dimensions import Dimension, DimensionLabel


class OMEStream(abc.ABC):
    """Abstract base class for writing streams of image frames to OME-compliant files.

    This class defines the common interface for all OME stream writers, providing
    methods for creating streams, appending frames, flushing data, and managing
    stream lifecycle. Concrete implementations handle specific file formats
    (TIFF, Zarr) and storage backends.

    The class supports context manager protocol for automatic resource cleanup
    and provides path normalization utilities for cross-platform compatibility.
    """

    @abstractmethod
    def create(
        self,
        path: str,
        dtype: np.dtype,
        dimensions: Sequence[Dimension],
        *,
        overwrite: bool = False,
    ) -> Self:
        """Create a new stream for path, dtype, and dimensions.

        Parameters
        ----------
        path : str
            Path to the output file or directory.
        dtype : np.dtype
            NumPy data type for the image data.
        dimensions : Sequence[Dimension]
            Sequence of dimension information describing the data structure.
            The order of dimensions in this sequence determines the acquisition order
            (i.e., the order in which frames will be appended to the stream).
        overwrite : bool, optional
            Whether to overwrite existing files or directories. Default is False.

        Returns
        -------
        Self
            The instance of the stream, allowing for chaining.
        """

    @abstractmethod
    def append(self, frame: np.ndarray) -> None:
        """Append a frame to the stream."""

    @abstractmethod
    def is_active(self) -> bool:
        """Return True if stream is active."""

    @abstractmethod
    def flush(self) -> None:
        """Flush pending stream writes to disk."""

    def _normalize_path(self, path: str) -> str:
        return str(Path(path).expanduser().resolve())

    @classmethod
    @abstractmethod
    def is_available(cls) -> bool:
        """Return True if this stream type is available (has needed imports)."""

    def __enter__(self) -> Self:
        """Enter the runtime context related to this object."""
        return self

    def __exit__(
        self,
        exc_type: type[BaseException],
        exc_value: BaseException,
        traceback: TracebackType,
    ) -> None:
        """Exit the runtime context related to this object."""
        if self.is_active():
            self.flush()


class MultiPositionOMEStream(OMEStream):
    """Base class for OME streams supporting multi-position imaging datasets.

    Extends OMEStream to handle complex multi-dimensional acquisitions with
    position ('p') dimensions. This class automatically manages index mapping
    and coordinates between global frame indices and backend-specific array
    keys and dimensional indices.

    Provides a default append() implementation that handles multi-position
    data routing, while requiring subclasses to implement `_write_to_backend()`
    for their specific storage mechanisms. The class separates position
    dimensions from other dimensions (time, z, channel, y, x) and creates
    appropriate indexing schemes for efficient data organization.
    """

    def __init__(self) -> None:
        # property to track number of positions
        self._num_positions = 0
        # property to track non-position dimensions in storage order (as stored on disk)
        self._storage_order_dims: Sequence[Dimension] = []
        # iterator to yield (position_key, index) tuples in acquisition order
        self._dim_iter: Iterator[tuple[int, tuple[int, ...]]] = iter(())
        # mapping from acquisition index to storage index for non-spatial dims
        self._index_mapping: tuple[int, ...] | None = None

    def _init_dimensions(
        self, dimensions: Sequence[Dimension], *, ngff_order: bool = False
    ) -> None:
        """Initialize dimensions.

        This method performs two related tasks:

        1) Define the shape and logical ordering that will be used to store the
           multi-dimensional dataset on disk. By default, acquisition order is
           preserved. If ngff_order=True, dimensions are reordered to NGFF v0.5
           specification (time, channel, space).

        2) Build an iterator that yields per-frame indices in acquisition order.
           The iterator `self._dim_iter` yields tuples `(position_key, index_tuple)`
           where `position_key` is an integer identifying the position, and
           `index_tuple` contains the indices for the non-spatial axes in acquisition
           order (e.g., `(t, c, z)` or `(c, t, z)` depending on the acquisition).
           This allows the stream to correctly place each incoming frame in the
           correct location of the storage array.

        Properties Set
        --------------
        num_positions : int
            The number of positions in the stream.
        storage_order_dims : Sequence[Dimension]
            The non-position dimensions in storage order. If ngff_order=False,
            this is acquisition order. If ngff_order=True, this is NGFF v0.5 order
            (time, channel, space).
        dim_iter : Iterator[tuple[int, tuple[int, ...]]]
            An iterator over (position_key, index_tuple) tuples in acquisition order.
        index_mapping : tuple[int, ...] | None
            Mapping from acquisition index to storage index for non-spatial dimensions.
            None if no reordering (ngff_order=False).

        Parameters
        ----------
        dimensions : Sequence[Dimension]
            Dimensions in acquisition order.
        ngff_order : bool, optional
            If True, reorder dimensions to NGFF v0.5 order (time, channel, space).
            Default is False (preserve acquisition order).
        """
        # Retrieve the number of positions from the dimensions if any, otherwise 1
        position_dims = self._get_position_dim(dimensions)
        self._num_positions = position_dims.size if position_dims else 1

        # Filter out the position dimension to get only non-position dimensions (no 'p')
        non_position_dims = [d for d in dimensions if d.label != "p"]

        if ngff_order:
            # Reorder to NGFF v0.5 and get index mapping
            self._storage_order_dims, self._index_mapping = (
                self._reorder_to_ngff_storage(non_position_dims)
            )
        else:
            # Keep the acquisition order for non-position dimensions (no 'p')
            self._storage_order_dims = list(non_position_dims)
            self._index_mapping = None

        # Extract labels of non-spatial acquisition order dims
        # Iterator should use acquisition order labels, not storage order
        acquisition_order_labels: list[DimensionLabel] = [
            d.label for d in non_position_dims if d.label not in "yx"
        ]

        # Create iterator yielding (position_key, index_tuple) in acquisition order
        self._dim_iter = iter(
            DimensionIndexIterator(dimensions, acquisition_order_labels)
        )

    @property
    def num_positions(self) -> int:
        """Return the number of positions in the stream."""
        return self._num_positions

    @property
    def storage_order_dims(self) -> Sequence[Dimension]:
        """Return the storage order dimensions."""
        return self._storage_order_dims

    @property
    def dim_iter(self) -> Iterator[tuple[int, tuple[int, ...]]]:
        """Return an iterator over (position_key, index) tuples."""
        return self._dim_iter

    def _get_position_dim(self, dimensions: Sequence[Dimension]) -> Dimension | None:
        """Return the position Dimension if it exists, else None."""
        for dim in dimensions:
            if dim.label == "p":
                return dim
        return None

    def _reorder_to_ngff_storage(
        self, non_position_dims: list[Dimension]
    ) -> tuple[list[Dimension], tuple[int, ...]]:
        """Reorder dimensions to NGFF v0.5 order and create index mapping.

        Parameters
        ----------
        non_position_dims : list[Dimension]
            Dimensions in acquisition order (excluding position dimension).

        Returns
        -------
        tuple[list[Dimension], tuple[int, ...]]
            - Reordered dimensions in NGFF v0.5 order (time, channel, space)
            - Index mapping from acquisition to storage order for non-spatial dims
        """
        # Reorder to NGFF v0.5: time, channel, space (z, y, x)
        time_dims = [d for d in non_position_dims if d.label == "t"]
        channel_dims = [d for d in non_position_dims if d.label == "c"]
        space_dims = [d for d in non_position_dims if d.label in ("z", "y", "x")]

        # Space dimensions must be in z, y, x order
        space_order = {"z": 0, "y": 1, "x": 2}
        space_dims.sort(key=lambda d: space_order.get(d.label, 3))

        # Build NGFF order
        storage_dims = time_dims + channel_dims + space_dims

        # Build index mapping for non-spatial dims
        non_spatial_acq = [d for d in non_position_dims if d.label not in ("y", "x")]
        non_spatial_storage = [d for d in storage_dims if d.label not in ("y", "x")]

        if non_spatial_acq:
            # For each storage position, find which acquisition position it comes from
            index_mapping = tuple(
                non_spatial_acq.index(dim) for dim in non_spatial_storage
            )
        else:
            index_mapping = ()

        return storage_dims, index_mapping

    @abstractmethod
    def _write_to_backend(
        self, position_key: str, index: tuple[int, ...], frame: np.ndarray
    ) -> None:
        """Backend-specific write implementation.

        Parameters
        ----------
        position_key : str
            The key for the position in the backend (e.g., Zarr group).
        index : tuple[int, ...]
            A tuple of indices for the non-spatial dimensions in acquisition order
            (e.g., (t, c, z) or (c, t, z) depending on how dimensions were defined).
            Can be used directly for array indexing.
        frame : np.ndarray
            The frame data to write.

        Raises
        ------
        RuntimeError
            If the stream is not active or uninitialized.
        """

    def append(self, frame: np.ndarray) -> None:
        if not self.is_active():
            msg = "Stream is closed or uninitialized. Call create() first."
            raise RuntimeError(msg)
        position_key, acq_index = next(self._dim_iter)

        # Transpose index from acquisition order to storage order if needed
        if self._index_mapping:
            storage_index = tuple(acq_index[i] for i in self._index_mapping)
        else:
            storage_index = acq_index

        self._write_to_backend(str(position_key), storage_index, frame)
