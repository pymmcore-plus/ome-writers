from __future__ import annotations

import abc
from abc import abstractmethod
from pathlib import Path
from typing import TYPE_CHECKING

from typing_extensions import Self

from ome_writers._util import (
    DimensionIndexIterator,
    reorder_to_ome_ngff,
)

from ._dimensions import Dimension

if TYPE_CHECKING:
    from collections.abc import Iterator, Sequence
    from types import TracebackType

    import numpy as np

    from ._dimensions import Dimension, DimensionLabel
    from ._plate import Plate


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
        plate: Plate | None = None,
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
        plate : Plate | None, optional
            Optional plate metadata for organizing multi-well acquisitions.
            If provided, the store will be structured as a plate with wells.
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
        # property to track acquisition order dimensions
        self._acquisition_order_dims: Sequence[Dimension] = []
        # property to track non-position dimensions in storage order (as stored on disk)
        self._storage_order_dims: Sequence[Dimension] = []
        # iterator to yield (position_key, index) tuples in acquisition order
        self._dim_iter: Iterator[tuple[int, tuple[int, ...]]] = iter(())
        # plate metadata
        self._plate: Plate | None = None

    def _init_dimensions(
        self, dimensions: Sequence[Dimension], enforce_ome_order: bool = True
    ) -> None:
        """Initialize dimensions.

        This method performs two related tasks:

        1) Define the shape and logical ordering that will be used to store the
           multi-dimensional dataset on disk.

           If `enforce_ome_order` is True (default), non-position axes are reordered to
           match the OME-NGFF TCZYX storage order and the shape of the data to disk will
           always be TCZXY, no matter the acquisition order.

           If False, the acquisition order is preserved and the shape of the data
           written to disk will follow the acquisition order (this is useful for
           sequential writers such as OME-TIFF where enforcing TCZYX is not strictly
           required).

        2) Build an iterator that yields per-frame indices in acquisition order but
           formatted for storage indexing. The iterator `self._dim_iter` yields tuples
           `(position_key, index_tuple)` where `position_key` is an integer identifying
           the position, and `index_tuple` contains the storage indices for the
           non-spatial axes (e.g., `(t, c, z)`). This allows the stream to correctly
           place each incoming frame in the correct location of the storage array even
           when the storage ordering differs from the acquisition ordering.

        Properties
        ----------
        num_positions : int
            The number of positions in the stream.
        storage_order_dims : Sequence[Dimension]
            The non-position dimensions in storage order (as stored on disk - TCZYX if
            `enforce_ome_order` is True).
        dim_iter : Iterator[tuple[int, tuple[int, ...]]]
            An iterator over (position_key, index_tuple) tuples in acquisition order,
            where index_tuple is ordered according to storage_order_dims.

        Parameters
        ----------
        dimensions : Sequence[Dimension]
            Dimensions in acquisition order.
        enforce_ome_order : bool, optional
            If True, reorder dimensions to OME storage order (TCZYX) so that the data
            saved to disk will follow OME-NGFF order. This is required for OME-NGFF
            (Zarr). If False, keep acquisition order. Default is True.
        """
        # Store the acquisition order dimensions
        self._acquisition_order_dims = list(dimensions)

        # Retrieve the number of positions from the dimensions if any, otherwise 1
        position_dims = self._get_position_dim(dimensions)
        self._num_positions = position_dims.size if position_dims else 1

        # Filter out the position dimension to get only non-position dimensions (no 'p')
        non_position_dims = [d for d in dimensions if d.label != "p"]

        # If enforcing OME order, reorder non-position dims (no 'p') to OME-NGFF TCZYX
        if enforce_ome_order:
            self._storage_order_dims = reorder_to_ome_ngff(list(non_position_dims))
        # Otherwise, keep the acquisition order for non-position dimensions (no 'p')
        else:
            self._storage_order_dims = list(non_position_dims)

        # Extract labels of storage order dims, excluding spatial dims y and x
        storage_order_labels: list[DimensionLabel] = [
            d.label for d in self._storage_order_dims if d.label not in "yx"
        ]

        # Create iterator yielding (position_key, index_tuple) in acquisition order
        self._dim_iter = iter(DimensionIndexIterator(dimensions, storage_order_labels))

    @property
    def num_positions(self) -> int:
        """Return the number of positions in the stream."""
        return self._num_positions

    @property
    def acquisition_order_dims(self) -> Sequence[Dimension]:
        """Return the dimensions in acquisition order."""
        return self._acquisition_order_dims

    @property
    def storage_order_dims(self) -> Sequence[Dimension]:
        """Return the non-position dimensions in storage order (as stored on disk)."""
        return self._storage_order_dims

    @property
    def dim_iter(self) -> Iterator[tuple[int, tuple[int, ...]]]:
        """Return an iterator over (position_key, index) tuples."""
        return self._dim_iter

    @property
    def plate(self) -> Plate | None:
        """Return the plate metadata if available, else None."""
        return self._plate

    def _get_position_dim(self, dimensions: Sequence[Dimension]) -> Dimension | None:
        """Return the position Dimension if it exists, else None."""
        for dim in dimensions:
            if dim.label == "p":
                return dim
        return None

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
            A tuple of storage indices for the non-spatial dimensions (e.g., (t, c, z))
            in storage order. Can be used directly for array indexing.
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
        position_key, index = next(self._dim_iter)
        self._write_to_backend(str(position_key), index, frame)
