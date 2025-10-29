from __future__ import annotations

import abc
from abc import abstractmethod
from pathlib import Path
from typing import TYPE_CHECKING

from typing_extensions import Self

if TYPE_CHECKING:
    from collections.abc import Sequence
    from types import TracebackType

    import numpy as np

    from ome_writers._dimensions import DimensionLabel

    from ._dimensions import Dimension


OME_NGFF_ORDER = {"t": 0, "c": 1, "z": 2, "y": 3, "x": 4}


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
        # dimension info for position dimension, if any
        self._position_dim: Dimension | None = None
        # number of times append() has been called
        self._append_count = 0
        # number of positions in the stream
        self._num_positions = 0
        # non-position dimensions
        # (e.g. time, z, c, y, x) that are not
        self._non_position_dims: Sequence[Dimension] = []

    def _init_positions(
        self, dimensions: Sequence[Dimension], *, enforce_ome_order: bool = True
    ) -> tuple[int, Sequence[Dimension]]:
        """Initialize position tracking.

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

        2) Build a mapping from the acquisition frame sequence (the order in which the
           frames will be appended via the `append()` method) to the per-frame storage
           indices inside the on-disk array(s). This mapping allows the stream to
           correctly place each incoming frame in the correct location of the
           storage-shaped array even when the storage ordering differs from the
           acquisition ordering. The mapping is stored in `self._indices` as
           a dictionary mapping a frame sequence number -> (array_key, index), where
           `array_key` identifies the position/group and `index` is a small named-tuple
           (e.g. with attributes like `.t`, `.c`, `.z`) that can be used for array
           indexing.

        Parameters
        ----------
        dimensions : Sequence[Dimension]
            Dimensions in acquisition order.
        enforce_ome_order : bool, optional
            If True, reorder dimensions to OME storage order (TCZYX) so that the data
            saved to disk will follow OME-NGFF order. This is required for OME-NGFF
            (Zarr). If False, keep acquisition order. Default is True.

        Returns
        -------
        tuple[int, Sequence[Dimension]]
            The number of positions and the non-position dimensions in the
            appropriate storage order (if enforce_ome_order is True, OME-NGFF order,
            otherwise acquisition order). This tuple will be used by each backend
            to create arrays with the correct dimension ordering.
        """
        from ome_writers._util import DimensionIndexIterator

        # Separate position dimension from other dimensions
        position_dims = [d for d in dimensions if d.label == "p"]
        non_position_dims_acq_order = [d for d in dimensions if d.label != "p"]
        num_positions = position_dims[0].size if position_dims else 1

        # Determine storage order based on enforce_ome_order flag
        if enforce_ome_order:
            # Reorder non-position dimensions to OME storage order (TCZYX)
            # This is required for OME-NGFF (Zarr) spec compliance (but not for
            # acquire-zarr at the moment)
            non_position_dims_storage_order = self._reorder_to_ome_ngff(
                non_position_dims_acq_order
            )
        else:
            # Keep acquisition order (for TIFF)
            non_position_dims_storage_order = non_position_dims_acq_order

        # Use DimensionIndexIterator to generate indices in acquisition order
        # but formatted for storage order
        storage_order_labels: list[DimensionLabel] = [
            d.label for d in non_position_dims_storage_order if d.label not in "yx"
        ]

        # Create iterator that yields indices in acquisition order but with fields
        # matching storage order
        # e.g. [FrameIndex(p=0, t=0, c=0, z=0), FrameIndex(p=0, t=0, c=1, z=0), ...]
        self._dim_iter = iter(
            DimensionIndexIterator(
                dimensions,  # acquisition order
                output_order=storage_order_labels,  # storage order for output
            )
        )

        self._position_dim = position_dims[0] if position_dims else None
        self._append_count = 0
        self._num_positions = num_positions
        self._non_position_dims = non_position_dims_storage_order

        return num_positions, non_position_dims_storage_order

    def _reorder_to_ome_ngff(self, dims_order: list[Dimension]) -> list[Dimension]:
        """Reorder dimensions to OME-NGFF TCZYX order."""
        dims_order.sort(key=lambda d: OME_NGFF_ORDER.get(d.label, 5))
        return dims_order

    @abstractmethod
    def _write_to_backend(
        self, array_key: str, index: tuple[int, ...], frame: np.ndarray
    ) -> None:
        """Backend-specific write implementation.

        Parameters
        ----------
        array_key : str
            The key for the position in the backend (e.g., Zarr group).
        index : tuple[int, ...]
            A named tuple with the storage indices. The tuple has attributes
            like .t, .c, .z etc. based on which dimensions are present.
            Can be used directly for array indexing or converted to regular tuple.
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
        self._append_count += 1
