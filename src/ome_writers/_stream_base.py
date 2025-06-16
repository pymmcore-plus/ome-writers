import abc
from abc import abstractmethod
from collections.abc import Sequence
from itertools import product
from pathlib import Path
from typing import Self

import numpy as np

from ._dimensions import DimensionInfo


class OMEStream(abc.ABC):
    """Base API for writing a stream of frames to an OME file (tiff or zarr)."""

    @abstractmethod
    def create(
        self, path: str, dtype: np.dtype, dimensions: Sequence[DimensionInfo]
    ) -> Self:
        """Create a new stream for path, dtype, and dimensions."""

    @abstractmethod
    def append(self, frame: np.ndarray) -> None:
        """Append a frame to the stream."""

    @abstractmethod
    def is_active(self) -> bool:
        """Return True if stream is active."""

    @abstractmethod
    def flush(self) -> None:
        """Flush to disk."""

    def _normalize_path(self, path: str) -> str:
        return str(Path(path).expanduser().resolve())

    @classmethod
    @abstractmethod
    def is_available(cls) -> bool:
        """Return True if this stream type is available (has needed imports)."""


class MultiPositionOMEStream(OMEStream):
    """Subclass for OME streams that support multiple positions.

    This class provides a default "append()" implementation that handles
    multi-position data by managing indices and position dimensions, subclasses
    must now implement the "_write_to_backend()" method to handle
    backend-specific writing logic.
    """

    def __init__(self) -> None:
        # dimension info for position dimension, if any
        self._position_dim: DimensionInfo | None = None
        # A mapping of indices to (array_key, non-position index)
        self._indices: dict[int, tuple[str, tuple[int, ...]]] = {}
        self._append_count = 0
        self._num_positions = 0
        self._non_position_dims: Sequence[DimensionInfo] = []

    def _init_positions(
        self, dimensions: Sequence[DimensionInfo]
    ) -> tuple[int, Sequence[DimensionInfo]]:
        """Initialize position tracking and return num_positions, non_position_dims."""
        # Separate position dimension from other dimensions
        position_dims = [d for d in dimensions if d.label == "p"]
        non_position_dims = [d for d in dimensions if d.label != "p"]
        num_positions = position_dims[0].size if position_dims else 1
        non_p_ranges = [range(d.size) for d in non_position_dims if d.label not in "yx"]
        range_iter = enumerate(product(range(num_positions), *non_p_ranges))

        self._position_dim = position_dims[0] if position_dims else None
        self._indices = {i: (str(pos), tuple(idx)) for i, (pos, *idx) in range_iter}
        self._append_count = 0
        self._num_positions = num_positions
        self._non_position_dims = non_position_dims

        return num_positions, non_position_dims

    @abstractmethod
    def _write_to_backend(
        self, array_key: str, index: tuple[int, ...], frame: np.ndarray
    ) -> None:
        """Backend-specific write implementation."""

    def append(self, frame: np.ndarray) -> None:
        if not self.is_active():
            msg = "Stream is closed or uninitialized. Call create() first."
            raise RuntimeError(msg)
        array_key, index = self._indices[self._append_count]
        self._append_count += 1
        self._write_to_backend(array_key, index, frame)
