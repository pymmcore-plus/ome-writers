import abc
from abc import abstractmethod
from collections.abc import Sequence
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
