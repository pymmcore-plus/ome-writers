import abc
from abc import abstractmethod
from collections.abc import Sequence
from pathlib import Path
from typing import Self

import numpy as np

from ome_writers.dimensions import DimensionInfo


class OMEStream(abc.ABC):
    @abstractmethod
    def create(
        self, path: str, dtype: np.dtype, dimensions: Sequence[DimensionInfo]
    ) -> Self: ...
    @abstractmethod
    def append(self, frame: np.ndarray) -> None: ...
    @abstractmethod
    def is_active(self) -> bool: ...
    @abstractmethod
    def flush(self) -> None: ...
    def normalize_path(self, path: str) -> str:
        return str(Path(path).expanduser().resolve())
