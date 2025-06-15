"""OME-TIFF and OME-ZARR writer APIs designed for microscopy acquisition."""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version
from typing import TYPE_CHECKING

from ._acquire_zarr import AcquireZarrStream
from ._dimensions import DimensionInfo
from ._stream_base import OMEStream
from ._tensorstore import TensorStoreZarrStream

if TYPE_CHECKING:
    from collections.abc import Sequence

    import numpy as np


try:
    __version__ = version("ome-writers")
except PackageNotFoundError:
    __version__ = "uninstalled"

__all__ = [
    "AcquireZarrStream",
    "DimensionInfo",
    "OMEStream",
    "TensorStoreZarrStream",
    "__version__",
]


def create_stream(
    path: str, dtype: np.dtype, dimensions: Sequence[DimensionInfo]
) -> OMEStream:
    stream = AcquireZarrStream()
    return stream.create(path, dtype, dimensions)
