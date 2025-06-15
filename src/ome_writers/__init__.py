"""OME-TIFF and OME-ZARR writer APIs designed for microscopy acquisition."""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version
from typing import TYPE_CHECKING

from ._acquire_zarr import AcquireZarrStream
from ._dimensions import DimensionInfo
from ._stream_base import OMEStream
from ._tensorstore import TensorStoreZarrStream
from ._tiff_stream import TiffStream

if TYPE_CHECKING:
    from collections.abc import Sequence

    import numpy as np


try:
    __version__ = version("ome-writers")
except PackageNotFoundError:  # pragma: no cover
    __version__ = "uninstalled"

__all__ = [
    "AcquireZarrStream",
    "DimensionInfo",
    "OMEStream",
    "TensorStoreZarrStream",
    "TiffStream",
    "__version__",
    "create_stream",
]


def create_stream(
    path: str, dtype: np.dtype, dimensions: Sequence[DimensionInfo]
) -> OMEStream:
    """Create a stream for writing OME data using the acquire-zarr backend.

    Parameters
    ----------
    path : str
        Path to the output file or directory.
    dtype : np.dtype
        NumPy data type for the image data.
    dimensions : Sequence[DimensionInfo]
        Sequence of dimension information describing the data structure.

    Returns
    -------
    OMEStream
        A configured stream ready for writing frames.
    """
    stream = AcquireZarrStream()
    return stream.create(path, dtype, dimensions)
