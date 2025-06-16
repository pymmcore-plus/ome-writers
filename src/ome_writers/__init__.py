"""OME-TIFF and OME-ZARR writer APIs designed for microscopy acquisition."""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version

from ._acquire_zarr import AcquireZarrStream
from ._auto import create_stream
from ._dimensions import DimensionInfo
from ._stream_base import OMEStream
from ._tensorstore import TensorStoreZarrStream
from ._tiff_stream import TiffStream

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
