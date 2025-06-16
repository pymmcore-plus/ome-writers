"""OME-TIFF and OME-ZARR writer APIs designed for microscopy acquisition."""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("ome-writers")
except PackageNotFoundError:  # pragma: no cover
    __version__ = "uninstalled"

from ._acquire_zarr import AcquireZarrStream
from ._auto import BackendName, create_stream
from ._dimensions import DimensionInfo, DimensionLabel
from ._stream_base import OMEStream
from ._tensorstore import TensorStoreZarrStream
from ._tiff_stream import TiffStream
from ._util import fake_data_for_sizes

__all__ = [
    "AcquireZarrStream",
    "BackendName",
    "DimensionInfo",
    "DimensionLabel",
    "OMEStream",
    "TensorStoreZarrStream",
    "TiffStream",
    "__version__",
    "create_stream",
    "fake_data_for_sizes",
]
