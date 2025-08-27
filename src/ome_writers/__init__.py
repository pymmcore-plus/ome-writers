"""OME-TIFF and OME-ZARR writer APIs designed for microscopy acquisition."""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("ome-writers")
except PackageNotFoundError:  # pragma: no cover
    __version__ = "uninstalled"

from ._auto import BackendName, create_stream
from ._dimensions import Dimension, DimensionLabel, UnitTuple
from ._stream_base import OMEStream
from ._util import dims_from_useq, fake_data_for_sizes
from .backends._acquire_zarr import AcquireZarrStream
from .backends._tensorstore import TensorStoreZarrStream
from .backends._tifffile import TifffileStream

__all__ = [
    "AcquireZarrStream",
    "BackendName",
    "Dimension",
    "DimensionLabel",
    "OMEStream",
    "TensorStoreZarrStream",
    "TifffileStream",
    "UnitTuple",
    "__version__",
    "create_stream",
    "dims_from_useq",
    "fake_data_for_sizes",
]
