"""OME-TIFF and OME-ZARR writer APIs designed for microscopy acquisition."""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("ome-writers")
except PackageNotFoundError:  # pragma: no cover
    __version__ = "uninstalled"

from ._auto import BackendName, create_stream
from ._dimensions import Dimension, DimensionLabel, UnitTuple
from ._util import dims_from_useq, fake_data_for_sizes

__all__ = [
    "BackendName",
    "Dimension",
    "DimensionLabel",
    "UnitTuple",
    "__version__",
    "create_stream",
    "dims_from_useq",
    "fake_data_for_sizes",
]
