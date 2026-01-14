"""OME-TIFF and OME-ZARR writer APIs designed for microscopy acquisition."""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("ome-writers")
except PackageNotFoundError:  # pragma: no cover
    __version__ = "uninstalled"

from ._stream import BackendName, create_stream
from ._util import fake_data_for_sizes
from .schema import (
    AcquisitionSettings,
    ArraySettings,
    Dimension,
    Plate,
    Position,
    PositionDimension,
)

__all__ = [
    "AcquisitionSettings",
    "ArraySettings",
    "BackendName",
    "Dimension",
    "Plate",
    "Position",
    "PositionDimension",
    "__version__",
    "create_stream",
    "fake_data_for_sizes",
]
