"""OME-TIFF and OME-ZARR writer APIs designed for microscopy acquisition."""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("ome-writers")
except PackageNotFoundError:  # pragma: no cover
    __version__ = "uninstalled"

from ._stream import BackendName, create_stream
from .schema import (
    AcquisitionSettings,
    Dimension,
    Plate,
    Position,
    PositionDimension,
    StandardAxis,
    dims_from_standard_axes,
)

__all__ = [
    "AcquisitionSettings",
    "BackendName",
    "Dimension",
    "Plate",
    "Position",
    "PositionDimension",
    "StandardAxis",
    "__version__",
    "create_stream",
    "dims_from_standard_axes",
]
