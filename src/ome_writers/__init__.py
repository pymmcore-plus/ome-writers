"""OME-TIFF and OME-ZARR writer APIs designed for microscopy acquisition."""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("ome-writers")
except PackageNotFoundError:  # pragma: no cover
    __version__ = "uninstalled"

from ome_writers._stream import create_stream
from ome_writers._util import dims_from_useq
from ome_writers.schema import (
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
    "Dimension",
    "Plate",
    "Position",
    "PositionDimension",
    "StandardAxis",
    "__version__",
    "create_stream",
    "dims_from_standard_axes",
    "dims_from_useq",
]
