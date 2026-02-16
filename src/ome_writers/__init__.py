"""OME-TIFF and OME-ZARR writer APIs designed for microscopy acquisition."""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("ome-writers")
except PackageNotFoundError:  # pragma: no cover
    __version__ = "uninstalled"


from ome_writers._schema import (
    AcquisitionSettings,
    Channel,
    Dimension,
    OmeTiffFormat,
    OmeZarrFormat,
    Plate,
    Position,
    PositionDimension,
    StandardAxis,
    dims_from_standard_axes,
)
from ome_writers._stream import CoordUpdate, OMEStream, create_stream
from ome_writers._useq import dims_from_useq, useq_to_acquisition_settings  # ty: ignore

__all__ = [
    "AcquisitionSettings",
    "Channel",
    "CoordUpdate",
    "Dimension",
    "OMEStream",
    "OmeTiffFormat",
    "OmeZarrFormat",
    "Plate",
    "Position",
    "PositionDimension",
    "StandardAxis",
    "__version__",
    "create_stream",
    "dims_from_standard_axes",
    "dims_from_useq",
    "useq_to_acquisition_settings",
]
