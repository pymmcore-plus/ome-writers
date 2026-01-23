"""OME-TIFF and OME-ZARR writer APIs designed for microscopy acquisition."""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version
from typing import TYPE_CHECKING

try:
    __version__ = version("ome-writers")
except PackageNotFoundError:  # pragma: no cover
    __version__ = "uninstalled"

if TYPE_CHECKING:
    # may be imported at top level, but only for type checking
    from ome_writers._stream import OMEStream as OMEStream

from ome_writers._schema import (
    AcquisitionSettings,
    Dimension,
    Plate,
    Position,
    PositionDimension,
    StandardAxis,
    dims_from_standard_axes,
)
from ome_writers._stream import OMEStream, create_stream
from ome_writers._util import dims_from_useq

__all__ = [
    "AcquisitionSettings",
    "Dimension",
    "OMEStream",
    "Plate",
    "Position",
    "PositionDimension",
    "StandardAxis",
    "__version__",
    "create_stream",
    "dims_from_standard_axes",
    "dims_from_useq",
]
