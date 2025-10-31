"""OME-TIFF and OME-ZARR writer APIs designed for microscopy acquisition."""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("ome-writers")
except PackageNotFoundError:  # pragma: no cover
    __version__ = "uninstalled"

from ._auto import BackendName, create_stream
from ._dimensions import (
    Dimension,
    DimensionLabel,
    UnitTuple,
    dims_to_ome,
    dims_to_yaozarrs_v5,
    ome_meta_v5,
)
from ._plate import (
    Plate,
    PlateAcquisition,
    WellPosition,
    plate_to_acquire_zarr,
    plate_to_ome_types,
    plate_to_yaozarrs_v5,
)
from ._stream_base import OMEStream
from ._util import dims_from_useq, fake_data_for_sizes, plate_from_useq
from .backends._acquire_zarr import AcquireZarrStream
from .backends._tensorstore import TensorStoreZarrStream
from .backends._tifffile import TifffileStream

__all__ = [
    "AcquireZarrStream",
    "BackendName",
    "Dimension",
    "DimensionLabel",
    "OMEStream",
    "Plate",
    "PlateAcquisition",
    "TensorStoreZarrStream",
    "TifffileStream",
    "UnitTuple",
    "WellPosition",
    "__version__",
    "create_stream",
    "dims_from_useq",
    "dims_to_ngff_v5",
    "dims_to_ome",
    "dims_to_yaozarrs_v5",
    "fake_data_for_sizes",
    "ome_meta_v5",
    "plate_from_useq",
    "plate_to_acquire_zarr",
    "plate_to_ome_types",
    "plate_to_yaozarrs_v5",
]
