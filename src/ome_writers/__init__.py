"""OME-TIFF and OME-ZARR writer APIs designed for microscopy acquisition."""

from collections.abc import Sequence
from importlib.metadata import PackageNotFoundError, version

import numpy as np

from ome_writers.DimensionInfo import DimensionInfo
from ome_writers.OMEStream import OMEStream
from ome_writers.AquireZarrStream import AquireZarrStream
from ome_writers.TensorStoreZarrStream import TensorStoreZarrStream

try:
    __version__ = version("ome-writers")
except PackageNotFoundError:
    __version__ = "uninstalled"
__author__ = "Talley Lambert"
__email__ = "talley.lambert@gmail.com"


def create_stream(
    path: str, dtype: np.dtype, dimensions: Sequence[DimensionInfo]
) -> OMEStream:
    # stream = AquireZarrStream()
    stream = TensorStoreZarrStream()
    return stream.create(path, dtype, dimensions)
