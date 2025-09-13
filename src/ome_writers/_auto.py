"""OME-TIFF and OME-ZARR writer APIs designed for microscopy acquisition."""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal, TypeAlias

import numpy as np

from .backends._acquire_zarr import AcquireZarrStream
from .backends._tensorstore import TensorStoreZarrStream
from .backends._tifffile import TifffileStream

if TYPE_CHECKING:
    from collections.abc import Sequence
    from pathlib import Path

    from numpy.typing import DTypeLike

    from ._dimensions import Dimension
    from ._stream_base import OMEStream

__all__ = ["create_stream", "init_stream"]

BackendName: TypeAlias = Literal["acquire-zarr", "tensorstore", "tiff"]
BACKENDS: dict[BackendName, type[OMEStream]] = {
    "acquire-zarr": AcquireZarrStream,
    "tensorstore": TensorStoreZarrStream,
    "tiff": TifffileStream,
}


def init_stream(
    path: str | Path,
    *,
    backend: Literal[BackendName, "auto"] = "auto",
) -> OMEStream:
    """Initialize a stream object for `path` using the specified backend.

    Parameters
    ----------
    path : str
        Path to the output file or directory.
    backend : Literal["acquire-zarr", "tensorstore", "tiff", "auto"], optional
        The backend to use for writing the data. Options are:

        - "acquire-zarr": Use acquire-zarr backend.
        - "tensorstore": Use tensorstore backend.
        - "tiff": Use tifffile backend.
        - "auto": Automatically determine the backend based on the file extension.

        Default is "auto".

    Returns
    -------
    OMEStream
        A stream object configured for the specified backend.
    """
    if backend == "auto":
        backend = _autobackend(path)
    elif backend not in {"acquire-zarr", "tensorstore", "tiff"}:
        raise ValueError(  # pragma: no cover
            f"Invalid backend '{backend}'. "
            "Choose from 'acquire-zarr', 'tensorstore', or 'tiff'."
        )

    return BACKENDS[backend]()


def create_stream(
    path: str | Path,
    dtype: DTypeLike,
    dimensions: Sequence[Dimension],
    *,
    backend: Literal[BackendName, "auto"] = "auto",
    overwrite: bool = False,
) -> OMEStream:
    """Create a stream for writing OME-TIFF or OME-ZARR data.

    Parameters
    ----------
    path : str
        Path to the output file or directory.
    dtype : np.dtype
        NumPy data type for the image data.
    dimensions : Sequence[DimensionInfo]
        Sequence of dimension information describing the data structure.

    backend : Literal["acquire-zarr", "tensorstore", "tiff", "auto"], optional
        The backend to use for writing the data. Options are:

        - "acquire-zarr": Use acquire-zarr backend.
        - "tensorstore": Use tensorstore backend.
        - "tiff": Use tifffile backend.
        - "auto": Automatically determine the backend based on the file extension.

        Default is "auto".
    overwrite : bool, optional
        Whether to overwrite existing files or directories. Default is False.

    Returns
    -------
    OMEStream
        A configured stream ready for writing frames.
    """
    stream = init_stream(path, backend=backend)
    return stream.create(str(path), np.dtype(dtype), dimensions, overwrite=overwrite)


def _autobackend(path: str | Path) -> Literal["acquire-zarr", "tensorstore", "tiff"]:
    path = str(path)
    if path.endswith(".zarr"):
        if AcquireZarrStream.is_available():
            return "acquire-zarr"
        elif TensorStoreZarrStream.is_available():  # pragma: no cover
            return "tensorstore"
        raise ValueError(  # pragma: no cover
            "Cannot determine backend automatically for .zarr file. "
            "Neither acquire-zarr nor tensorstore is available. "
            "Please install one of these packages."
        )
    elif path.endswith(".tiff") or path.endswith(".ome.tiff"):
        if TifffileStream.is_available():
            return "tiff"
        raise ValueError(  # pragma: no cover
            "Cannot determine backend automatically for .tiff file. "
            "Please install tifffile."
        )
    raise ValueError(  # pragma: no cover
        "Cannot determine backend automatically. "
        "Please specify 'acquire-zarr', 'tensorstore', or 'tiff'."
    )
