"""OME-TIFF and OME-ZARR writer APIs designed for microscopy acquisition."""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal, TypeAlias

import numpy as np

from ome_writers import schema

from .backends._acquire_zarr import AcquireZarrStream
from .backends._tifffile import TifffileStream
from .backends._yaozarrs import TensorStoreZarrStream, ZarrPythonStream

if TYPE_CHECKING:
    from pathlib import Path

    from ome_writers._dimensions import Dimension

    from . import schema
    from ._stream_base import OMEStream

__all__ = ["create_stream", "init_stream"]

BackendName: TypeAlias = Literal["acquire-zarr", "tensorstore", "zarr", "tiff"]
BACKENDS: dict[BackendName, type[OMEStream]] = {
    "acquire-zarr": AcquireZarrStream,
    "tensorstore": TensorStoreZarrStream,
    "zarr": ZarrPythonStream,
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
    backend : Literal["acquire-zarr", "tensorstore", "zarr", "tiff", "auto"]
        The backend to use for writing the data. Options are:

        - "acquire-zarr": Use acquire-zarr backend.
        - "tensorstore": Use tensorstore backend (yaozarrs with tensorstore).
        - "zarr": Use zarr-python backend (yaozarrs with zarr-python).
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
    elif backend not in {"acquire-zarr", "tensorstore", "zarr", "tiff"}:
        raise ValueError(  # pragma: no cover
            f"Invalid backend '{backend}'. "
            "Choose from 'acquire-zarr', 'tensorstore', 'zarr', or 'tiff'."
        )

    return BACKENDS[backend]()


# FIXME
def convert_array_settings_to_dimensions(
    array_settings: schema.ArraySettings,
) -> tuple[list[Dimension], np.dtype]:
    """Convert ArraySettings to old-style dimension list and dtype.

    Parameters
    ----------
    array_settings : schema.ArraySettings
        New schema array settings to convert.

    Returns
    -------
    tuple[list[Dimension], np.dtype]
        Tuple of (dimensions list, dtype) for passing to backend.
    """
    from ._dimensions import Dimension

    dimensions = []
    for dim in array_settings.dimensions:
        # Convert name to label
        label = dim.name
        if label not in ("x", "y", "z", "t", "c", "p"):
            raise ValueError(
                f"Dimension name '{label}' is not a standard axis. "
                "Must be one of: x, y, z, t, c, p"
            )

        # Convert size_px to size
        size = dim.size_px
        if size is None:
            # Only first dimension can be unlimited
            if dimensions:  # not the first dimension
                raise ValueError(
                    f"Only the first dimension may have size_px=None, "
                    f"but dimension '{label}' (position {len(dimensions)}) has None."
                )
            size = 0  # Backend will handle unlimited dimension

        # Convert chunk_size_px to chunk_size
        chunk_size = dim.chunk_size_px

        # Convert scale + unit to unit tuple, with defaults for standard axes
        unit = None
        if dim.scale is not None and dim.unit is not None:
            unit = (dim.scale, dim.unit)
        elif label in ("x", "y", "z"):
            # Default spatial units to micrometers
            unit = (1.0, "um")
        elif label == "t":
            # Default time units to seconds
            unit = (1.0, "s")

        dimensions.append(
            Dimension(
                label=label,  # type: ignore
                size=size,
                unit=unit,
                chunk_size=chunk_size,
            )
        )

    return dimensions, array_settings.dtype


def create_stream(
    settings: schema.AcquisitionSettings,
) -> OMEStream:
    """Create a stream for writing OME-TIFF or OME-ZARR data.

    Parameters
    ----------
    settings : schema.AcquisitionSettings
        Acquisition settings containing array configuration, path, backend, etc.

    Returns
    -------
    OMEStream
        A configured stream ready for writing frames.
    """
    # Validate settings
    if settings.plate is not None:
        raise NotImplementedError(
            "Plate support not yet implemented. Use settings.arrays instead."
        )
    if settings.arrays is None or len(settings.arrays) == 0:
        raise ValueError("settings.arrays must contain at least one ArraySettings.")
    if len(settings.arrays) > 1:
        raise NotImplementedError(
            "Multi-array support not yet implemented. "
            "Use a single ArraySettings in settings.arrays."
        )

    # Convert new schema to old schema
    array_settings = settings.arrays[0]
    dimensions, dtype = convert_array_settings_to_dimensions(array_settings)

    # Initialize and create stream with converted parameters
    backend = settings.backend
    if backend not in {"acquire-zarr", "tensorstore", "zarr", "tiff", "auto"}:
        raise ValueError(
            f"Invalid backend '{backend}'. "
            "Choose from 'acquire-zarr', 'tensorstore', 'zarr', 'tiff', or 'auto'."
        )
    stream = init_stream(settings.root_path, backend=backend)  # type: ignore
    return stream.create(
        str(settings.root_path),
        np.dtype(dtype),
        dimensions,
        overwrite=settings.overwrite,
    )


def _autobackend(
    path: str | Path,
) -> Literal["acquire-zarr", "tensorstore", "zarr", "tiff"]:
    path = str(path)
    if path.endswith(".zarr"):
        if AcquireZarrStream.is_available():
            return "acquire-zarr"
        elif TensorStoreZarrStream.is_available():  # pragma: no cover
            return "tensorstore"
        elif ZarrPythonStream.is_available():  # pragma: no cover
            return "zarr"
        raise ValueError(  # pragma: no cover
            "Cannot determine backend automatically for .zarr file. "
            "Neither acquire-zarr, tensorstore, nor zarr-python is installed. "
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
        "Please specify 'acquire-zarr', 'tensorstore', 'zarr', or 'tiff'."
    )
