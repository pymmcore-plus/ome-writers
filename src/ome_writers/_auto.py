"""OME-TIFF and OME-ZARR writer APIs designed for microscopy acquisition."""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal, TypeAlias

from .backends._acquire_zarr import AcquireZarrStream
from .backends._tifffile import TifffileStream
from .backends._yaozarrs import TensorStoreZarrStream, ZarrPythonStream

if TYPE_CHECKING:
    from pathlib import Path

    import numpy as np

    from ome_writers._dimensions import Dimension

    from ._stream_base import OMEStream
    from .backend import ArrayBackend
    from .router import FrameRouter
    from .schema_pydantic import AcquisitionSettings, ArraySettings

__all__ = ["Stream", "create_stream", "init_stream"]

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
    backend: BackendName | Literal["auto"] = "auto",
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


# FIXME - DEPRECATED: This function is for legacy backends only
def convert_array_settings_to_dimensions(
    array_settings: ArraySettings,
) -> tuple[list[Dimension], str]:
    """Convert ArraySettings to old-style dimension list and dtype.

    DEPRECATED: This function is only for legacy backends that don't support
    the new ArrayBackend protocol. New code should use the new protocol directly.

    Parameters
    ----------
    array_settings : ArraySettings
        New schema array settings to convert.

    Returns
    -------
    tuple[list[Dimension], str]
        Tuple of (dimensions list, dtype) for passing to backend.
    """
    from typing import cast

    from ._dimensions import Dimension
    from .schema_pydantic import Dimension as NewDimension
    from .schema_pydantic import PositionDimension as NewPositionDimension

    dimensions: list[Dimension] = []
    for dim in array_settings.dimensions:
        # Skip PositionDimension - old backends handle positions differently
        if isinstance(dim, NewPositionDimension):
            continue

        # After isinstance check, dim is guaranteed to be NewDimension
        dim = cast("NewDimension", dim)

        # Convert name to label
        label = dim.name
        if label not in ("x", "y", "z", "t", "c", "p"):
            raise ValueError(
                f"Dimension name '{label}' is not a standard axis. "
                "Must be one of: x, y, z, t, c, p"
            )

        # Convert size_px to size
        size = dim.count
        if size is None:
            # Only first dimension can be unlimited
            if dimensions:  # not the first dimension
                raise ValueError(
                    f"Only the first dimension may have size_px=None, "
                    f"but dimension '{label}' (position {len(dimensions)}) has None."
                )
            size = 0  # Backend will handle unlimited dimension

        # Convert chunk_size_px to chunk_size
        chunk_size = dim.chunk_size

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


class Stream:
    """A stream wrapper for writing frames using the router + backend pattern.

    This class manages the iteration through frames in acquisition order and
    delegates writing to the backend in storage order.

    Usage
    -----
    >>> with create_stream(settings) as stream:
    ...     for frame in acquisition:
    ...         stream.append(frame)
    """

    def __init__(
        self,
        backend: ArrayBackend,
        router: FrameRouter,
    ) -> None:
        self._backend = backend
        self._router = router
        self._iterator = iter(router)

    def append(self, frame: np.ndarray) -> None:
        """Write the next frame in acquisition order.

        Parameters
        ----------
        frame : np.ndarray
            2D array containing the frame data (Y, X).

        Raises
        ------
        StopIteration
            If all frames have been written (for finite dimensions only).
            For unlimited dimensions, never raises StopIteration.
        """
        pos_key, idx = next(self._iterator)
        self._backend.write(pos_key, idx, frame)

    def __enter__(self) -> Stream:
        """Enter context manager."""
        return self

    def __exit__(self, exc_type: object, exc_val: object, exc_tb: object) -> None:
        """Exit context manager, finalizing the backend."""
        self._backend.finalize()


def create_stream(
    settings: AcquisitionSettings,
) -> Stream:
    """Create a stream for writing OME-TIFF or OME-ZARR data.

    Parameters
    ----------
    settings : AcquisitionSettings
        Acquisition settings containing array configuration, path, backend, etc.

    Returns
    -------
    Stream
        A configured stream ready for writing frames via `append()`.

    Raises
    ------
    ValueError
        If settings are invalid or backend is incompatible.
    NotImplementedError
        If requesting unsupported features (e.g., plate mode).

    Examples
    --------
    >>> settings = AcquisitionSettings(
    ...     root_path="output.zarr",
    ...     array_settings=ArraySettings(
    ...         dimensions=dims_from_standard_axes(
    ...             {"t": 10, "c": 2, "y": 512, "x": 512}
    ...         ),
    ...         dtype="uint16",
    ...     ),
    ...     overwrite=True,
    ... )
    >>> with create_stream(settings) as stream:
    ...     for i in range(20):  # 10 timepoints x 2 channels
    ...         stream.append(np.zeros((512, 512), dtype=np.uint16))
    """
    from .backends._zarr import ZarrBackend
    from .router import FrameRouter

    # Validate settings
    if settings.plate is not None:
        raise NotImplementedError(
            "Plate support not yet implemented. Use array_settings instead."
        )

    # TODO: Support other backends in new protocol
    backend: ArrayBackend = ZarrBackend()
    if reason := backend.is_incompatible(settings):
        raise ValueError(
            f"Backend '{type(backend).__name__}' is incompatible: {reason}"
        )

    router = FrameRouter(settings.array_settings)
    backend.prepare(settings, router)
    return Stream(backend, router)


def _autobackend(
    path: str | Path,
) -> Literal["acquire-zarr", "tensorstore", "zarr", "tiff"]:
    """Determine backend from file extension.

    For the new ArrayBackend protocol (used by create_stream), only 'zarr'
    is currently supported. This returns 'zarr' for .zarr files if zarr-python
    is available.
    """
    path = str(path)
    if path.endswith(".zarr"):
        # For now, prefer zarr-python backend for new protocol
        try:
            import zarr  # noqa: F401

            return "zarr"
        except ImportError:
            pass

        # Fallback to other zarr implementations (for legacy init_stream)
        if AcquireZarrStream.is_available():
            return "acquire-zarr"
        elif TensorStoreZarrStream.is_available():  # pragma: no cover
            return "tensorstore"

        raise ValueError(  # pragma: no cover
            "Cannot determine backend automatically for .zarr file. "
            "zarr-python is not installed. Please install: pip install zarr"
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
