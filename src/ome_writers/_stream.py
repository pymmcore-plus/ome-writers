"""OME-TIFF and OME-ZARR writer APIs designed for microscopy acquisition."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal, TypeAlias, get_args

if TYPE_CHECKING:
    import numpy as np

    from ._backend import ArrayBackend
    from ._router import FrameRouter
    from .schema import AcquisitionSettings

__all__ = ["OMEStream", "create_stream"]

BackendName: TypeAlias = Literal["acquire-zarr", "tensorstore", "zarr", "tiff"]
VALID_BACKEND_NAMES: set[str] = set(get_args(BackendName)) | {"auto"}


class OMEStream:
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
        pos_info, idx = next(self._iterator)
        self._backend.write(pos_info, idx, frame)

    def update_metadata(self, metadata: Any) -> None:
        """Update metadata in the backend.  Meaning is format-dependent."""
        self._backend.update_metadata(metadata)

    def __enter__(self) -> OMEStream:
        """Enter context manager."""
        return self

    def __exit__(self, exc_type: object, exc_val: object, exc_tb: object) -> None:
        """Exit context manager, finalizing the backend."""
        self._backend.finalize()


def create_stream(settings: AcquisitionSettings) -> OMEStream:
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
    from ._router import FrameRouter

    # TODO: Support other backends in new protocol
    backend: ArrayBackend = _create_backend(settings)
    router = FrameRouter(settings.array_settings)
    backend.prepare(settings, router)
    return OMEStream(backend, router)


def _create_backend(settings: AcquisitionSettings) -> ArrayBackend:
    """Create and prepare the appropriate backend based on settings.

    Parameters
    ----------
    settings : AcquisitionSettings
        The acquisition settings specifying the desired backend.

    Returns
    -------
    ArrayBackend
        An initialized backend ready for writing.

    Raises
    ------
    ValueError
        If the specified backend is unknown or incompatible.
    """

    requested_backend = settings.backend.lower()
    if requested_backend not in VALID_BACKEND_NAMES:
        raise ValueError(
            f"Unknown backend requested: '{requested_backend!r}'.  "
            f"Must be one of {VALID_BACKEND_NAMES}."
        )

    # TODO:
    # this needs work...
    # we need better error handling for incompatibilities, etc...

    backend: ArrayBackend | None = None
    if requested_backend in ("auto", "zarr"):
        try:
            from .backends._zarr import ZarrBackend
        except ImportError as e:
            if requested_backend == "zarr":
                raise ValueError(
                    "Zarr backend requested but 'zarr' package is not installed."
                ) from e
        else:
            backend = ZarrBackend()
    elif requested_backend == "tensorstore":
        try:
            from .backends._zarr import TensorstoreBackend
        except ImportError as e:
            raise ValueError(
                "Tensorstore backend requested but required packages are not installed."
            ) from e
        else:
            backend = TensorstoreBackend()
    elif requested_backend == "tiff":
        try:
            from .backends._tifffile import TiffBackend
        except ImportError as e:
            raise ValueError(
                "TIFF backend requested but required packages are not installed. "
                "Install with: pip install ome-writers[tifffile]"
            ) from e
        else:
            backend = TiffBackend()
    elif requested_backend == "acquire-zarr":
        try:
            from .backends._acquire_zarr import AcquireZarrBackend
        except ImportError as e:
            raise ValueError(
                "AcquireZarr backend requested but 'acquire-zarr' package is "
                "not installed. "
                "Install with: pip install acquire-zarr"
            ) from e
        else:
            backend = AcquireZarrBackend()

    if backend is None:
        raise ValueError(
            f"Could not create backend for requested name: '{requested_backend}'."
        )

    if reason := backend.is_incompatible(settings):
        raise ValueError(
            f"Backend '{type(backend).__name__}' is incompatible: {reason}"
        )

    return backend
