"""OME-TIFF and OME-ZARR writer APIs designed for microscopy acquisition."""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal, TypeAlias

if TYPE_CHECKING:
    import numpy as np

    from .backend import ArrayBackend
    from .router import FrameRouter
    from .schema_pydantic import AcquisitionSettings

__all__ = ["Stream", "create_stream"]

BackendName: TypeAlias = Literal["acquire-zarr", "tensorstore", "zarr", "tiff"]


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
