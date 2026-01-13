"""ArrayBackend protocol for ome-writers.

ArrayBackend defines the interface that format-specific writers must implement.
Backends are responsible for:

1. **Compatibility validation** - Determine if settings can be handled by this backend
2. **Preparation** - Initialize storage structure (buffers, directories, files)
3. **Writing** - Accept frames at specified locations
4. **Finalization** - Flush, close, and clean up resources

Design Notes
------------
Backends are intentionally simple adapters. They:

- Receive storage-order indices and don't need to know about acquisition order
- Don't own the FrameRouter - that's the orchestrator's responsibility
- May ignore the index parameter for sequential-only backends (TIFF, acquire-zarr)

The compatibility check (`is_compatible`) allows backend selection to fail fast
when settings don't match capabilities. For example:
- acquire-zarr cannot handle `storage_order != "acquisition"` (sequential writes)
- tifffile cannot handle `count=None` on first dimension (no resizing)

Example Usage
-------------
>>> from ome_writers.backend import ArrayBackend
>>> from ome_writers.schema_pydantic import ArraySettings, dims_from_standard_axes
>>>
>>> settings = ArraySettings(
...     dimensions=dims_from_standard_axes({"t": 10, "c": 2, "y": 512, "x": 512}),
...     dtype="uint16",
... )
>>>
>>> backend = SomeConcreteBackend()
>>> if not backend.is_compatible(settings):
...     raise ValueError(backend.compatibility_error(settings))
>>>
>>> backend.prepare(settings, "/data/output.zarr", router, overwrite=True)
>>> for pos_key, idx in router:
...     frame = get_next_frame()
...     backend.write(pos_key, idx, frame)
>>> backend.finalize()
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from typing import Any

    import numpy as np

    from ome_writers.router import FrameRouter
    from ome_writers.schema_pydantic import ArraySettings


class ArrayBackend(ABC):
    """Abstract base class for array storage backends.

    Backends handle format-specific I/O (Zarr, TIFF, etc.) and receive frames
    with storage-order indices from the orchestration layer.

    Subclasses must implement all abstract methods. The typical lifecycle is:

    1. Check `is_compatible(settings)` to verify settings work with this backend
    2. Call `prepare(settings, path, ...)` to initialize storage
    3. Call `write(pos_key, idx, frame)` for each frame
    4. Call `finalize()` to flush and close
    """

    # -------------------------------------------------------------------------
    # Compatibility checking
    # -------------------------------------------------------------------------

    @abstractmethod
    def is_compatible(self, settings: ArraySettings) -> bool:
        """Check if this backend can handle the given settings.

        This method validates that the backend supports the requested
        configuration. Common incompatibilities include:

        - Sequential backends (acquire-zarr, tifffile) cannot reorder storage
        - TIFF cannot handle unlimited first dimension
        - Some backends may not support certain compression codecs

        Parameters
        ----------
        settings
            The array settings to validate.

        Returns
        -------
        bool
            True if compatible, False otherwise.

        See Also
        --------
        compatibility_error : Get a human-readable explanation of incompatibility.
        """

    def compatibility_error(self, settings: ArraySettings) -> str | None:
        """Return a human-readable error if settings are incompatible.

        Parameters
        ----------
        settings
            The array settings to check.

        Returns
        -------
        str | None
            Error message if incompatible, None if compatible.

        Examples
        --------
        >>> error = backend.compatibility_error(settings)
        >>> if error:
        ...     raise ValueError(f"Backend incompatible: {error}")
        """
        if self.is_compatible(settings):
            return None
        return "Settings are incompatible with this backend."

    # -------------------------------------------------------------------------
    # Lifecycle methods
    # -------------------------------------------------------------------------

    @abstractmethod
    def prepare(
        self,
        settings: ArraySettings,
        path: str,
        router: FrameRouter,
        *,
        overwrite: bool = False,
    ) -> None:
        """Initialize storage structure for the given settings.

        This method creates arrays, files, or directories as needed. After
        calling `prepare()`, the backend is ready to receive `write()` calls.

        Parameters
        ----------
        settings
            Array settings describing dimensions, dtype, chunking, etc.
        path
            Output path (file or directory depending on format).
        router
            The FrameRouter that will be used for iteration. Backends can use
            `router.position_keys` to get the list of positions to create.
        overwrite
            If True, remove existing data at path. If False and path exists,
            raise FileExistsError.

        Raises
        ------
        FileExistsError
            If path exists and overwrite is False.
        ValueError
            If settings are incompatible with this backend.
        """

    @abstractmethod
    def write(
        self,
        position_key: str,
        index: tuple[int, ...],
        frame: np.ndarray,
    ) -> None:
        """Write a frame to the specified location.

        Parameters
        ----------
        position_key
            Identifier for the position/array to write to. For single-position
            data, this is typically "0".
        index
            N-dimensional index in storage order (excludes Y/X spatial dims).
            Sequential backends may ignore this parameter.
        frame
            2D array (Y, X) containing the frame data.

        Raises
        ------
        RuntimeError
            If `prepare()` has not been called or `finalize()` was already called.
        """

    @abstractmethod
    def finalize(self) -> None:
        """Flush pending writes and release resources.

        After calling `finalize()`, the backend cannot accept more writes.
        This method should:
        - Flush any buffered data to disk
        - Close file handles
        - Write any deferred metadata
        - Release memory

        This method is idempotent - calling it multiple times is safe.
        """

    # -------------------------------------------------------------------------
    # Optional hooks
    # -------------------------------------------------------------------------

    def update_metadata(self, metadata: Any) -> None:  # noqa: B027
        """Update metadata after writing is complete.

        This optional hook allows updating file metadata after `finalize()`.
        The default implementation is a no-op. Backends that support post-hoc
        metadata updates (e.g., tifffile's `tiffcomment`) should override.

        Parameters
        ----------
        metadata
            Format-specific metadata object. The exact type depends on the
            backend (e.g., `ome_types.OME` for TIFF).

        Notes
        -----
        This method is typically called after `finalize()` when complete
        acquisition metadata is available (e.g., actual timestamps, stage
        positions recorded during acquisition).
        """

    # -------------------------------------------------------------------------
    # Context manager support
    # -------------------------------------------------------------------------

    def __enter__(self) -> ArrayBackend:
        """Enter context manager."""
        return self

    def __exit__(self, exc_type: object, exc_val: object, exc_tb: object) -> None:
        """Exit context manager, calling finalize()."""
        self.finalize()
