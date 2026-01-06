from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

from typing_extensions import Self

from ome_writers._dimensions import build_yaozarrs_image_metadata_v05
from ome_writers._stream_base import MultiPositionOMEStream

if TYPE_CHECKING:
    from collections.abc import Sequence

    import numpy as np

    from ome_writers._dimensions import Dimension


class _YaozarrsStreamBase(MultiPositionOMEStream):
    """OME-Zarr writer using the yaozarrs library.

    This stream uses yaozarrs to create OME-Zarr v0.5 compatible stores with
    proper metadata. It supports both zarr-python and tensorstore backends
    through yaozarrs' unified API.

    For multi-position data, this uses the bioformats2raw layout pattern where
    each position is a separate Image in the hierarchy.
    """

    @classmethod
    def is_available(cls) -> bool:  # pragma: no cover
        """Check if the yaozarrs package with write support is available."""
        if importlib.util.find_spec("yaozarrs") is None:
            return False
        # Also check that the write module is available
        try:
            from yaozarrs.write.v05 import Bf2RawBuilder  # noqa: F401

            return True
        except ImportError:
            return False

    def __init__(self) -> None:
        try:
            from yaozarrs.write.v05 import Bf2RawBuilder, prepare_image
        except ImportError as e:
            msg = (
                "YaozarrsStream requires yaozarrs with write support: "
                "`pip install yaozarrs[write-tensorstore]` or"
                "`pip install yaozarrs[write-zarr]`."
            )
            raise ImportError(msg) from e

        self._prepare_image = prepare_image
        self._Bf2RawBuilder = Bf2RawBuilder

        super().__init__()

        self._group_path: Path | None = None
        self._arrays: dict[str, Any] = {}
        self._futures: list[Any] = []
        self._builder: Bf2RawBuilder | None = None
        self._writer: Literal["tensorstore", "zarr"] = "tensorstore"

    def _create(
        self,
        path: str,
        dtype: np.dtype,
        dimensions: Sequence[Dimension],
        *,
        overwrite: bool = False,
        writer: Literal["tensorstore", "zarr"],
    ) -> Self:
        """Internal method to create the OME-Zarr storage structure.

        Parameters
        ----------
        path : str
            Path to the output zarr store.
        dtype : np.dtype
            Data type for the arrays.
        dimensions : Sequence[Dimension]
            Sequence of dimensions describing the data structure.
        overwrite : bool, optional
            Whether to overwrite existing stores. Default is False.
        writer : Literal["tensorstore", "zarr"]
            Backend to use for writing arrays.

        Returns
        -------
        Self
            The configured stream instance.
        """
        self._writer = writer
        # Use MultiPositionOMEStream with NGFF ordering
        self._init_dimensions(dimensions, ngff_order=True)
        num_positions = self._num_positions
        storage_dims = self._storage_order_dims
        self._group_path = Path(self._normalize_path(path))

        # Get shape from NGFF-ordered dimensions
        shape = tuple(d.size for d in storage_dims)

        # Build the Image metadata with NGFF-ordered dimensions
        image = build_yaozarrs_image_metadata_v05(storage_dims)

        if num_positions == 1:
            # Single position: use prepare_image directly
            _, arrays = self._prepare_image(
                self._group_path,
                image,
                datasets=[(shape, dtype)],
                overwrite=overwrite,
                writer=self._writer,
            )
            # Store array handle for the single position
            self._arrays["0"] = arrays["0"]
        else:
            # Multi-position: use Bf2RawBuilder for bioformats2raw layout
            self._builder = self._Bf2RawBuilder(
                self._group_path,
                overwrite=overwrite,
                writer=self._writer,
            )

            # Add each position as a separate series
            for pos_idx in range(num_positions):
                array_key = str(pos_idx)
                self._builder.add_series(array_key, image, [(shape, dtype)])

            # Prepare all arrays
            _, all_arrays = self._builder.prepare()

            # Store array handles for each position
            # Bf2RawBuilder returns arrays with keys like "0/0" (series/dataset)
            for pos_idx in range(num_positions):
                array_key = str(pos_idx)
                # The dataset path within each image is "0" (first/only resolution)
                self._arrays[array_key] = all_arrays[f"{array_key}/0"]

        return self

    def _write_to_backend(
        self, position_key: str, index: tuple[int, ...], frame: np.ndarray
    ) -> None:
        """Write frame to the yaozarrs-created array.

        The index is already in storage (NGFF) order thanks to base class.
        """
        array = self._arrays[position_key]

        # Check if this is a tensorstore array or zarr array
        if hasattr(array, "store"):
            # zarr.Array - direct indexing
            array[index] = frame
        else:
            # tensorstore.TensorStore - use write() method
            future = array[index].write(frame)
            self._futures.append(future)

    def flush(self) -> None:
        """Flush pending writes and close the stream."""
        # Wait for all tensorstore writes to finish
        for future in self._futures:
            future.result()
        self._futures.clear()
        self._arrays.clear()
        self._builder = None

    def is_active(self) -> bool:
        """Return True if the stream has active arrays."""
        return bool(self._arrays)


class TensorStoreZarrStream(_YaozarrsStreamBase):
    """OME-Zarr writer using yaozarrs with tensorstore backend.

    This stream creates OME-Zarr v0.5 compatible stores using tensorstore for
    efficient array I/O. Data is always stored in NGFF canonical order (tczyx).
    """

    @classmethod
    def is_available(cls) -> bool:
        """Check if yaozarrs and tensorstore are available."""
        if not super().is_available():
            return False
        return importlib.util.find_spec("tensorstore") is not None

    def create(
        self,
        path: str,
        dtype: np.dtype,
        dimensions: Sequence[Dimension],
        *,
        overwrite: bool = False,
    ) -> Self:
        """Create the OME-Zarr storage structure using tensorstore.

        Parameters
        ----------
        path : str
            Path to the output zarr store.
        dtype : np.dtype
            Data type for the arrays.
        dimensions : Sequence[Dimension]
            Sequence of dimensions describing the data structure.
        overwrite : bool, optional
            Whether to overwrite existing stores. Default is False.
        **kwargs
            Additional keyword arguments (unused, for compatibility).

        Returns
        -------
        Self
            The configured stream instance.
        """
        return self._create(
            path, dtype, dimensions, overwrite=overwrite, writer="tensorstore"
        )


class ZarrPythonStream(_YaozarrsStreamBase):
    """OME-Zarr writer using yaozarrs with zarr-python backend.

    This stream creates OME-Zarr v0.5 compatible stores using zarr-python for
    array I/O. Data is always stored in NGFF canonical order (tczyx).
    """

    @classmethod
    def is_available(cls) -> bool:
        """Check if yaozarrs and zarr-python are available."""
        if not super().is_available():
            return False
        return importlib.util.find_spec("zarr") is not None

    def create(
        self,
        path: str,
        dtype: np.dtype,
        dimensions: Sequence[Dimension],
        *,
        overwrite: bool = False,
    ) -> Self:
        """Create the OME-Zarr storage structure using zarr-python.

        Parameters
        ----------
        path : str
            Path to the output zarr store.
        dtype : np.dtype
            Data type for the arrays.
        dimensions : Sequence[Dimension]
            Sequence of dimensions describing the data structure.
        overwrite : bool, optional
            Whether to overwrite existing stores. Default is False.
        **kwargs
            Additional keyword arguments (unused, for compatibility).

        Returns
        -------
        Self
            The configured stream instance.
        """
        return self._create(path, dtype, dimensions, overwrite=overwrite, writer="zarr")
