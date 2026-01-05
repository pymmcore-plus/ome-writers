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


class YaozarrsStream(MultiPositionOMEStream):
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
                "`pip install yaozarrs[write-zarr]` or "
                "`pip install yaozarrs[write-tensorstore]`."
            )
            raise ImportError(msg) from e

        self._yaozarrs_prepare_image = prepare_image
        self._yaozarrs_Bf2RawBuilder = Bf2RawBuilder

        super().__init__()

        self._group_path: Path | None = None
        self._arrays: dict[str, Any] = {}
        self._futures: list[Any] = []
        self._builder: Bf2RawBuilder | None = None
        self._writer: Literal["auto", "tensorstore", "zarr"] = "tensorstore"
        # Mapping from acquisition order to storage order indices
        self._index_mapping: tuple[int, ...] | None = None

    def create(
        self,
        path: str,
        dtype: np.dtype,
        dimensions: Sequence[Dimension],
        *,
        overwrite: bool = False,
        writer: Literal["auto", "tensorstore", "zarr"] = "tensorstore",
    ) -> Self:
        """Create the OME-Zarr storage structure.

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
        writer : Literal["auto", "tensorstore", "zarr"], optional
            Backend to use for writing arrays. Options are:
            - "auto": Automatically select based on availability
            - "tensorstore": Use tensorstore backend (default)
            - "zarr": Use zarr-python backend

        Returns
        -------
        Self
            The configured stream instance.
        """
        self._writer = writer
        # Use MultiPositionOMEStream to handle position logic
        num_positions, non_position_dims = self._init_positions(dimensions)
        self._group_path = Path(self._normalize_path(path))

        # Reorder dimensions to NGFF order and get index mapping
        ngff_dims, self._index_mapping = self._reorder_to_ngff(non_position_dims)

        # Get shape from NGFF-ordered dimensions
        shape = tuple(d.size for d in ngff_dims)

        # Build the Image metadata template with NGFF-ordered dimensions
        image = build_yaozarrs_image_metadata_v05(ngff_dims)

        if num_positions == 1 and self._position_dim is None:
            # Single position: use prepare_image directly
            _, arrays = self._yaozarrs_prepare_image(
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
            self._builder = self._yaozarrs_Bf2RawBuilder(
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

    def _reorder_to_ngff(
        self, dims: Sequence[Dimension]
    ) -> tuple[list[Dimension], tuple[int, ...]]:
        """Reorder dimensions to NGFF v0.5 order: time, channel, space.

        NGFF v0.5 requires axes to be ordered as:
        1. time (t) - optional
        2. channel (c) or custom - optional
        3. space (z, y, x) - required 2-3 axes

        Parameters
        ----------
        dims : Sequence[Dimension]
            Dimensions in acquisition order (including spatial dims).

        Returns
        -------
        tuple[list[Dimension], tuple[int, ...]]
            - Reordered dimensions in NGFF order
            - Index mapping for non-spatial dimensions only (for use during append)
        """
        # Categorize dimensions
        time_dims = [d for d in dims if d.label == "t"]
        channel_dims = [d for d in dims if d.label == "c"]
        space_dims = [d for d in dims if d.label in ("z", "y", "x")]
        non_spatial_dims = [d for d in dims if d.label not in ("y", "x")]

        # Space dimensions must be in z, y, x order
        space_order = {"z": 0, "y": 1, "x": 2}
        space_dims.sort(key=lambda d: space_order.get(d.label, 3))

        # Build NGFF order: time, channel, space
        ngff_dims = time_dims + channel_dims + space_dims

        # Build NGFF order for non-spatial dims (for index mapping during append)
        ngff_non_spatial = [d for d in ngff_dims if d.label not in ("y", "x")]

        # Create mapping for non-spatial dimensions only
        # This maps from acquisition position to storage position for non-spatial dims
        acq_to_ngff = tuple(ngff_non_spatial.index(dim) for dim in non_spatial_dims)

        return ngff_dims, acq_to_ngff

    def _write_to_backend(
        self, array_key: str, index: tuple[int, ...], frame: np.ndarray
    ) -> None:
        """Write frame to the yaozarrs-created array.

        The index is in acquisition order, but we need to transpose it to
        storage (NGFF) order before writing.

        _index_mapping[i] tells us which acquisition dimension goes to storage position i.
        So storage_index[i] = acquisition_index[_index_mapping[i]].
        """
        array = self._arrays[array_key]

        # Transpose index from acquisition order to storage order
        if self._index_mapping is not None:
            storage_index = tuple(index[acq_pos] for acq_pos in self._index_mapping)
        else:
            storage_index = index

        # Check if this is a tensorstore array or zarr array
        if hasattr(array, "store"):
            # zarr.Array - direct indexing
            array[storage_index] = frame
        else:
            # tensorstore.TensorStore - use write() method
            future = array[storage_index].write(frame)
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
