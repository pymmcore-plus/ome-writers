"""Zarr-python backend using yaozarrs for OME-Zarr v0.5 structure."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

from ome_writers.backend import ArrayBackend
from ome_writers.schema_pydantic import Dimension, PositionDimension

if TYPE_CHECKING:
    import numpy as np

    from ome_writers.router import FrameRouter
    from ome_writers.schema_pydantic import AcquisitionSettings, ArraySettings


class ZarrBackend(ArrayBackend):
    """OME-Zarr writer using zarr-python via yaozarrs.

    This backend creates OME-Zarr v0.5 compatible stores using zarr-python for
    array I/O. The yaozarrs library handles the OME-NGFF metadata structure.

    For multi-position data, uses the bioformats2raw layout where each position
    is a separate Image in the hierarchy.
    """

    def __init__(self) -> None:
        self._arrays: dict[str, Any] = {}
        self._finalized = False

    # -------------------------------------------------------------------------
    # Compatibility
    # -------------------------------------------------------------------------

    def is_incompatible(self, settings: AcquisitionSettings) -> Literal[False] | str:
        """Check compatibility with settings.

        If incompatible, returns a string describing the issue.
        """
        if not settings.root_path.endswith(".zarr"):
            return "Root path must end with .zarr for ZarrBackend."
        return False

    # -------------------------------------------------------------------------
    # Lifecycle
    # -------------------------------------------------------------------------

    def prepare(
        self,
        settings: AcquisitionSettings,
        router: FrameRouter,
    ) -> None:
        """Initialize OME-Zarr storage structure."""
        try:
            from yaozarrs.write.v05 import Bf2RawBuilder, prepare_image
        except ImportError as e:
            raise ImportError(
                "ZarrBackend requires yaozarrs with write support: "
                "`pip install yaozarrs[write-zarr]`."
            ) from e

        if settings.array_settings is None:
            raise ValueError("ZarrBackend requires arrays to be set.")

        self._finalized = False
        array_settings = settings.array_settings
        group_path = Path(settings.root_path).expanduser().resolve()
        position_keys = router.position_keys

        # Build storage dimensions (excluding position, in storage order)
        storage_dims = _get_storage_dims(array_settings)
        # For unlimited dimensions (count=None), start with size 1
        shape = tuple(d.count if d.count is not None else 1 for d in storage_dims)
        chunks = _get_chunks(storage_dims)

        # Build yaozarrs Image metadata
        image = _build_image_metadata(storage_dims)

        # Create arrays for each position
        if len(position_keys) == 1:
            _, arrays = prepare_image(
                group_path,
                image,
                datasets=[(shape, array_settings.dtype)],
                overwrite=settings.overwrite,
                chunks=chunks,
                writer="zarr",
            )
            # Map position key to the array (dataset path is "0")
            self._arrays = {position_keys[0]: arrays["0"]}
        else:
            builder = Bf2RawBuilder(
                group_path,
                overwrite=settings.overwrite,
                chunks=chunks,
                writer="zarr",
            )
            for pos_key in position_keys:
                builder.add_series(pos_key, image, [(shape, array_settings.dtype)])

            _, all_arrays = builder.prepare()

            # Remap keys: "pos_key/0" -> "pos_key"
            for pos_key in position_keys:
                self._arrays[pos_key] = all_arrays[f"{pos_key}/0"]

    def write(
        self,
        position_key: str,
        index: tuple[int, ...],
        frame: np.ndarray,
    ) -> None:
        """Write frame to the specified location.

        For unlimited dimensions, automatically resizes the array as needed.
        """
        if self._finalized:
            raise RuntimeError("Cannot write after finalize().")
        if not self._arrays:
            raise RuntimeError("Backend not prepared. Call prepare() first.")

        array = self._arrays[position_key]
        current_shape = array.shape

        # Check if we need to resize for any dimension (excluding Y, X)
        new_shape = list(current_shape)
        needs_resize = False

        for i, idx_val in enumerate(index):
            if idx_val >= current_shape[i]:
                # Grow to accommodate this index
                new_shape[i] = idx_val + 1
                needs_resize = True

        if needs_resize:
            array.resize(new_shape)

        self._arrays[position_key][index] = frame

    def finalize(self) -> None:
        """Flush and release resources."""
        if self._finalized:
            return
        self._arrays.clear()
        self._finalized = True


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------


def _get_storage_dims(settings: ArraySettings) -> list[Dimension]:
    """Extract storage dimensions from settings (excludes PositionDimension)."""
    dims = []
    for dim in settings.dimensions:
        if isinstance(dim, PositionDimension):
            continue
        dims.append(dim)
    return dims


def _get_chunks(dims: list[Dimension]) -> tuple[int, ...]:
    """Compute chunk sizes from dimensions.

    Defaults: full size for Y/X (last 2 dims), 1 for others.
    """
    chunks = []
    n = len(dims)
    for i, dim in enumerate(dims):
        if dim.chunk_size is not None:
            chunks.append(dim.chunk_size)
        elif i >= n - 2:  # Last 2 dims (spatial)
            chunks.append(dim.count or 1)
        else:
            chunks.append(1)
    return tuple(chunks)


def _build_image_metadata(dims: list[Dimension]) -> Any:
    """Build yaozarrs v05 Image metadata from Dimension objects."""
    from yaozarrs import v05

    axes = []
    scales = []

    for dim in dims:
        axes.append(_dim_to_axis(dim))
        scales.append(dim.scale if dim.scale is not None else 1.0)

    return v05.Image(
        multiscales=[
            v05.Multiscale(
                axes=axes,
                datasets=[
                    v05.Dataset(
                        path="0",
                        coordinateTransformations=[
                            v05.ScaleTransformation(scale=scales)
                        ],
                    )
                ],
            )
        ],
    )


def _dim_to_axis(dim: Dimension) -> Any:
    """Convert a schema Dimension to a yaozarrs v05 Axis."""
    from yaozarrs import v05

    name = dim.name
    unit = dim.unit

    if dim.type == "time" or name == "t":
        return v05.TimeAxis(name=name, unit=unit)
    elif dim.type == "channel" or name == "c":
        return v05.ChannelAxis(name=name)
    elif dim.type == "space" or name in ("x", "y", "z"):
        return v05.SpaceAxis(name=name, unit=unit)
    else:
        return v05.CustomAxis(name=name)
