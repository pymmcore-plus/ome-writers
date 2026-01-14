"""Zarr-python backend using yaozarrs for OME-Zarr v0.5 structure."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

from ome_writers.backend import ArrayBackend
from ome_writers.schema import (
    ArraySettings,
    Dimension,
    Plate,
    Position,
    PositionDimension,
)

if TYPE_CHECKING:
    import numpy as np

    from ome_writers.router import FrameRouter, PositionInfo
    from ome_writers.schema import AcquisitionSettings, ArraySettings


class ZarrBackend(ArrayBackend):
    """OME-Zarr writer using zarr-python via yaozarrs.

    This backend creates OME-Zarr v0.5 compatible stores using zarr-python for
    array I/O. The yaozarrs library handles the OME-NGFF metadata structure.

    For multi-position data, uses the bioformats2raw layout where each position
    is a separate Image in the hierarchy.
    """

    def __init__(self) -> None:
        self._arrays: list[Any] = []  # Indexed by position index
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

        self._finalized = False
        array_settings = settings.array_settings
        group_path = Path(settings.root_path).expanduser().resolve()
        positions = router.positions

        # Build storage dimensions (excluding position, in storage order)
        storage_dims = _get_storage_dims(array_settings)
        # For unlimited dimensions (count=None), start with size 1
        shape = tuple(d.count if d.count is not None else 1 for d in storage_dims)
        chunks = _get_chunks(storage_dims)

        # Build yaozarrs Image metadata
        image = _build_image_metadata(storage_dims)

        # Plate mode: use PlateBuilder for well/field hierarchy
        if settings.plate is not None:
            self._prepare_plate(settings, group_path, image, shape, chunks, positions)
            return

        # Non-plate mode: use simple image or Bf2RawBuilder
        if len(positions) == 1:
            _, arrays = prepare_image(
                group_path,
                image,
                datasets=[(shape, array_settings.dtype)],
                overwrite=settings.overwrite,
                chunks=chunks,
                writer="zarr",
            )
            # Single position -> single array at index 0
            self._arrays = [arrays["0"]]
        else:
            builder = Bf2RawBuilder(
                group_path,
                overwrite=settings.overwrite,
                chunks=chunks,
                writer="zarr",
            )
            for pos in positions:
                builder.add_series(pos.name, image, [(shape, array_settings.dtype)])

            _, all_arrays = builder.prepare()

            # Build array list indexed by position order
            self._arrays = [all_arrays[f"{pos.name}/0"] for pos in positions]

    def _prepare_plate(
        self,
        settings: AcquisitionSettings,
        group_path: Path,
        image: Any,
        shape: tuple[int, ...],
        chunks: tuple[int, ...],
        positions: list[Position],
    ) -> None:
        """Initialize plate structure using PlateBuilder."""
        from yaozarrs.write.v05 import PlateBuilder

        array_settings = settings.array_settings

        # Validate all positions have row/column for plate mode
        for i, pos in enumerate(positions):
            if pos.row is None or pos.column is None:
                raise ValueError(
                    f"Position '{pos.name}' (index {i}) must have row and column "
                    "defined for plate mode."
                )

        # Group positions by (row, column) to identify wells
        # well_positions: {(row, col): [(pos_index, pos), ...]}
        well_positions: dict[tuple[str, str], list[tuple[int, Position]]] = {}
        for i, pos in enumerate(positions):
            key = (pos.row, pos.column)  # type: ignore[arg-type]
            well_positions.setdefault(key, []).append((i, pos))

        # Build mapping: position_index -> array path using Position.name as FOV path
        # e.g., Position(name="p0", row="A", column="1") -> "A/1/p0"
        pos_idx_to_path: dict[int, str] = {}
        for (row, col), pos_list in well_positions.items():
            for pos_idx, pos in pos_list:
                pos_idx_to_path[pos_idx] = f"{row}/{col}/{pos.name}"

        # Build yaozarrs Plate metadata from settings.plate
        plate_meta = _build_plate_metadata(settings.plate, well_positions)

        # Create PlateBuilder and add wells
        builder = PlateBuilder(
            group_path,
            plate=plate_meta,
            overwrite=settings.overwrite,
            chunks=chunks,
            writer="zarr",
        )

        for (row, col), pos_list in well_positions.items():
            # Each position in this well becomes a FOV, using Position.name as path
            images_dict = {
                pos.name: (image, [(shape, array_settings.dtype)])
                for _, pos in pos_list
            }
            builder.add_well(row=row, col=col, images=images_dict)

        _, all_arrays = builder.prepare()

        # Build array list indexed by position order
        # all_arrays has keys like "A/1/p0/0" (row/col/fov_name/dataset)
        self._arrays = [
            all_arrays[f"{pos_idx_to_path[i]}/0"] for i in range(len(positions))
        ]

    def write(
        self,
        position_info: PositionInfo,
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

        pos_idx, _pos = position_info
        array = self._arrays[pos_idx]
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

        self._arrays[pos_idx][index] = frame

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


def _build_plate_metadata(
    plate: Plate,
    well_positions: dict[tuple[str, str], list[tuple[int, Position]]],
) -> Any:
    """Build yaozarrs v05 Plate metadata from ome-writers Plate schema.

    Parameters
    ----------
    plate
        The ome-writers Plate schema with row_names, column_names, and name.
    well_positions
        Mapping of (row, col) to list of (position_index, Position) tuples.
    """
    from yaozarrs import v05

    # Build row and column lists from the plate schema
    rows = [v05.Row(name=name) for name in plate.row_names]
    columns = [v05.Column(name=name) for name in plate.column_names]

    # Build wells list from the positions that were added
    wells = []
    for row_name, col_name in well_positions:
        row_idx = plate.row_names.index(row_name)
        col_idx = plate.column_names.index(col_name)
        wells.append(
            v05.PlateWell(
                path=f"{row_name}/{col_name}",
                rowIndex=row_idx,
                columnIndex=col_idx,
            )
        )

    return v05.Plate(
        plate=v05.PlateDef(
            rows=rows,
            columns=columns,
            wells=wells,
            name=plate.name,
        )
    )
