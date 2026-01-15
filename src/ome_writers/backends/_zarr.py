"""Yaozarrs-based backends for OME-Zarr v0.5."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

from ome_writers._backend import ArrayBackend

if TYPE_CHECKING:
    import numpy as np

    from ome_writers.router import FrameRouter, PositionInfo
    from ome_writers.schema import AcquisitionSettings, Dimension, Plate, Position

try:
    from yaozarrs import DimSpec, v05
    from yaozarrs.write.v05 import Bf2RawBuilder, PlateBuilder, prepare_image
except ImportError as e:
    raise ImportError(
        f"{__name__} requires yaozarrs with write support: "
        "`pip install yaozarrs[write-zarr]`."
    ) from e


class YaozarrsBackend(ArrayBackend):
    """Base backend using yaozarrs for OME-Zarr v0.5 structure.

    Subclasses must define the `writer` class attribute to specify which
    yaozarrs writer to use (e.g., "zarr", "tensorstore").
    """

    writer: Literal["zarr", "tensorstore"]

    def __init__(self) -> None:
        self._arrays: list[Any] = []
        self._finalized = False

    def prepare(self, settings: AcquisitionSettings, router: FrameRouter) -> None:
        """Initialize OME-Zarr storage structure."""

        self._finalized = False
        array_settings = settings.array_settings
        root = Path(settings.root_path).expanduser().resolve()
        positions = router.positions

        # Build storage metadata
        storage_dims = router.storage_dimensions
        shape = tuple(d.count if d.count is not None else 1 for d in storage_dims)
        chunks = _get_chunks(storage_dims)
        image = _build_image_metadata(storage_dims)

        # Plate mode
        if settings.plate is not None:
            # Validate and group positions
            for i, pos in enumerate(positions):
                if pos.row is None or pos.column is None:
                    raise ValueError(
                        f"Position '{pos.name}' (index {i}) must have row and "
                        "column for plate mode."
                    )

            # mapping of {(row, column): [(position_index, Position), ...]}
            well_positions: dict[tuple[str, str], list[PositionInfo]] = {}
            for i, pos in enumerate(positions):
                key = (pos.row, pos.column)
                well_positions.setdefault(key, []).append((i, pos))

            # Build plate metadata and arrays
            builder = PlateBuilder(
                root,
                plate=_build_plate_metadata(settings.plate, well_positions),
                overwrite=settings.overwrite,
                chunks=chunks,
                writer=self.writer,
            )

            for (row, col), pos_list in well_positions.items():
                images_dict = {
                    pos.name: (image, [(shape, array_settings.dtype)])
                    for _idx, pos in pos_list
                }
                builder.add_well(row=row, col=col, images=images_dict)

            _, all_arrays = builder.prepare()
            # Map position index to array path: row/col/name/0
            self._arrays = [
                all_arrays[f"{pos.row}/{pos.column}/{pos.name}/0"] for pos in positions
            ]

        # Single position
        elif len(positions) == 1:
            _, all_arrays = prepare_image(
                root,
                image,
                datasets=[(shape, array_settings.dtype)],
                overwrite=settings.overwrite,
                chunks=chunks,
                writer=self.writer,
            )
            self._arrays = [all_arrays["0"]]

        # Multi-position (bf2raw layout)
        else:
            builder = Bf2RawBuilder(
                root,
                overwrite=settings.overwrite,
                chunks=chunks,
                writer=self.writer,
            )
            for pos in positions:
                builder.add_series(pos.name, image, [(shape, array_settings.dtype)])

            _, all_arrays = builder.prepare()
            self._arrays = [all_arrays[f"{pos.name}/0"] for pos in positions]

    def write(
        self,
        position_info: PositionInfo,
        index: tuple[int, ...],
        frame: np.ndarray,
    ) -> None:
        """Write frame to specified location with auto-resize for unlimited dims."""
        if self._finalized:
            raise RuntimeError("Cannot write after finalize().")
        if not self._arrays:
            raise RuntimeError("Backend not prepared. Call prepare() first.")

        array = self._arrays[position_info[0]]

        # Resize if needed (index may be shorter than shape due to spatial dims)
        new_shape = list(array.shape)
        for i, idx_val in enumerate(index):
            new_shape[i] = max(new_shape[i], idx_val + 1)

        if new_shape != list(array.shape):
            array.resize(new_shape)

        array[index] = frame

    def finalize(self) -> None:
        """Flush and release resources."""
        if not self._finalized:
            self._arrays.clear()
            self._finalized = True


class ZarrBackend(YaozarrsBackend):
    """OME-Zarr writer using zarr-python via yaozarrs."""

    writer = "zarr"

    def is_incompatible(self, settings: AcquisitionSettings) -> Literal[False] | str:
        if not settings.root_path.endswith(".zarr"):
            return "Root path must end with .zarr for ZarrBackend."
        return False


class TensorstoreBackend(YaozarrsBackend):
    """OME-Zarr writer using tensorstore via yaozarrs."""

    writer = "tensorstore"

    def is_incompatible(self, settings: AcquisitionSettings) -> Literal[False] | str:
        if not settings.root_path.endswith(".zarr"):
            return "Root path must end with .zarr for TensorstoreBackend."
        return False


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------


def _get_chunks(dims: list[Dimension]) -> tuple[int, ...]:
    """Compute chunk sizes from dimensions.

    Defaults: full size for Y/X (last 2 dims), 1 for others.
    """
    chunks = []
    n = len(dims)
    for i, dim in enumerate(dims):
        if dim.chunk_size is not None:
            chunks.append(dim.chunk_size)
        elif i >= n - 2:  # Last 2 dims  # FIXME:  MAGIC NUMBER
            chunks.append(dim.count or 1)
        else:
            chunks.append(1)
    return tuple(chunks)


def _build_image_metadata(dims: list[Dimension]) -> Any:
    """Build yaozarrs v05 Image metadata from Dimensions."""

    dim_specs = [
        DimSpec(
            name=dim.name,
            size=dim.count,
            type=dim.type,
            unit=dim.unit,
            scale=1.0 if dim.scale is None else dim.scale,
            translation=dim.translation,
        )
        for dim in dims
    ]
    return v05.Image(multiscales=[v05.Multiscale.from_dims(dim_specs)])


def _build_plate_metadata(
    plate: Plate, well_positions: dict[tuple[str, str], list[tuple[int, Position]]]
) -> Any:
    """Build yaozarrs v05 Plate metadata from ome-writers Plate schema."""
    return v05.Plate(
        plate=v05.PlateDef(
            name=plate.name,
            rows=[v05.Row(name=name) for name in plate.row_names],
            columns=[v05.Column(name=name) for name in plate.column_names],
            wells=[
                v05.PlateWell(
                    path=f"{row}/{col}",
                    rowIndex=plate.row_names.index(row),
                    columnIndex=plate.column_names.index(col),
                )
                for row, col in well_positions
            ],
        )
    )
