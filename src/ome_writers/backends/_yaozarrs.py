"""Yaozarrs-based backends for OME-Zarr v0.5."""

from __future__ import annotations

import json
import warnings
from abc import abstractmethod
from copy import deepcopy
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, cast

from ome_writers.backends._backend import ArrayBackend

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    import numpy as np

    from ome_writers._router import FrameRouter, PositionInfo
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

    def __init__(self) -> None:
        self._arrays: list[Any] = []
        self._image_group_paths: list[str] = []  # Parallel to _arrays, for metadata
        self._metadata_cache: dict[str, dict] = {}  # group path -> attrs dict
        self._finalized = False
        self._root: Path | None = None

    @abstractmethod
    def _get_writer(self) -> Literal["zarr", "tensorstore"] | Callable[..., Any]:
        """Return the writer to use for array creation.

        Subclasses can override to provide a custom CreateArrayFunc.
        """

    def _post_prepare(self, settings: AcquisitionSettings) -> None:
        """Hook called after yaozarrs creates the structure, before metadata caching.

        Subclasses can override to do additional setup (e.g., create streams).
        """

    def prepare(self, settings: AcquisitionSettings, router: FrameRouter) -> None:
        """Initialize OME-Zarr storage structure."""

        self._finalized = False
        root = Path(settings.root_path).expanduser().resolve()
        self._root = root
        positions = settings.positions

        # Build storage metadata
        storage_dims = settings.array_storage_dimensions
        shape = tuple(d.count if d.count is not None else 1 for d in storage_dims)
        dtype = settings.dtype
        chunks, shards = _get_chunks_and_shards(storage_dims)
        # this single image model is reused for all positions
        # (the underlying assumption is that we currently don't support inhomogeneous
        # shapes/dtypes across positions)
        image = _build_yaozarrs_image_model(storage_dims)

        # Get writer from hook (subclasses can override)
        writer = self._get_writer()

        # Plate mode
        if settings.plate is not None:
            # mapping of {(row, column): [(position_index, Position), ...]}
            well_positions = {}
            for i, pos in enumerate(positions):
                key = (pos.row, pos.column)
                well_positions.setdefault(key, []).append((i, pos))

            # Build plate metadata and arrays
            builder = PlateBuilder(
                root,
                plate=_build_yaozarrs_plate_model(settings.plate, well_positions),
                overwrite=settings.overwrite,
                chunks=chunks,
                shards=shards,
                writer=writer,
            )

            for (row, col), pos_list in well_positions.items():
                images_dict = {
                    pos.name: (image, [(shape, dtype)]) for _idx, pos in pos_list
                }
                builder.add_well(row=row, col=col, images=images_dict)

            _, all_arrays = builder.prepare()
            # Map position index to array path and store arrays
            self._image_group_paths = [
                f"{pos.row}/{pos.column}/{pos.name}" for pos in positions
            ]
            self._arrays = [
                all_arrays[f"{parent}/0"] for parent in self._image_group_paths
            ]

        # Single position
        elif len(positions) == 1:
            _, all_arrays = prepare_image(
                root,
                image,
                datasets=[(shape, dtype)],
                overwrite=settings.overwrite,
                chunks=chunks,
                shards=shards,
                writer=writer,  # type: ignore[arg-type]
            )
            self._image_group_paths = ["."]
            self._arrays = [all_arrays["0"]]

        # Multi-position (bf2raw layout)
        else:
            builder = Bf2RawBuilder(
                root,
                overwrite=settings.overwrite,
                chunks=chunks,
                shards=shards,
                writer=writer,
            )
            for pos in positions:
                builder.add_series(pos.name, image, [(shape, dtype)])

            _, all_arrays = builder.prepare()
            self._image_group_paths = [pos.name for pos in positions]
            self._arrays = [
                all_arrays[f"{parent}/0"] for parent in self._image_group_paths
            ]

        # Post-prepare hook (subclasses can do additional work)
        self._post_prepare(settings)

        # Cache metadata immediately after creation
        # This is used later for get_metadata() and update_metadata()
        self._cache_metadata_from_arrays(root)

    def write(
        self,
        position_info: PositionInfo,
        index: tuple[int, ...],
        frame: np.ndarray,
    ) -> None:
        """Write frame to specified location with auto-resize for unlimited dims."""
        if self._finalized:  # pragma: no cover
            raise RuntimeError("Cannot write after finalize().")
        if not self._arrays:  # pragma: no cover
            raise RuntimeError("Backend not prepared. Call prepare() first.")

        array = self._arrays[position_info[0]]

        # Resize if needed (index may be shorter than shape due to spatial dims)
        new_shape = list(array.shape)
        for i, idx_val in enumerate(index):
            new_shape[i] = max(new_shape[i], idx_val + 1)

        if new_shape != list(array.shape):
            self._resize(array, new_shape)

        self._write(array, index, frame)

    def get_metadata(self) -> dict[str, dict]:
        """Get metadata from all array groups in the zarr hierarchy.

        Returns a dict mapping group paths to their .zattrs contents.
        Each .zattrs dict typically contains:
        - "ome": yaozarrs v05.Image model (structured OME metadata)
        - "omero": dict with channel colors/names (if applicable)
        - Custom keys for non-standard metadata (timestamps, etc.)

        Returns
        -------
        dict[str, dict]
            Mapping of group paths to .zattrs dicts, or empty dict if not prepared.

        Examples
        --------
        For single position:
            {".": {"ome": {...}}}  # root group attrs

        For multi-position:
            {"pos0": {"ome": {...}}, "pos1": {"ome": {...}}}

        For plates:
            {"A/1/field_0": {"ome": {...}}, "A/2/field_0": {"ome": {...}}}
        """
        if not self._metadata_cache:  # pragma: no cover
            return {}

        return deepcopy(self._metadata_cache)

    def update_metadata(self, metadata: dict[str, dict]) -> None:
        """Update metadata for all array groups in the zarr hierarchy.

        Parameters
        ----------
        metadata : dict[str, dict]
            Mapping of group paths to .zattrs dicts. Keys should match those
            returned by get_metadata(). Values are dicts that will be written
            to the corresponding .zattrs files.

        Raises
        ------
        KeyError
            If a path in metadata doesn't correspond to a group.
        RuntimeError
            If backend is not prepared.
        """
        if not self._metadata_cache:  # pragma: no cover
            raise RuntimeError("Backend not prepared. Call prepare() first.")

        if self._root is None:  # pragma: no cover
            warnings.warn("Root path is unknown. Cannot update metadata.", stacklevel=2)
            return

        for path, attrs in metadata.items():
            if path not in self._metadata_cache:
                raise KeyError(f"Unknown path: {path}")

            if (zarr_json := self._root / path / "zarr.json").exists():
                data = cast("dict", json.loads(zarr_json.read_text()))
                data.setdefault("attributes", {}).update(attrs)
                zarr_json.write_text(json.dumps(data, indent=2))
                self._metadata_cache[path] = deepcopy(attrs)

    def finalize(self) -> None:
        """Flush and release resources."""
        if not self._finalized:
            self._arrays.clear()
            self._image_group_paths.clear()
            self._finalized = True

    def _write(self, array: Any, index: tuple[int, ...], frame: np.ndarray) -> None:
        """Write frame to array at specified index."""
        array[index] = frame

    def _resize(self, array: Any, new_shape: Sequence[int]) -> None:
        """Resize array to new shape."""
        array.resize(new_shape)

    def _cache_metadata_from_arrays(self, root: Path) -> None:
        """Cache metadata from parent groups.

        Reads zarr.json and caches their attributes in position order.
        """
        for parent_path in self._image_group_paths:
            if (zarr_json := root / parent_path / "zarr.json").exists():
                data = json.loads(zarr_json.read_text())
                self._metadata_cache[parent_path] = data.get("attributes", {})


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------


def _get_chunks_and_shards(
    dims: list[Dimension],
) -> tuple[tuple[int, ...], tuple[int, ...] | None]:
    """Compute chunk and shard sizes from dimensions."""
    chunks = []
    shards = []
    has_shards = False
    n = len(dims)

    for i, dim in enumerate(dims):
        # Chunks
        if dim.chunk_size is not None:
            chunks.append(dim.chunk_size)
        elif i >= n - 2:  # Last 2 dims (spatial)
            chunks.append(dim.count)
        else:
            chunks.append(1)

        # Shards
        if dim.shard_size is not None:
            shards.append(dim.shard_size)
            has_shards = True
        else:
            shards.append(chunks[-1])  # Default to chunk size

    return tuple(chunks), tuple(shards) if has_shards else None


def _build_yaozarrs_image_model(dims: list[Dimension]) -> v05.Image:
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


def _build_yaozarrs_plate_model(
    plate: Plate, well_positions: dict[tuple[str, str], list[tuple[int, Position]]]
) -> v05.Plate:
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
