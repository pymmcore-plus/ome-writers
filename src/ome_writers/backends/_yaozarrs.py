from __future__ import annotations

import importlib.util
from abc import abstractmethod
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

from typing_extensions import Self

from ome_writers._ngff_metadata import build_yaozarrs_image_metadata_v05
from ome_writers._stream_base import MultiPositionOMEStream

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    import numpy as np
    from yaozarrs.v05 import Image, PlateDef

    from ome_writers._dimensions import Dimension
    from ome_writers._stream_base import PlateType


class _YaozarrsStreamBase(MultiPositionOMEStream):
    """OME-Zarr writer using the yaozarrs library.

    This stream uses yaozarrs to create OME-Zarr v0.5 compatible stores with
    proper metadata. It supports both zarr-python and tensorstore backends
    through yaozarrs' unified API.

    For multi-position data, this uses the bioformats2raw layout pattern where
    each position is a separate Image in the hierarchy. For plate data, it uses
    the HCS (High-Content Screening) layout with proper well/field structure.
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
            from yaozarrs import v05
            from yaozarrs.write.v05 import Bf2RawBuilder, PlateBuilder, prepare_image
        except ImportError as e:
            msg = (
                "YaozarrsStream requires yaozarrs with write support: "
                "`pip install yaozarrs[write-tensorstore]` or"
                "`pip install yaozarrs[write-zarr]`."
            )
            raise ImportError(msg) from e

        self._prepare_image = prepare_image
        self._PlateBuilder = PlateBuilder
        self._Bf2RawBuilder = Bf2RawBuilder
        self._v05 = v05

        super().__init__()

        self._group_path: Path | None = None
        self._arrays: dict[str, Any] = {}

    @abstractmethod
    def _get_writer(self) -> Literal["tensorstore", "zarr"] | Callable: ...

    def create(
        self,
        path: str,
        dtype: np.dtype,
        dimensions: Sequence[Dimension],
        *,
        plate: PlateType = None,
        overwrite: bool = False,
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
        plate : yaozarrs.v05.PlateDef | None, optional
            Plate metadata for HCS layout. If provided, creates a plate structure
            instead of a simple multi-position layout. Default is None.

        Returns
        -------
        Self
            The configured stream instance.
        """
        # Use MultiPositionOMEStream with NGFF ordering
        self._configure_dimensions(dimensions, ngff_order=True)

        # Validate plate type - yaozarrs backend only supports PlateDef
        if plate is not None and not isinstance(plate, self._v05.PlateDef):
            msg = (
                "YaozarrsStream only supports yaozarrs.v05.PlateDef for plate metadata."
                f" Received: {type(plate).__name__}"
            )
            raise TypeError(msg)

        writer = self._get_writer()
        self._group_path = Path(self._normalize_path(path))

        # Get shape from NGFF-ordered dimensions
        shape = tuple(d.size for d in self.storage_dims)
        # For chunking, if not specified, use full size for last 2 dims (spatial), 1\
        # for others if not specified
        chunks = tuple(
            d.chunk_size or (d.size if i >= len(self.storage_dims) - 2 else 1)
            for i, d in enumerate(self.storage_dims)
        )
        # Build the Image metadata with NGFF-ordered dimensions
        image = build_yaozarrs_image_metadata_v05(self.storage_dims)

        self._arrays = self._prepare_arrays(
            image, plate, shape, dtype, chunks, writer, overwrite
        )

        return self

    def flush(self) -> None:
        """Flush pending writes and close the stream."""
        self._arrays.clear()

    def is_active(self) -> bool:
        """Return True if the stream has active arrays."""
        return bool(self._arrays)

    def _prepare_arrays(
        self,
        image: Image,
        plate: PlateDef | None,
        shape: tuple[int, ...],
        dtype: np.dtype,
        chunks: tuple[int, ...],
        writer: Literal["tensorstore", "zarr"] | Callable,
        overwrite: bool,
    ) -> dict[str, Any]:
        """Prepare arrays for the appropriate layout (plate, single, or multi-position).

        Parameters
        ----------
        image : yaozarrs.v05.Image
            Yaozarrs Image metadata.
        plate : yaozarrs.v05.PlateDef | None
            Plate definition for HCS layout. If None, creates single or multi-position
            layout.
        shape : tuple[int, ...]
            Shape of arrays in NGFF order.
        dtype : np.dtype
            Data type for arrays.
        chunks : tuple[int, ...]
            Chunk sizes for arrays.
        writer : Literal["tensorstore", "zarr"] | Callable
            Writer backend to use.
        overwrite : bool
            Whether to overwrite existing data.

        Returns
        -------
        dict[str, Any]
            Dictionary mapping position indices to array handles.
        """
        # Validate plate dimensions match the position count
        if plate is not None:
            expected_positions = len(plate.wells) * (plate.field_count or 1)
            if self.num_positions != expected_positions:
                msg = (
                    f"Position dimension size ({self.num_positions}) does not match "
                    f"plate structure ({len(plate.wells)} wells * "
                    f"{plate.field_count or 1} fields = {expected_positions} positions)"
                )
                raise ValueError(msg)

        if plate is not None:
            return self._prepare_plate_arrays(
                image, plate, shape, dtype, chunks, writer, overwrite
            )
        elif self.num_positions == 1:
            return self._prepare_single_position_arrays(
                image, shape, dtype, chunks, writer, overwrite
            )
        else:
            return self._prepare_multi_position_arrays(
                image, shape, dtype, chunks, writer, overwrite
            )

    def _prepare_plate_arrays(
        self,
        image: Image,
        plate: PlateDef,
        shape: tuple[int, ...],
        dtype: np.dtype,
        chunks: tuple[int, ...],
        writer: Literal["tensorstore", "zarr"] | Callable,
        overwrite: bool,
    ) -> dict[str, Any]:
        """Prepare arrays for HCS plate layout.

        Parameters
        ----------
        image : yaozarrs.v05.Image
            Yaozarrs Image metadata.
        plate : yaozarrs.v05.PlateDef
            Plate definition with wells and fields.
        shape : tuple[int, ...]
            Shape of arrays in NGFF order.
        dtype : np.dtype
            Data type for arrays.
        chunks : tuple[int, ...]
            Chunk sizes for arrays.
        writer : Literal["tensorstore", "zarr"] | Callable
            Writer backend to use.
        overwrite : bool
            Whether to overwrite existing data.
        """
        if self._group_path is None:
            raise RuntimeError("Stream has not been created yet.")

        plate_metadata = self._v05.Plate(version="0.5", plate=plate)
        plate_builder = self._PlateBuilder(
            self._group_path,
            plate=plate_metadata,
            overwrite=overwrite,
            chunks=chunks,
            writer=writer,
        )

        # Add wells to the plate
        # Build mapping from position index to well/field path
        position_to_array_key: dict[str, str] = {}
        position_idx = 0
        fields_per_well = plate.field_count or 1

        for well in plate.wells:
            row_name, col_name = well.path.split("/")
            # Create field entries for this well
            well_images = {}
            for field_idx in range(fields_per_well):
                field_key = str(field_idx)
                well_images[field_key] = (image, [(shape, dtype)])

                # Map position index to array key (well_path/field_idx)
                # PlateBuilder creates keys like "row/col/field/dataset"
                # so we need to append "/0" for the first (and only) dataset
                array_key = f"{well.path}/{field_idx}/0"
                position_to_array_key[str(position_idx)] = array_key
                position_idx += 1

            plate_builder.add_well(row=row_name, col=col_name, images=well_images)

        # Prepare the plate and remap arrays to use position indices as keys
        _, plate_arrays = plate_builder.prepare()

        # Remap from PlateBuilder's keys (e.g., "A/1/0/0") to position indices
        remapped_arrays = {}
        for pos_idx_str, array_key in position_to_array_key.items():
            remapped_arrays[pos_idx_str] = plate_arrays[array_key]

        return remapped_arrays

    def _prepare_single_position_arrays(
        self,
        image: Image,
        shape: tuple[int, ...],
        dtype: np.dtype,
        chunks: tuple[int, ...],
        writer: Literal["tensorstore", "zarr"] | Callable,
        overwrite: bool,
    ) -> dict[str, Any]:
        """Prepare arrays for single position data.

        Parameters
        ----------
        image : yaozarrs.v05.Image
            Yaozarrs Image metadata.
        shape : tuple[int, ...]
            Shape of arrays in NGFF order.
        dtype : np.dtype
            Data type for arrays.
        chunks : tuple[int, ...]
            Chunk sizes for arrays.
        writer : Literal["tensorstore", "zarr"] | Callable
            Writer backend to use.
        overwrite : bool
            Whether to overwrite existing data.

        Returns
        -------
        dict[str, Any]
            Dictionary mapping position index "0" to array handle.
        """
        if self._group_path is None:
            raise RuntimeError("Stream has not been created yet.")

        _, arrays = self._prepare_image(
            self._group_path,
            image,
            datasets=[(shape, dtype)],
            overwrite=overwrite,
            chunks=chunks,
            writer=writer,
        )
        return arrays  # type: ignore

    def _prepare_multi_position_arrays(
        self,
        image: Image,
        shape: tuple[int, ...],
        dtype: np.dtype,
        chunks: tuple[int, ...],
        writer: Literal["tensorstore", "zarr"] | Callable,
        overwrite: bool,
    ) -> dict[str, Any]:
        """Prepare arrays for multi-position data using bioformats2raw layout.

        Parameters
        ----------
        image : yaozarrs.v05.Image
            Yaozarrs Image metadata.
        shape : tuple[int, ...]
            Shape of arrays in NGFF order.
        dtype : np.dtype
            Data type for arrays.
        chunks : tuple[int, ...]
            Chunk sizes for arrays.
        writer : Literal["tensorstore", "zarr"] | Callable
            Writer backend to use.
        overwrite : bool
            Whether to overwrite existing data.

        Returns
        -------
        dict[str, Any]
            Dictionary mapping position indices to array handles.
        """
        if self._group_path is None:
            raise RuntimeError("Stream has not been created yet.")

        builder = self._Bf2RawBuilder(
            self._group_path,
            overwrite=overwrite,
            chunks=chunks,
            writer=writer,
        )

        # Add each position as a separate series
        for pos_idx in range(self.num_positions):
            array_key = str(pos_idx)
            builder.add_series(array_key, image, [(shape, dtype)])

        # Prepare all arrays
        _, all_arrays = builder.prepare()

        # Remap Bf2RawBuilder keys ("0/0", "1/0", etc.) to position indices
        arrays = {}
        for pos_idx in range(self.num_positions):
            array_key = str(pos_idx)
            # The dataset path within each image is "0" (first/only resolution)
            arrays[array_key] = all_arrays[f"{array_key}/0"]

        return arrays


class TensorStoreZarrStream(_YaozarrsStreamBase):
    """OME-Zarr writer using yaozarrs with tensorstore backend.

    This stream creates OME-Zarr v0.5 compatible stores using tensorstore for
    efficient array I/O. Data is always stored in NGFF canonical order (tczyx).
    """

    def __init__(self) -> None:
        super().__init__()
        self._futures: list[Any] = []

    @classmethod
    def is_available(cls) -> bool:
        """Check if yaozarrs and tensorstore are available."""
        if not super().is_available():  # pragma: no cover
            return False
        return importlib.util.find_spec("tensorstore") is not None

    def _get_writer(self) -> Literal["tensorstore"]:
        return "tensorstore"

    def _write_to_backend(
        self, position_key: str, index: tuple[int, ...], frame: np.ndarray
    ) -> None:
        """Write frame to the yaozarrs-created array.

        The index is already in storage (NGFF) order thanks to base class.
        """
        array = self._arrays[position_key]
        future = array[index].write(frame)
        self._futures.append(future)

    def flush(self) -> None:
        for future in self._futures:
            future.result()
        return super().flush()


class ZarrPythonStream(_YaozarrsStreamBase):
    """OME-Zarr writer using yaozarrs with zarr-python backend.

    This stream creates OME-Zarr v0.5 compatible stores using zarr-python for
    array I/O. Data is always stored in NGFF canonical order (tczyx).
    """

    @classmethod
    def is_available(cls) -> bool:
        """Check if yaozarrs and zarr-python are available."""
        if not super().is_available():  # pragma: no cover
            return False
        try:
            import zarr
            from packaging.version import Version

            return bool(Version(zarr.__version__) >= Version("3.0.0"))
        except ImportError:
            return False

    def _get_writer(self) -> Literal["zarr"]:
        return "zarr"

    def _write_to_backend(
        self, position_key: str, index: tuple[int, ...], frame: np.ndarray
    ) -> None:
        """Write frame to the yaozarrs-created array.

        The index is already in storage (NGFF) order thanks to base class.
        """
        self._arrays[position_key][index] = frame


# MESSY PROOF OF PRINCIPLE
# class AcquireZarrStream(_YaozarrsStreamBase):
#     """OME-Zarr writer using yaozarrs with tensorstore backend.

#     This stream creates OME-Zarr v0.5 compatible stores using tensorstore for
#     efficient array I/O. Data is always stored in NGFF canonical order (tczyx).
#     """

#     @classmethod
#     def is_available(cls) -> bool:
#         """Check if yaozarrs and tensorstore are available."""
#         if not super().is_available():
#             return False
#         return importlib.util.find_spec("tensorstore") is not None

#     def _init_dimensions(
#         self, dimensions: Sequence[Dimension], ngff_order: bool = True
#     ) -> None:
#         # Force NGFF order for AcquireZarrStream
#         super()._init_dimensions(dimensions, ngff_order=True)

#         def _create_stream(
#             path: Path,
#             shape: tuple[int, ...],
#             dtype: Any,
#             chunks: tuple[int, ...],
#             *,
#             shards: tuple[int, ...] | None,  # = None,
#             overwrite: bool,  # = False,
#             compression: str,  # = "blosc-zstd",
#             dimension_names: list[str] | None,  # = None,
#         ) -> Any:
#             # Dimensions will be the same across all positions,
#             # so we can create them once
#             az_dims = [_dim_toaqz_dim(dim) for dim in self.storage_order_dims]

#             # Create AcquireZarr array settings for each position
#             az_array_settings = [
#                 _aqz_pos_array(pos_idx, az_dims, dtype)
#                 for pos_idx in range(self.num_positions)
#             ]

#             for arr in az_array_settings:
#                 for d in arr.dimensions:
#                     assert d.chunk_size_px > 0, (d.name, d.chunk_size_px)
#                     assert d.shard_size_chunks > 0, (d.name, d.shard_size_chunks)

#             # keep a strong reference (avoid segfaults)
#             self._az_dims_keepalive = az_dims
#             self._az_arrays_keepalive = az_array_settings

#             # Create streams for each position
#             settings = az.StreamSettings(
#                 arrays=az_array_settings,
#                 store_path=str(self._group_path),
#                 version=az.ZarrVersion.V3,
#             )
#             return az.ZarrStream(settings)

#         self._create_stream = _create_stream

#     def _get_writer(self) -> Callable:  # type: ignore[override]
#         return self._create_stream

#     def _write_to_backend(
#         self, position_key: str, index: tuple[int, ...], frame: np.ndarray
#     ) -> None:
#         """Write frame to the yaozarrs-created array.

#         The index is already in storage (NGFF) order thanks to base class.
#         """
#         self._arrays[position_key].append(frame, key=position_key)

#     def flush(self) -> None:
#         super().flush()
#         _write_metadata(self._group_path, self._ome_model)

# def _dim_toaqz_dim(
#     dim: Dimension,
#     shard_size_chunks: int = 1,
# ) -> az.Dimension:
#     return az.Dimension(
#         name=dim.label,
#         kind=getattr(az.DimensionType, dim.ome_dim_type.upper()),
#         array_size_px=dim.size,
#         chunk_size_px=(dim.chunk_size if dim.chunk_size is not None else dim.size),
#         shard_size_chunks=shard_size_chunks,
#     )


# def _aqz_pos_array(
#     position_index: int,
#     dimensions: list[az.Dimension],
#     dtype: np.dtype,
# ) -> az.ArraySettings:
#     """Create an AcquireZarr ArraySettings for a position."""
#     return az.ArraySettings(
#         # this matches the position index key from the base class
#         output_key=str(position_index),
#         dimensions=dimensions,
#         data_type=dtype,
#     )


# def _write_metadata(
#     dest_path: Path,
#     ome_model: BaseModel | None = None,
#     indent: int = 2,
# ) -> None:
#     """Create a zarr group directory with optional OME metadata in zarr.json."""
#     zarr_json_path = dest_path / "zarr.json"
#     dest_path.mkdir(parents=True, exist_ok=True)
#     zarr_json: dict[str, Any] = {
#         "zarr_format": 3,
#         "node_type": "group",
#     }
#     if ome_model is not None:
#         zarr_json["attributes"] = {
#             "ome": ome_model.model_dump(mode="json", exclude_none=True),
#         }
#     zarr_json_path.write_text(json.dumps(zarr_json, indent=indent))
