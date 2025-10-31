from __future__ import annotations

import gc
import importlib
import importlib.util
import json
import shutil
from contextlib import suppress
from pathlib import Path
from typing import TYPE_CHECKING

from typing_extensions import Self

from ome_writers._dimensions import dims_to_yaozarrs_v5
from ome_writers._plate import plate_to_acquire_zarr
from ome_writers._stream_base import MultiPositionOMEStream
from ome_writers._util import reorder_to_ome_ngff

if TYPE_CHECKING:
    from collections.abc import Sequence

    import acquire_zarr
    import numpy as np

    from ome_writers._dimensions import Dimension
    from ome_writers._plate import Plate


class AcquireZarrStream(MultiPositionOMEStream):
    @classmethod
    def is_available(cls) -> bool:
        """Check if the acquire-zarr package is available."""
        return importlib.util.find_spec("acquire_zarr") is not None

    def __init__(self) -> None:
        try:
            import acquire_zarr
        except ImportError as e:
            msg = (
                "AcquireZarrStream requires the acquire-zarr package: "
                "pip install acquire-zarr"
            )
            raise ImportError(msg) from e
        self._aqz = acquire_zarr
        super().__init__()
        self._stream: acquire_zarr.ZarrStream | None = None

    def create(
        self,
        path: str,
        dtype: np.dtype,
        dimensions: Sequence[Dimension],
        plate: Plate | None = None,
        *,
        overwrite: bool = False,
    ) -> Self:
        # to store plate information
        self._plate = plate
        self._plate_position_keys: dict[int, str] = {}

        # Initialize dimensions from MultiPositionOMEStream
        # NOTE: since acquire-zarr can save data in any order, we do not enforce
        # OME-NGFF order here. However, to always have valid OME-NGFF metadata, we will
        # patch the metadata at the end in _patch_group_metadata by adding a
        # transposition codec at the array level.
        self._init_dimensions(dimensions, enforce_ome_order=False)

        self._group_path = Path(self._normalize_path(path))

        # Check if directory exists and handle overwrite parameter
        if self._group_path.exists():
            if not overwrite:
                raise FileExistsError(
                    f"Directory {self._group_path} already exists. "
                    "Use overwrite=True to overwrite it."
                )
            shutil.rmtree(self._group_path)

        # Dimensions will be the same across all positions, so we can create them once
        az_dims = [self._dim_toaqz_dim(dim) for dim in self.storage_order_dims]
        # keep a strong reference (avoid segfaults)
        self._az_dims_keepalive = az_dims

        if plate is not None:
            # Use HCS plate mode with acquire-zarr's hcs_plates parameter
            self._create_with_hcs_plates(str(self._group_path), az_dims, dtype)
        else:
            # Create array settings for each position
            self._create(az_dims, dtype)

        return self

    def _create(
        self,
        az_dims: list[acquire_zarr.Dimension],
        dtype: np.dtype,
    ) -> None:
        """Create stream without HCS plates (standard multi-position mode)."""
        # Create AcquireZarr array settings for each position
        az_array_settings = [
            self._aqz_pos_array(str(pos_idx), az_dims, dtype)
            for pos_idx in range(self.num_positions)
        ]

        for arr in az_array_settings:
            for d in arr.dimensions:
                assert d.chunk_size_px > 0, (d.name, d.chunk_size_px)
                assert d.shard_size_chunks > 0, (d.name, d.shard_size_chunks)

        # Keep a strong reference (avoid segfaults)
        self._az_arrays_keepalive = az_array_settings

        # Create streams for each position
        settings = self._aqz.StreamSettings(
            arrays=az_array_settings,
            store_path=str(self._group_path),
            version=self._aqz.ZarrVersion.V3,
        )
        # Keep a strong reference (avoid segfaults)
        self._az_settings_keepalive = settings

        self._stream = self._aqz.ZarrStream(settings)

    def _create_with_hcs_plates(
        self,
        store_path: str,
        az_dims: list[acquire_zarr.Dimension],
        dtype: np.dtype,
    ) -> None:
        """Create stream using acquire-zarr's HCS plates support.

        This method uses the hcs_plates parameter in StreamSettings to properly
        structure the data as an HCS plate with wells and fields of view.
        """
        if self._plate is None:
            raise ValueError("Plate must be provided for HCS plate mode.")

        # Create the acquire-zarr plate object
        plate_aqz = plate_to_acquire_zarr(self._plate, az_dims, dtype)
        fields_per_well = self._plate.field_count or 1

        # Create stream with HCS configuration
        settings = self._aqz.StreamSettings(
            store_path=store_path,
            version=self._aqz.ZarrVersion.V3,
            hcs_plates=[plate_aqz],
        )

        # keep a strong reference (avoid segfaults)
        self._az_settings_keepalive = settings

        self._stream = self._aqz.ZarrStream(settings)

        # Build mapping from position index to well/field key for HCS
        plate_path = (self._plate.name or "plate").replace(" ", "_")
        pos_idx = 0
        for well in self._plate.wells:
            for field_idx in range(fields_per_well):
                if fields_per_well > 1:
                    fov_key = f"fov{field_idx}"
                else:
                    fov_key = "0"
                key = f"{plate_path}/{well.path}/{fov_key}"
                self._plate_position_keys[pos_idx] = key
                pos_idx += 1

    def _patch_metadata_to_ngff_v05(self) -> None:
        """Patch the Zarr group metadata to ensure OME-NGFF v0.5 compliance.

        This is necessary for two main reasons:

        1. To support datasets containing multiple positions.
        2. To ensure data conforms to the OME-NGFF TCZYX dimension specification.

        This method writes OME-NGFF v0.5-compliant metadata for both the group and
        its position arrays using the `dims_to_yaozarrs_v5()` utility function,
        regardless of the order in which the data was originally stored.
        Since `acquire-zarr` allows writing data to disk in arbitrary dimension orders
        (e.g., TZCYX), the stored arrays may not follow the mandated OME-NGFF TCZYX
        convention.

        When the stored order differs from TCZYX, this method inserts a *transpose*
        codec into each position's array metadata to indicate how the data should be
        reordered on read, ensuring it can always be interpreted correctly in
        OME-NGFF order.
        """
        reordered_dims = reorder_to_ome_ngff(list(self.storage_order_dims))
        if reordered_dims != self.storage_order_dims:
            original_labels = [d.label for d in self.storage_order_dims]
            ome_labels = [d.label for d in reordered_dims]
            transpose_order = [
                ome_labels.index(label) for label in original_labels
            ] + list(range(len(original_labels), len(reordered_dims)))
        else:
            transpose_order = None

        if self._plate is not None and self._plate_position_keys:
            position_paths = list(self._plate_position_keys.values())
        else:
            position_paths = [str(i) for i in range(self.num_positions)]

        attrs = dims_to_yaozarrs_v5(dict.fromkeys(position_paths, reordered_dims))
        zarr_json = Path(self._group_path) / "zarr.json"
        current_meta: dict = {
            "consolidated_metadata": None,
            "node_type": "group",
            "zarr_format": 3,
        }
        if zarr_json.exists():
            with suppress(json.JSONDecodeError):
                with open(zarr_json) as f:
                    current_meta = json.load(f)

        current_meta.setdefault("attributes", {}).update({"ome": attrs.model_dump()})
        zarr_json.write_text(json.dumps(current_meta, indent=2))

        if transpose_order is not None:
            self._rearrange_positions_metadata(reordered_dims, transpose_order)

    def _rearrange_positions_metadata(
        self, reordered_dims: Sequence[Dimension], transpose_order: list[int] | None
    ) -> None:
        """Reorganize metadata for each position array.

        This method uses the [zarr transpose codec](https://zarr-specs.readthedocs.io/en/latest/v3/codecs/transpose/index.html)
        to indicate the transposition needed to go from the stored order to the
        OME-NGFF order and will allow to read the data back in OME-NGFF order.

        (As suggested in https://github.com/acquire-project/acquire-zarr/issues/171#issuecomment-3458544335).
        """
        if self._plate is not None and self._plate_position_keys:
            position_paths = list(self._plate_position_keys.values())
        else:
            position_paths = [str(i) for i in range(self.num_positions)]

        for path in position_paths:
            array_zarr_json = Path(self._group_path) / path / "zarr.json"
            if array_zarr_json.exists():
                with open(array_zarr_json) as f:
                    array_meta = json.load(f)

                array_meta["dimension_names"] = [d.label for d in reordered_dims]
                array_meta["shape"] = [d.size for d in reordered_dims]
                chunk_shape = [
                    d.chunk_size if d.chunk_size is not None else d.size
                    for d in reordered_dims
                ]
                array_meta["chunk_grid"]["configuration"]["chunk_shape"] = chunk_shape
                array_meta["codecs"][0]["configuration"]["chunk_shape"] = chunk_shape

                if transpose_order:
                    array_meta["codecs"][0]["configuration"]["codecs"].insert(
                        0,
                        {
                            "name": "transpose",
                            "configuration": {"order": transpose_order},
                        },
                    )

                with open(array_zarr_json, "w") as f:
                    json.dump(array_meta, f, indent=2)

    def _write_to_backend(
        self, position_key: str, index: tuple[int, ...], frame: np.ndarray
    ) -> None:
        """AcquireZarr-specific write implementation.

        NOTE: For AcquireZarr, frames are written sequentially, so index is not used.
        """
        if self._stream is not None:
            if self._plate is not None and self._plate_position_keys:
                key = self._plate_position_keys.get(int(position_key), position_key)
            else:
                key = position_key
            self._stream.append(frame, key=key)

    def flush(self) -> None:
        if not self._stream:  # pragma: no cover
            raise RuntimeError("Stream is closed or uninitialized. Cannot flush.")
        # Flush the stream to ensure all data is written to disk.
        self._stream.close()
        self._stream = None
        gc.collect()
        self._patch_metadata_to_ngff_v05()

    def is_active(self) -> bool:
        if self._stream is None:
            return False
        return self._stream.is_active()

    def _dim_toaqz_dim(
        self,
        dim: Dimension,
        shard_size_chunks: int = 1,
    ) -> acquire_zarr.Dimension:
        return self._aqz.Dimension(
            name=dim.label,
            kind=getattr(self._aqz.DimensionType, dim.ome_dim_type.upper()),
            array_size_px=dim.size,
            chunk_size_px=(dim.chunk_size if dim.chunk_size is not None else dim.size),
            shard_size_chunks=shard_size_chunks,
        )

    def _aqz_pos_array(
        self,
        position_index: str,
        dimensions: list[acquire_zarr.Dimension],
        dtype: np.dtype,
    ) -> acquire_zarr.ArraySettings:
        """Create an AcquireZarr ArraySettings for a position or well/field.

        Parameters
        ----------
        position_index : str
            The output key for the array. For non-plate mode, this is the position
            index (as string, e.g. "0" or "1"), for plate mode, this is the
            well/field path (e.g., "A/01/0").
        dimensions : list[acquire_zarr.Dimension]
            The dimensions for this array.
        dtype : np.dtype
            The data type for this array.

        Returns
        -------
        acquire_zarr.ArraySettings
            The array settings for this position or well/field.
        """
        return self._aqz.ArraySettings(
            output_key=str(position_index), dimensions=dimensions, data_type=dtype
        )
