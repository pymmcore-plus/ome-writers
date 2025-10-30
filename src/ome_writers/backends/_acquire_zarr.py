from __future__ import annotations

import gc
import importlib
import importlib.util
import json
import shutil
from contextlib import suppress
from pathlib import Path
from typing import TYPE_CHECKING

from tifffile import product
from typing_extensions import Self

from ome_writers._dimensions import dims_to_yaozarrs_v5
from ome_writers._plate import plate_to_acquire_zarr_plate
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
        # Initialize dimensions from MultiPositionOMEStream
        # NOTE: since acquire-zarr can save data in any order, we do not enforce
        # OME-NGFF order here. However, to always have valid OME-NGFF metadata, we will
        # patch the metadata at the end in _patch_group_metadata by adding a
        # transposition codec at the array level.
        self._init_dimensions(dimensions, enforce_ome_order=False)

        self._plate = plate

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
            self._create_with_hcs_plates(str(self._group_path), az_dims, dtype, plate)
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
        plate: Plate,
    ) -> None:
        """Create stream using acquire-zarr's HCS plates support.

        This method uses the hcs_plates parameter in StreamSettings to properly
        structure the data as an HCS plate with wells and fields of view.
        """
        # Create the acquire-zarr plate object
        plate_aqz = plate_to_acquire_zarr_plate(plate, az_dims, dtype)
        fields_per_well = plate.field_count or 1

        # Create stream with HCS configuration
        settings = self._aqz.StreamSettings(
            store_path=store_path,
            version=self._aqz.ZarrVersion.V3,
            hcs_plates=[plate_aqz],
        )

        # keep a strong reference (avoid segfaults)
        self._az_settings_keepalive = settings

        self._stream = self._aqz.ZarrStream(settings)

        # Build indices mapping for writing
        # We need to account for the dimension order when building indices.
        # The dimensions may be in any order (e.g., [t, p, c] or [p, t, c]),
        # so we need to find where the position dimension is and iterate
        # in the correct order.

        # Build ranges for all non-spatial dimensions in the order they appear
        all_dims_ranges: list[range] = []
        position_dim_index = -1
        for d in self.acquisition_order_dims:
            if d.label in "xy":
                continue
            if d.label == "p":
                position_dim_index = len(all_dims_ranges)
            all_dims_ranges.append(range(d.size))

        self._indices = {}
        # remove any spaces from the plate name for acquire-zarr
        plate_path = (plate.name or "plate").replace(" ", "_")
        for idx, combo in enumerate(product(*all_dims_ranges)):
            # Extract position index and non-position indices
            # Example:
            # idx, combo = 0, (0, 0, 0)
            # idx, combo = 1, (0, 0, 1)
            # idx, combo = 2, (0, 1, 0)
            # ...
            if position_dim_index >= 0:
                pos_idx = combo[position_dim_index]
                non_p_idx = tuple(
                    v for i, v in enumerate(combo) if i != position_dim_index
                )
            else:
                pos_idx = 0
                non_p_idx = combo

            # Map position index to well/FOV path
            well_idx = pos_idx // fields_per_well
            fov_idx_in_well = pos_idx % fields_per_well

            well_pos = plate.wells[well_idx]
            row_name, col_name = well_pos.path.split("/")
            fov_key = f"fov{fov_idx_in_well}" if fields_per_well > 1 else "0"
            array_key = f"{plate_path}/{row_name}/{col_name}/{fov_key}"

            self._indices[idx] = (array_key, non_p_idx)

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

        attrs = dims_to_yaozarrs_v5(
            {str(i): reordered_dims for i in range(self.num_positions)}
        )
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


        current_meta.setdefault("attributes", {}).update(attrs.model_dump())
        from rich import print
        print(current_meta)
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
        for pos in range(self.num_positions):
            array_zarr_json = Path(self._group_path) / str(pos) / "zarr.json"
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
            self._stream.append(frame, key=position_key)

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
