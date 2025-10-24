from __future__ import annotations

import gc
import importlib
import importlib.util
import json
import shutil
from contextlib import suppress
from itertools import product
from pathlib import Path
from typing import TYPE_CHECKING, Any

from typing_extensions import Self

from ome_writers._stream_base import MultiPositionOMEStream

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
        self._plate: Plate | None = None

    def create(
        self,
        path: str,
        dtype: np.dtype,
        dimensions: Sequence[Dimension],
        *,
        overwrite: bool = False,
        downsampling_method: acquire_zarr.DownsamplingMethod | None = None,
        plate: Plate | None = None,
        **kwargs: Any,
    ) -> Self:
        """Create a new stream for writing data.

        Parameters
        ----------
        path : str
            Path to the output directory.
        dtype : np.dtype
            NumPy data type for the image data.
        dimensions : Sequence[Dimension]
            Sequence of dimension information describing the data structure.
        overwrite : bool, optional
            Whether to overwrite existing directories. Default is False.
        plate : Plate | None, optional
            Optional plate metadata for organizing multi-well acquisitions.
            If provided, the store will be structured as a plate with wells.
        downsampling_method : acquire_zarr.DownsamplingMethod | None, optional
            The method to use when generating multiscale (downsampled) image pyramids.
            If provided, acquire-zarr will automatically create additional downsampled
            datasets for each array until the final spatial (XY) resolution
            approximately matches the chunk size. Available methods include "mean",
            "min", "max" and "decimate".
            By default, None (no downsampling is performed).

        Returns
        -------
        Self
            The instance of the stream, allowing for chaining.
        """
        self._downsampling_method = downsampling_method
        self._plate = plate

        # Use MultiPositionOMEStream to handle position logic
        num_positions, non_position_dims = self._init_positions(dimensions)
        self._group_path = Path(self._normalize_path(path))

        # Save original dimensions for index mapping
        self._all_dimensions = dimensions

        # Check if directory exists and handle overwrite parameter
        if self._group_path.exists():
            if not overwrite:
                raise FileExistsError(
                    f"Directory {self._group_path} already exists. "
                    "Use overwrite=True to overwrite it."
                )
            shutil.rmtree(self._group_path)

        # Dimensions will be the same across all positions, so we can create them once
        az_dims = [self._dim_toaqz_dim(dim) for dim in non_position_dims]

        if plate is not None:
            # Use HCS plate mode with acquire-zarr's hcs_plates parameter
            self._create_with_hcs_plates(str(self._group_path), az_dims, dtype, plate)
        else:
            # Create array settings for each position
            self._create(str(self._group_path), az_dims, dtype, num_positions)

        return self

    def _create(
        self,
        store_path: str,
        az_dims: list[acquire_zarr.Dimension],
        dtype: np.dtype,
        num_positions: int,
    ) -> None:
        """Create stream without HCS plates (standard multi-position mode)."""
        # Create AcquireZarr array settings for each position
        array_settings: list[acquire_zarr.ArraySettings] = []
        for pos_idx in range(num_positions):
            array_key = str(pos_idx)
            array_settings.append(self._aqz_pos_array(array_key, az_dims, dtype))

        for arr in array_settings:
            for d in arr.dimensions:
                assert d.chunk_size_px > 0, (d.name, d.chunk_size_px)
                assert d.shard_size_chunks > 0, (d.name, d.shard_size_chunks)

        # Create streams for each position
        settings = self._aqz.StreamSettings(
            arrays=array_settings,
            store_path=store_path,
            version=self._aqz.ZarrVersion.V3,
        )
        self._az_settings_keepalive = settings
        self._stream = self._aqz.ZarrStream(settings)
        self._patch_group_metadata()

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
        fields_per_well = plate.field_count or 1

        # Create array settings for each field of view
        fovs_arrays_settings: list[acquire_zarr.ArraySettings] = []
        for fov_idx in range(fields_per_well):
            fov_key = f"fov{fov_idx}" if fields_per_well > 1 else "0"
            fov_array = self._aqz.ArraySettings(
                output_key=fov_key,
                dimensions=az_dims,
                data_type=dtype,
                downsampling_method=self._downsampling_method,
            )
            fovs_arrays_settings.append(fov_array)

        # Create acquisition metadata if provided
        acquisitions = []
        if plate.acquisitions:
            for acq in plate.acquisitions:
                acquisitions.append(
                    self._aqz.Acquisition(
                        id=acq.id,
                        name=acq.name or f"Acquisition {acq.id}",
                        start_time=acq.start_time,
                        end_time=acq.end_time,
                    )
                )

        # Create wells with fields of view
        wells = []
        for well_pos in plate.wells:
            row_name, col_name = well_pos.path.split("/")

            # Create field of view entries for this well
            fov_entries = []
            for fov_idx, fov_array in enumerate(fovs_arrays_settings):
                fov_key = f"fov{fov_idx}" if fields_per_well > 1 else "0"
                fov_entries.append(
                    self._aqz.FieldOfView(
                        path=fov_key,
                        acquisition_id=acquisitions[0].id if acquisitions else None,
                        array_settings=fov_array,
                    )
                )

            well = self._aqz.Well(
                row_name=row_name,
                column_name=col_name,
                images=fov_entries,
            )
            wells.append(well)

        # Create the HCS plate
        # remove any spaces from the plate name for acquire-zarr
        plate_path = (plate.name or "plate").replace(" ", "_")
        plate_aqz = self._aqz.Plate(
            path=plate_path,
            name=plate.name,
            row_names=list(plate.rows),
            column_names=list(plate.columns),
            wells=wells,
            acquisitions=acquisitions if acquisitions else None,
        )

        # Create stream with HCS configuration
        settings = self._aqz.StreamSettings(
            store_path=store_path,
            version=self._aqz.ZarrVersion.V3,
            hcs_plates=[plate_aqz],
        )
        self._az_settings_keepalive = settings
        self._stream = self._aqz.ZarrStream(settings)

        # Build indices mapping for writing
        # We need to account for the dimension order when building indices.
        # The dimensions may be in any order (e.g., [t, p, c] or [p, t, c]),
        # so we need to find where the position dimension is and iterate
        # in the correct order.

        # Build ranges for all non-spatial dimensions in the order they appear
        all_dims_ranges = []
        position_dim_index = -1
        for d in self._all_dimensions:
            if d.label in "xy":
                continue
            if d.label == "p":
                position_dim_index = len(all_dims_ranges)
            all_dims_ranges.append(range(d.size))

        self._indices = {}
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

    def _patch_group_metadata(self) -> None:
        """Patch the group metadata with OME NGFF 0.5 metadata.

        This method exists because there are cases in which the standard acquire-zarr
        API is not flexible enough to handle all the cases we need (such as multiple
        positions).  For HCS plates, acquire-zarr handles metadata automatically,
        so we don't need to patch anything.
        """
        if self._plate is not None:
            # HCS plates are handled by acquire-zarr's hcs_plates parameter
            # No need to patch metadata
            return

        # Write standard multi-position metadata
        dims = self._non_position_dims
        from ome_writers._dimensions import dims_to_yaozarrs_v5

        image = dims_to_yaozarrs_v5({str(i): dims for i in range(self._num_positions)})
        attrs = {"ome": image.model_dump(exclude_unset=True, by_alias=True)}
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

        current_meta.setdefault("attributes", {}).update(attrs)
        zarr_json.write_text(json.dumps(current_meta, indent=2))

    def _write_to_backend(
        self, array_key: str, index: tuple[int, ...], frame: np.ndarray
    ) -> None:
        """AcquireZarr-specific write implementation."""
        if self._stream is not None:
            self._stream.append(frame, key=array_key)

    def flush(self) -> None:
        if not self._stream:  # pragma: no cover
            raise RuntimeError("Stream is closed or uninitialized. Cannot flush.")
        # Flush the stream to ensure all data is written to disk.
        self._stream.close()
        self._stream = None
        gc.collect()
        # Only patch metadata for non-plate mode
        if self._plate is None:
            self._patch_group_metadata()

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
        array_key: str,
        dimensions: list[acquire_zarr.Dimension],
        dtype: np.dtype,
    ) -> acquire_zarr.ArraySettings:
        """Create an AcquireZarr ArraySettings for a position or well/field.

        Parameters
        ----------
        array_key : str
            The output key for the array. For non-plate mode, this is the position
            index as a string (e.g., "0", "1"). For plate mode, this is the
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
            output_key=array_key,
            dimensions=dimensions,
            data_type=dtype,
            downsampling_method=self._downsampling_method,
        )
