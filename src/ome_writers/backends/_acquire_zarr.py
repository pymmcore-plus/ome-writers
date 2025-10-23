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
        self._well_paths: dict[int, str] = {}  # position_idx -> well_path mapping

    def create(
        self,
        path: str,
        dtype: np.dtype,
        dimensions: Sequence[Dimension],
        *,
        overwrite: bool = False,
        plate: Plate | None = None,
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

        Returns
        -------
        Self
            The instance of the stream, allowing for chaining.
        """
        # Store plate metadata
        self._plate = plate

        # Use MultiPositionOMEStream to handle position logic
        num_positions, non_position_dims = self._init_positions(dimensions)
        self._group_path = Path(self._normalize_path(path))

        # Check if directory exists and handle overwrite parameter
        if self._group_path.exists():
            if not overwrite:
                raise FileExistsError(
                    f"Directory {self._group_path} already exists. "
                    "Use overwrite=True to overwrite it."
                )
            shutil.rmtree(self._group_path)

        # Validate plate metadata if provided
        if plate is not None:
            # Calculate expected positions: wells * fields_per_well
            fields_per_well = plate.field_count if plate.field_count is not None else 1
            expected_positions = len(plate.wells) * fields_per_well
            if num_positions != expected_positions:
                raise ValueError(
                    f"Number of positions ({num_positions}) does not match "
                    f"expected positions ({expected_positions}) for plate with "
                    f"{len(plate.wells)} wells and {fields_per_well} field(s) per well"
                )

        # Dimensions will be the same across all positions, so we can create them once
        az_dims = [self._dim_toaqz_dim(dim) for dim in non_position_dims]
        # keep a strong reference (avoid segfaults)
        self._az_dims_keepalive = az_dims

        # Create AcquireZarr array settings for each position
        az_array_settings = []
        for pos_idx in range(num_positions):
            if plate is not None:
                # For plate mode, use well path as the array key
                well_pos = plate.wells[pos_idx]
                array_key = f"{well_pos.path}/0"  # /0 is the field index
                self._well_paths[pos_idx] = well_pos.path
            else:
                # For non-plate mode, use position index as key
                array_key = str(pos_idx)

            az_array_settings.append(self._aqz_pos_array(array_key, az_dims, dtype))

        for arr in az_array_settings:
            for d in arr.dimensions:
                assert d.chunk_size_px > 0, (d.name, d.chunk_size_px)
                assert d.shard_size_chunks > 0, (d.name, d.shard_size_chunks)

        self._az_arrays_keepalive = az_array_settings

        # Create streams for each position
        settings = self._aqz.StreamSettings(
            arrays=az_array_settings,
            store_path=str(self._group_path),
            version=self._aqz.ZarrVersion.V3,
        )
        self._az_settings_keepalive = settings
        self._stream = self._aqz.ZarrStream(settings)

        # If plate mode, update the indices mapping to use well paths
        if plate is not None:
            # Rebuild the indices mapping with well paths instead of position indices
            from itertools import product

            self._indices = {}
            non_p_ranges = [
                range(d.size) for d in non_position_dims if d.label not in "yx"
            ]
            idx = 0
            for pos_idx in range(num_positions):
                well_pos = plate.wells[pos_idx]
                array_key = f"{well_pos.path}/0"  # /0 is the field index
                for non_p_idx in (tuple(v) for v in product(*non_p_ranges)):
                    self._indices[idx] = (array_key, non_p_idx)
                    idx += 1

        self._patch_group_metadata()
        return self

    def _patch_group_metadata(self) -> None:
        """Patch the group metadata with OME NGFF 0.5 metadata.

        This method exists because there are cases in which the standard acquire-zarr
        API is not flexible enough to handle all the cases we need (such as multiple
        positions or plate layouts).  This method manually writes the OME NGFF v0.5
        metadata with our manually constructed metadata.
        """
        if self._plate is not None:
            # Write plate metadata
            self._write_plate_metadata()
        else:
            # Write standard multi-position metadata
            dims = self._non_position_dims
            from ome_writers._dimensions import dims_to_ngff_v5

            attrs = dims_to_ngff_v5({str(i): dims for i in range(self._num_positions)})
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

    def _write_plate_metadata(self) -> None:
        """Write plate metadata to top-level and well metadata.

        This creates the proper NGFF v0.5 plate structure with plate metadata
        at the top level and well metadata in each well group.
        """
        if self._plate is None:
            return

        # Convert plate to yaozarrs format
        try:
            from ome_writers._plate import plate_to_yaozarrs_v5

            yao_plate = plate_to_yaozarrs_v5(self._plate)
        except ImportError as e:
            raise ImportError(
                "yaozarrs is required for plate support. "
                "Please install it via `pip install yaozarrs[io]`"
            ) from e

        # Write top-level plate metadata
        zarr_json = Path(self._group_path) / "zarr.json"
        plate_meta = {
            "zarr_format": 3,
            "node_type": "group",
            "attributes": yao_plate.model_dump(exclude_unset=True, by_alias=True),
        }
        zarr_json.write_text(json.dumps(plate_meta, indent=2))

        # Write well metadata for each well
        dims = self._non_position_dims
        for well_pos in self._plate.wells:
            well_path = Path(self._group_path) / well_pos.path
            well_zarr_json = well_path / "zarr.json"

            # Create well metadata with field images
            # For now, assume 1 field per well (field "0")
            from yaozarrs import v05

            well_def = v05.WellDef(images=[v05.FieldOfView(path="0", acquisition=None)])
            well_obj = v05.Well(version="0.5", well=well_def)

            well_meta = {
                "zarr_format": 3,
                "node_type": "group",
                "attributes": well_obj.model_dump(exclude_unset=True, by_alias=True),
            }
            well_zarr_json.parent.mkdir(parents=True, exist_ok=True)
            well_zarr_json.write_text(json.dumps(well_meta, indent=2))

            # Write image metadata for the field (field "0")
            field_path = well_path / "0"
            field_zarr_json = field_path / "zarr.json"

            # Create image metadata with multiscales
            from ome_writers._dimensions import dims_to_ngff_v5

            image_attrs = dims_to_ngff_v5({"0": dims})

            field_meta = {
                "zarr_format": 3,
                "node_type": "group",
                "attributes": image_attrs,
            }
            field_zarr_json.parent.mkdir(parents=True, exist_ok=True)
            field_zarr_json.write_text(json.dumps(field_meta, indent=2))

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
        )
