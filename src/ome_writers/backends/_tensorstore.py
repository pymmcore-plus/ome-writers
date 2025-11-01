from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from typing import TYPE_CHECKING

from typing_extensions import Self

from ome_writers._dimensions import dims_to_yaozarrs_v5
from ome_writers._stream_base import MultiPositionOMEStream

if TYPE_CHECKING:
    from collections.abc import Sequence

    import numpy as np

    from ome_writers._dimensions import Dimension
    from ome_writers._plate import Plate


class TensorStoreZarrStream(MultiPositionOMEStream):
    @classmethod
    def is_available(cls) -> bool:  # pragma: no cover
        """Check if the tensorstore package is available."""
        return importlib.util.find_spec("tensorstore") is not None

    def __init__(self) -> None:
        try:
            import tensorstore
        except ImportError as e:
            msg = (
                "TensorStoreZarrStream requires tensorstore: `pip install tensorstore`."
            )
            raise ImportError(msg) from e

        self._ts = tensorstore
        super().__init__()
        self._group_path: Path | None = None
        self._array_paths: dict[str, Path] = {}  # array_key -> path mapping
        self._futures: list = []
        self._stores: dict[str, tensorstore.TensorStore] = {}  # array_key -> store
        self._delete_existing = True
        self._plate: Plate | None = None
        self._plate_position_keys: dict[int, str] = {}

    def create(
        self,
        path: str,
        dtype: np.dtype,
        dimensions: Sequence[Dimension],
        plate: Plate | None = None,
        *,
        overwrite: bool = False,
    ) -> Self:
        # Store plate information
        self._plate = plate
        self._plate_position_keys.clear()

        # Initialize dimensions from MultiPositionOMEStream
        self._init_dimensions(dimensions)

        self._delete_existing = overwrite

        # Build position keys for HCS if plate is provided
        if plate is not None:
            plate_path = (plate.name or "plate").replace(" ", "_")
            fields_per_well = plate.field_count or 1
            pos_idx = 0
            for well in plate.wells:
                for field_idx in range(fields_per_well):
                    fov_key = f"fov{field_idx}"
                    key = f"{plate_path}/{well.path}/{fov_key}"
                    self._plate_position_keys[pos_idx] = key
                    pos_idx += 1

        # Pass self._non_position_dims (storage order) to _create_group so metadata
        # matches the actual array dimension order (TCZYX)
        self._create_group(self._normalize_path(path), self.storage_order_dims)

        # Create stores for each array
        for pos_idx in range(self.num_positions):
            if self._plate is not None:
                array_key = self._plate_position_keys[pos_idx]
            else:
                array_key = str(pos_idx)
            spec = self._create_spec(dtype, self.storage_order_dims, array_key)
            try:
                self._stores[array_key] = self._ts.open(spec).result()
            except ValueError as e:
                if "ALREADY_EXISTS" in str(e):
                    raise FileExistsError(
                        f"Array {array_key} already exists at "
                        f"{self._array_paths[array_key]}. "
                        "Use overwrite=True to overwrite it."
                    ) from e
                else:
                    raise
        return self

    def _create_spec(
        self, dtype: np.dtype, dimensions: Sequence[Dimension], array_key: str
    ) -> dict:
        labels, shape, units, chunk_shape = zip(*dimensions, strict=False)
        labels = tuple(str(x) for x in labels)
        return {
            "driver": "zarr3",
            "kvstore": {"driver": "file", "path": str(self._array_paths[array_key])},
            "schema": {
                "domain": {"shape": shape, "labels": labels},
                "dtype": dtype.name,
                "chunk_layout": {"chunk": {"shape": chunk_shape}},
                "dimension_units": units,
            },
            "create": True,
            "delete_existing": self._delete_existing,
        }

    def _write_to_backend(
        self, position_key: str, index: tuple[int, ...], frame: np.ndarray
    ) -> None:
        """TensorStore-specific write implementation."""
        if self._plate is not None and self._plate_position_keys:
            actual_key = self._plate_position_keys.get(int(position_key), position_key)
        else:
            actual_key = position_key
        store = self._stores[actual_key]
        # Named tuples work directly as indices
        future = store[index].write(frame)  # type: ignore[index]
        self._futures.append(future)

    def flush(self) -> None:
        # Wait for all writes to finish.
        for future in self._futures:
            future.result()
        self._futures.clear()
        self._stores.clear()

    def is_active(self) -> bool:
        return bool(self._stores)

    def _create_group(self, path: str, dims: Sequence[Dimension]) -> Path:
        self._group_path = Path(path)
        self._group_path.mkdir(parents=True, exist_ok=True)

        array_dims: dict[str, Sequence[Dimension]] = {}
        if self._plate is not None:
            # For HCS plates, use the position keys as array keys
            for array_key in self._plate_position_keys.values():
                self._array_paths[array_key] = self._group_path / array_key
                # Ensure parent directories exist
                self._array_paths[array_key].parent.mkdir(parents=True, exist_ok=True)
                # Use dims (non-position dimensions in storage order)
                array_dims[array_key] = dims
        else:
            # Standard multi-position mode
            for pos_idx in range(self.num_positions):
                array_key = str(pos_idx)
                self._array_paths[array_key] = self._group_path / array_key
                # Use dims (non-position dimensions in storage order)
                array_dims[array_key] = dims

        group_zarr = self._group_path / "zarr.json"
        group_meta = {
            "zarr_format": 3,
            "node_type": "group",
            "attributes": {
                "ome": dims_to_yaozarrs_v5(array_dims=array_dims).model_dump()
            },
        }
        group_zarr.write_text(json.dumps(group_meta, indent=2))

        # For HCS plates, create plate-level metadata
        if self._plate is not None:
            from ome_writers._plate import plate_to_yaozarrs_v5

            plate_name_sanitized = (self._plate.name or "plate").replace(" ", "_")
            plate_path = self._group_path / plate_name_sanitized
            plate_zarr = plate_path / "zarr.json"
            plate_meta = {
                "zarr_format": 3,
                "node_type": "group",
                "attributes": {"ome": plate_to_yaozarrs_v5(self._plate).model_dump()},
            }
            plate_zarr.write_text(json.dumps(plate_meta, indent=2))

            # Create intermediate group metadata for row and well directories
            # This is required for Zarr v3 - every group in the hierarchy needs metadata
            self._create_intermediate_groups()

        return self._group_path

    def _create_intermediate_groups(self) -> None:
        """Create zarr.json metadata for intermediate HCS groups (rows and wells).

        In Zarr v3, every directory in the hierarchy must be a recognized zarr node.
        This method creates minimal group metadata for row-level (e.g., 'A/', 'B/')
        and well-level (e.g., '01/', '02/') directories.
        """
        if self._plate is None or self._group_path is None:
            return

        # Get all unique parent directories from array paths
        intermediate_dirs: set[Path] = set()
        for array_path in self._array_paths.values():
            # Get all parents between the group path and the array
            current = array_path.parent
            while current != self._group_path and current.is_relative_to(
                self._group_path
            ):
                intermediate_dirs.add(current)
                current = current.parent

        # Create minimal group metadata for each intermediate directory
        minimal_group_meta = {
            "zarr_format": 3,
            "node_type": "group",
        }

        for dir_path in intermediate_dirs:
            zarr_json = dir_path / "zarr.json"
            if not zarr_json.exists():
                zarr_json.write_text(json.dumps(minimal_group_meta, indent=2))
