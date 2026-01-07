from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from typing import TYPE_CHECKING

from typing_extensions import Self

from ome_writers._ngff_metadata import ome_meta_v5
from ome_writers._stream_base import MultiPositionOMEStream

if TYPE_CHECKING:
    from collections.abc import Sequence

    import numpy as np

    from ome_writers._dimensions import Dimension


class OldTensorStoreZarrStream(MultiPositionOMEStream):
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

    def create(
        self,
        path: str,
        dtype: np.dtype,
        dimensions: Sequence[Dimension],
        *,
        overwrite: bool = False,
    ) -> Self:
        # Initialize dimensions from MultiPositionOMEStream
        # NOTE: Data will be stored in acquisition order.
        self._configure_dimensions(dimensions)
        self._delete_existing = overwrite

        # Create group and array paths
        self._create_group(self._normalize_path(path), self.storage_dims)

        # Create stores for each array
        for pos_idx in range(self.num_positions):
            array_key = str(pos_idx)
            spec = self._create_spec(dtype, self.storage_dims, array_key)
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
        store = self._stores[position_key]
        # Named tuples work directly as indices
        future = store[index].write(frame)  # type: ignore[index]
        self._futures.append(future)

    def flush(self) -> None:
        # Wait for all writes to finish.
        for future in self._futures:
            future.result()
        self._futures.clear()
        self._stores.clear()
        # Patch metadata to ensure OME-NGFF v0.5 compliance
        self._patch_metadata_to_ngff_v05()

    def is_active(self) -> bool:
        return bool(self._stores)

    def _create_group(self, path: str, dims: Sequence[Dimension]) -> Path:
        self._group_path = Path(path)
        self._group_path.mkdir(parents=True, exist_ok=True)

        # Create array paths for each position
        for pos_idx in range(self.num_positions):
            array_key = str(pos_idx)
            self._array_paths[array_key] = self._group_path / array_key

        # Note: We'll create proper OME-NGFF metadata in _patch_metadata_to_ngff_v05
        # after data writing is complete
        group_zarr = self._group_path / "zarr.json"
        group_meta = {
            "zarr_format": 3,
            "node_type": "group",
            "attributes": {},
        }
        group_zarr.write_text(json.dumps(group_meta, indent=2))
        return self._group_path

    def _patch_metadata_to_ngff_v05(self) -> None:
        """Patch the Zarr group metadata to ensure OME-NGFF v0.5 compliance.

        This method writes OME-NGFF v0.5-compliant metadata for the group,
        preserving the acquisition order (storage order) for the dimensions.
        Data is stored and read back in the same order it was acquired.
        """
        if self._group_path is None:
            return

        # Use storage order dims as-is (acquisition order)
        attrs = ome_meta_v5(
            {str(i): self.storage_dims for i in range(self.num_positions)}
        )
        zarr_json = self._group_path / "zarr.json"
        current_meta: dict = {
            "consolidated_metadata": None,
            "node_type": "group",
            "zarr_format": 3,
        }
        if zarr_json.exists():
            with open(zarr_json) as f:
                current_meta = json.load(f)

        current_meta.setdefault("attributes", {}).update(attrs)
        zarr_json.write_text(json.dumps(current_meta, indent=2))
