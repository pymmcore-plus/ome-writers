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

    def create(
        self,
        path: str,
        dtype: np.dtype,
        dimensions: Sequence[Dimension],
        *,
        overwrite: bool = False,
    ) -> Self:
        # Use MultiPositionOMEStream to handle position logic
        num_positions, non_position_dims = self._init_positions(dimensions)
        self._delete_existing = overwrite

        self._create_group(self._normalize_path(path), dimensions)

        # Create stores for each array
        for pos_idx in range(num_positions):
            array_key = str(pos_idx)
            spec = self._create_spec(dtype, non_position_dims, array_key)
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
        self, array_key: str, index: tuple[int, ...], frame: np.ndarray
    ) -> None:
        """TensorStore-specific write implementation."""
        store = self._stores[array_key]
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

        # Determine array keys and dimensions based on position dimension
        position_dims = [d for d in dims if d.label == "p"]
        non_position_dims = [d for d in dims if d.label != "p"]

        # Determine number of positions (1 if no position dimension)
        num_positions = position_dims[0].size if position_dims else 1

        array_dims: dict[str, Sequence[Dimension]] = {}
        for pos_idx in range(num_positions):
            array_key = str(pos_idx)
            self._array_paths[array_key] = self._group_path / array_key
            # Use non_position_dims for multi-pos, full dims for single pos
            array_dims[array_key] = non_position_dims if self._position_dim else dims

        group_zarr = self._group_path / "zarr.json"
        group_meta = {
            "zarr_format": 3,
            "node_type": "group",
            "attributes": ome_meta_v5(array_dims=array_dims),
        }
        group_zarr.write_text(json.dumps(group_meta, indent=2))
        return self._group_path
