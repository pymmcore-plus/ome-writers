import importlib.util
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Self

import numpy as np

from ._dimensions import DimensionInfo
from ._stream_base import MultiPositionOMEStream


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
        self, path: str, dtype: np.dtype, dimensions: Sequence[DimensionInfo]
    ) -> Self:
        # Use MultiPositionOMEStream to handle position logic
        num_positions, non_position_dims = self._init_positions(dimensions)

        self._create_group(self._normalize_path(path), dimensions)

        # Create stores for each array
        for pos_idx in range(num_positions):
            array_key = str(pos_idx)
            spec = self._create_spec(dtype, non_position_dims, array_key)
            self._stores[array_key] = self._ts.open(spec).result()

        return self

    def _create_spec(
        self, dtype: np.dtype, dimensions: Sequence[DimensionInfo], array_key: str
    ) -> dict:
        labels, shape, units, chunk_shape = zip(*dimensions)
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
        future = store[index].write(frame)
        self._futures.append(future)

    def flush(self) -> None:
        # Wait for all writes to finish.
        for future in self._futures:
            future.result()
        self._futures.clear()
        self._stores.clear()

    def is_active(self) -> bool:
        return bool(self._stores)

    def _create_group(self, path: str, dims: Sequence[DimensionInfo]) -> Path:
        self._group_path = Path(path)
        self._group_path.mkdir(parents=True, exist_ok=True)

        # Determine array keys and dimensions based on position dimension
        position_dims = [d for d in dims if d.label == "p"]
        non_position_dims = [d for d in dims if d.label != "p"]

        # Determine number of positions (1 if no position dimension)
        num_positions = position_dims[0].size if position_dims else 1

        array_dims: dict[str, Sequence[DimensionInfo]] = {}
        for pos_idx in range(num_positions):
            array_key = str(pos_idx)
            self._array_paths[array_key] = self._group_path / array_key
            # Use non_position_dims for multi-pos, full dims for single pos
            array_dims[array_key] = non_position_dims if self._position_dim else dims

        group_zarr = self._group_path / "zarr.json"
        group_zarr.write_text(json.dumps(self._group_meta(array_dims), indent=2))
        return self._group_path

    def _group_meta(self, array_dims: Mapping[str, Sequence[DimensionInfo]]) -> dict:
        multiscales = []
        for array_path, dims in array_dims.items():
            axes, scales = _ome_axes_scales(dims)
            ct = {"scale": scales, "type": "scale"}
            ds = {"path": array_path, "coordinateTransformations": [ct]}
            multiscales.append({"axes": axes, "datasets": [ds]})
        attrs = {"ome": {"version": "0.5", "multiscales": multiscales}}
        return {"zarr_format": 3, "node_type": "group", "attributes": attrs}


def _ome_axes_scales(dims: Sequence[DimensionInfo]) -> tuple[list[dict], list[float]]:
    """Return ome axes meta.

    The length of "axes" must be between 2 and 5 and MUST be equal to the
    dimensionality of the zarr arrays storing the image data. The "axes" MUST
    contain 2 or 3 entries of "type:space" and MAY contain one additional
    entry of "type:time" and MAY contain one additional entry of
    "type:channel" or a null / custom type. The order of the entries MUST
    correspond to the order of dimensions of the zarr arrays. In addition, the
    entries MUST be ordered by "type" where the "time" axis must come first
    (if present), followed by the "channel" or custom axis (if present) and
    the axes of type "space".
    """
    axes: list[dict] = []
    scales: list[float] = []
    for dim in dims:
        axes.append(
            {"name": dim.label, "type": dim.ome_dim_type, "unit": dim.ome_unit},
        )
        scales.append(dim.ome_scale)
    return axes, scales
