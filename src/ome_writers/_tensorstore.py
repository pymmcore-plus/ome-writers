import json
from collections.abc import Mapping, Sequence
from itertools import product
from pathlib import Path
from typing import Self

import numpy as np

from ome_writers._stream_base import OMEStream
from ome_writers.dimensions import DimensionInfo


class TensorStoreZarrStream(OMEStream):
    def __init__(self) -> None:
        try:
            import tensorstore
        except ImportError as e:
            msg = (
                "TensorStoreZarrStream requires tensorstore: `pip install tensorstore`."
            )
            raise ImportError(
                msg,
            ) from e

        self._ts = tensorstore
        super().__init__()
        self._count = 0
        self._group_path: Path | None = None
        self._array_path: Path | None = None
        self._futures: tensorstore.FutureLike = []
        self._store: tensorstore.TensorStore | None = None
        self._indices: dict[int, tuple[int, ...]] = {}
        self._delete_existing = True

    def create(
        self, path: str, dtype: np.dtype, dimensions: Sequence[DimensionInfo]
    ) -> Self:
        # Define a TensorStore spec for a Zarr v3 store.
        prod = product(*[range(d.size) for d in dimensions if d.label not in "yx"])
        self._indices = dict(enumerate(prod))

        self._create_group(self.normalize_path(path), dimensions)
        spec = self._create_spec(dtype, dimensions)
        self._store = self._ts.open(spec).result()
        return self

    def _create_spec(
        self, dtype: np.dtype, dimensions: Sequence[DimensionInfo]
    ) -> dict:
        labels, shape, units, chunk_shape = zip(*dimensions)
        labels = tuple(str(x) for x in labels)
        return {
            "driver": "zarr3",
            "kvstore": {"driver": "file", "path": str(self._array_path)},
            "schema": {
                "domain": {"shape": shape, "labels": labels},
                "dtype": dtype.name,
                "chunk_layout": {"chunk": {"shape": chunk_shape}},
                "dimension_units": units,
            },
            "create": True,
            "delete_existing": self._delete_existing,
        }

    def append(self, frame: np.ndarray) -> None:
        if self._store is None:
            msg = "Stream is closed or uninitialized. Call create() first."
            raise RuntimeError(msg)
        index = self._indices[self._count]
        future = self._store[index].write(frame)
        self._futures.append(future)
        self._count += 1

    def flush(self) -> None:
        # Wait for all writes to finish.
        for future in self._futures:
            future.result()
        self._futures.clear()
        self._store = None

    def is_active(self) -> bool:
        return self._store is not None

    def _create_group(self, path: str, dims: Sequence[DimensionInfo]) -> Path:
        self._group_path = Path(path)
        array_key = "0"
        self._array_path = self._group_path / array_key
        self._group_path.mkdir(parents=True, exist_ok=True)
        group_zarr = self._group_path / "zarr.json"
        group_zarr.write_text(json.dumps(self._group_meta({array_key: dims}), indent=2))
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
