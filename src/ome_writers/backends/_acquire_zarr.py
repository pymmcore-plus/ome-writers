from __future__ import annotations

import importlib
import importlib.util
import json
from contextlib import suppress
from pathlib import Path
from typing import TYPE_CHECKING, Self

import numpy as np

from .._ngff_metadata import ome_meta_v5
from .._stream_base import MultiPositionOMEStream

if TYPE_CHECKING:
    from collections.abc import Sequence

    import acquire_zarr

    from ome_writers._dimensions import DimensionInfo


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
        self._streams: dict[str, acquire_zarr.ZarrStream] = {}  # array_key -> stream

    def create(
        self, path: str, dtype: np.dtype, dimensions: Sequence[DimensionInfo]
    ) -> Self:
        # Use MultiPositionOMEStream to handle position logic
        num_positions, non_position_dims = self._init_positions(dimensions)
        self._group_path = Path(self._normalize_path(path))

        try:
            data_type = getattr(self._aqz.DataType, np.dtype(dtype).name.upper())
        except AttributeError as e:  # pragma: no cover
            raise ValueError(f"Cannot cast {dtype!r} to an acquire-zarr dtype.") from e

        az_dims = [self._dim_toaqz_dim(dim) for dim in non_position_dims]
        # Create streams for each position
        for pos_idx in range(num_positions):
            array_key = str(pos_idx)
            settings = self._aqz.StreamSettings(
                store_path=str(self._group_path),
                output_key=array_key,
                data_type=data_type,
                version=self._aqz.ZarrVersion.V3,
            )
            settings.dimensions.extend(az_dims)
            self._streams[array_key] = self._aqz.ZarrStream(settings)

        self._patch_group_metadata()
        return self

    def _patch_group_metadata(self) -> None:
        """Patch the group metadata with OME NGFF v0.5 metadata.

        This method exists because there are cases in which the standard acquire-zarr
        API is not flexible enough to handle all the cases we need (such as multiple
        positions).  This method manually writes the OME NGFF v0.5 metadata with our
        manually constructed metadata.
        """
        dims = self._non_position_dims
        attrs = ome_meta_v5({str(i): dims for i in range(self._num_positions)})
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
        stream = self._streams[array_key]
        stream.append(frame)

    def flush(self) -> None:
        if not self._streams:  # pragma: no cover
            raise RuntimeError("Stream is closed or uninitialized. Cannot flush.")
        # Flush all streams to ensure all data is written to disk.
        self._streams.clear()
        self._patch_group_metadata()

    def is_active(self) -> bool:
        return bool(self._streams) and any(
            stream.is_active() for stream in self._streams.values()
        )

    def _dim_toaqz_dim(
        self,
        dim: DimensionInfo,
        shard_size_chunks: int = 1,
    ) -> acquire_zarr.Dimension:
        return self._aqz.Dimension(
            name=dim.label,
            type=getattr(self._aqz.DimensionType, dim.ome_dim_type.upper()),
            array_size_px=dim.size,
            chunk_size_px=(dim.chunk_size if dim.chunk_size is not None else dim.size),
            shard_size_chunks=shard_size_chunks,
        )
