from __future__ import annotations

from typing import TYPE_CHECKING, Self

import numpy as np

from ._stream_base import OMEStream

if TYPE_CHECKING:
    from collections.abc import Sequence

    import acquire_zarr

    from ome_writers._dimensions import DimensionInfo


class AcquireZarrStream(OMEStream):
    def __init__(self) -> None:
        try:
            import acquire_zarr
        except ImportError as e:
            raise ImportError(
                "AcquireZarrStream requires the acquire-zarr package: "
                "pip install acquire-zarr"
            ) from e
        self._aqz = acquire_zarr
        super().__init__()
        self._stream: acquire_zarr.ZarrStream | None = None

    def create(
        self, path: str, dtype: np.dtype, dimensions: Sequence[DimensionInfo]
    ) -> Self:
        try:
            data_type = getattr(self._aqz.DataType, np.dtype(dtype).name.upper())
        except AttributeError as e:  # pragma: no cover
            raise ValueError(f"Cannot cast {dtype!r} to an acquire-zarr dtype.") from e
        settings = self._aqz.StreamSettings(
            store_path=self._normalize_path(path),
            data_type=data_type,
            version=self._aqz.ZarrVersion.V3,
        )
        settings.dimensions.extend(self._dim_toaqz_dim(x) for x in dimensions)
        self._stream = self._aqz.ZarrStream(settings)
        return self

    def append(self, frame: np.ndarray) -> None:
        self._stream = self._stream or None
        if self._stream is None:
            msg = "Stream is closed or uninitialized. Call create() first."
            raise RuntimeError(
                msg,
            )
        self._stream.append(frame)

    def flush(self) -> None:
        if self._stream is None:  # pragma: no cover
            raise RuntimeError("Stream is closed or uninitialized. Cannot flush.")
        # Flush the stream to ensure all data is written to disk.
        self._stream = None

    def is_active(self) -> bool:
        return self._stream is not None and self._stream.is_active()

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
