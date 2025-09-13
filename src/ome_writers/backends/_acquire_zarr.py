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

from ome_writers._ngff_metadata import ome_meta_v5
from ome_writers._stream_base import MultiPositionOMEStream

if TYPE_CHECKING:
    from collections.abc import Sequence

    import acquire_zarr
    import numpy as np

    from ome_writers._dimensions import Dimension


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
        *,
        overwrite: bool = False,
    ) -> Self:
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

        # Dimensions will be the same across all positions, so we can create them once
        az_dims = [self._dim_toaqz_dim(dim) for dim in non_position_dims]
        # keep a strong reference (avoid segfaults)
        self._az_dims_keepalive = az_dims

        # Create AcquireZarr array settings for each position
        az_array_settings = [
            self._aqz_pos_array(pos_idx, az_dims, dtype)
            for pos_idx in range(num_positions)
        ]

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

        self._patch_group_metadata()
        return self

    def _patch_group_metadata(self) -> None:
        """Patch the group metadata with OME NGFF 0.5 metadata.

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
        position_index: int,
        dimensions: list[acquire_zarr.Dimension],
        dtype: np.dtype,
    ) -> acquire_zarr.ArraySettings:
        """Create an AcquireZarr ArraySettings for a position."""
        return self._aqz.ArraySettings(
            output_key=str(
                position_index
            ),  # this matches the position index key from the base class
            dimensions=dimensions,
            data_type=dtype,
        )
