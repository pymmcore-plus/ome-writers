"""Yaozarrs-based backends for OME-Zarr v0.5."""

from __future__ import annotations

import gc
from typing import TYPE_CHECKING, Any, Literal

import acquire_zarr as az

from ._yaozarrs import YaozarrsBackend

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence
    from pathlib import Path

    import numpy as np

    from ome_writers._router import PositionInfo
    from ome_writers.schema import AcquisitionSettings, Dimension


class AcquireZarrBackend(YaozarrsBackend):
    """OME-Zarr backend using yaozarrs for metadata and acquire-zarr for writes.

    Combines:
    - yaozarrs: Creates OME-NGFF v0.5 compliant group structure and metadata
    - acquire-zarr: High-performance sequential writes for array data

    Requirements:
    - storage_order="acquisition" (sequential writes only)
    - Root path must end with .zarr
    """

    def __init__(self) -> None:
        super().__init__()
        self._stream: az.ZarrStream | None = None
        self._az_pos_keys: list[str] = []
        # hack to deal with the fact that acquire-zarr overwrites zarr.json files
        # with empty group metadata, even when using output_key="..."
        self._zarr_json_backup: dict[Path, bytes] = {}

    def is_incompatible(self, settings: AcquisitionSettings) -> Literal[False] | str:
        if not settings.root_path.endswith(".zarr"):
            return "Root path must end with .zarr for AcquireZarrBackend."
        if settings.storage_index_permutation is not None:
            return (
                "AcquireZarrBackend does not support permuted storage order. "
                "Data must be written in acquisition order."
            )
        if len(settings.array_storage_dimensions) < 3:
            return (
                "AcquireZarrBackend requires at least 3 dimensions. "
                "2D images are not currently supported."
            )
        return False

    def _get_writer(self) -> Callable[..., _ArrayPlaceholder]:
        """Return custom writer that collects array configs as placeholders."""

        def custom_writer(
            path: Path,
            shape: tuple[int, ...],
            dtype: Any,
            chunks: tuple[int, ...],
            *,
            shards: tuple[int, ...] | None,
            overwrite: bool,
            compression: Any,
            dimension_names: list[str] | None,
        ) -> _ArrayPlaceholder:
            output_key = str(path.relative_to(self._root))
            return _ArrayPlaceholder(output_key, shape)

        return custom_writer

    def _post_prepare(self, settings: AcquisitionSettings) -> None:
        """Create acquire-zarr stream after yaozarrs creates metadata."""
        assert self._root is not None
        # Build position -> output_key mapping from placeholder arrays
        self._az_pos_keys = [arr.output_key for arr in self._arrays]

        # Backup zarr.json files created by yaozarrs before acquire-zarr
        # potentially overwrites them (will be restored in finalize)
        for zarr_json in self._root.rglob("zarr.json"):
            self._zarr_json_backup[zarr_json] = zarr_json.read_bytes()

        # Create acquire-zarr stream
        ndims = len(settings.array_storage_dimensions)
        az_dims = [
            _to_acquire_dim(dim, frame_dim=(i >= ndims - 2))
            for i, dim in enumerate(settings.array_storage_dimensions)
        ]

        self._stream = az.ZarrStream(
            az.StreamSettings(
                arrays=[
                    az.ArraySettings(
                        output_key=key,
                        dimensions=az_dims,
                        data_type=settings.dtype,
                    )
                    for key in self._az_pos_keys
                ],
                store_path=str(self._root),
                version=az.ZarrVersion.V3,
            )
        )

    def write(
        self,
        position_info: PositionInfo,
        index: tuple[int, ...],
        frame: np.ndarray,
    ) -> None:
        """Write frame sequentially via acquire-zarr stream."""
        if self._stream is None:
            raise RuntimeError("Backend not prepared.")

        output_key = self._az_pos_keys[position_info[0]]
        self._stream.append(frame, key=output_key)

    def finalize(self) -> None:
        """Close stream and release resources."""
        if not self._finalized and self._stream is not None:
            self._stream.close()
            self._stream = None
            gc.collect()

            # Restore yaozarrs metadata that acquire-zarr may have overwritten
            for path, content in self._zarr_json_backup.items():
                path.write_bytes(content)
            self._zarr_json_backup.clear()

        super().finalize()


# -----------------------------------------------------------------------------


def _to_acquire_dim(dim: Dimension, frame_dim: bool) -> az.Dimension:
    """Convert a Dimension to az.Dimension."""

    # Map dimension type to az DimensionType
    dim_type_map = {
        "time": az.DimensionType.TIME,
        "channel": az.DimensionType.CHANNEL,
        "space": az.DimensionType.SPACE,
    }

    if dim.chunk_size is not None:
        chunk_size = dim.chunk_size
    else:
        chunk_size = dim.count if frame_dim else 1

    kind = dim_type_map.get(dim.type or "", az.DimensionType.OTHER)
    return az.Dimension(
        name=dim.name,
        kind=kind,
        array_size_px=dim.count or 1,
        chunk_size_px=chunk_size,
        shard_size_chunks=dim.shard_size or 1,
    )


class _ArrayPlaceholder:
    """Placeholder returned by custom writer - writes are routed through ZarrStream."""

    def __init__(self, output_key: str, shape: tuple[int, ...]) -> None:
        self.output_key = output_key
        self.shape = list(shape)

    def resize(self, new_shape: Sequence[int]) -> None:
        self.shape = list(new_shape)
        # no updates needed to acquire-zarr stream, it already supports appending
        # to the first dimension

    def __setitem__(self, index: Any, value: Any) -> None:
        pass  # No-op - actual writes go through ZarrStream
