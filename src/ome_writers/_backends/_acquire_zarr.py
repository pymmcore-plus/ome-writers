"""Yaozarrs-based backends for OME-Zarr v0.5."""

from __future__ import annotations

import gc
from typing import TYPE_CHECKING, Any, Literal, NamedTuple

import acquire_zarr as az

from ome_writers._backends._yaozarrs import YaozarrsBackend
from ome_writers._schema import Dimension

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path

    import numpy as np

    from ome_writers._schema import AcquisitionSettings, Dimension


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
        # see https://github.com/acquire-project/acquire-zarr/issues/186
        self._zarr_json_backup: dict[Path, bytes] = {}

    def is_incompatible(self, settings: AcquisitionSettings) -> Literal[False] | str:
        if not settings.root_path.endswith(".zarr"):  # pragma: no cover
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
        try:
            self._resolve_compression(settings.compression)
        except ValueError as e:
            return str(e)
        return False

    def _get_yaozarrs_writer(self) -> Callable[..., _ArrayPlaceholder]:
        """Return custom writer that collects array configs as placeholders."""

        def custom_writer(
            path: Path, shape: tuple[int, ...], *_: Any, **__: Any
        ) -> _ArrayPlaceholder:
            # as_posix is critical on windows to ensure forward slashes
            # which is what acquire-zarr expects
            output_key = path.relative_to(self._root).as_posix()
            return _ArrayPlaceholder(output_key, shape)

        return custom_writer

    def _post_prepare(self, settings: AcquisitionSettings) -> None:
        """Create acquire-zarr stream after yaozarrs creates metadata."""
        assert self._root is not None
        # Build position -> output_key mapping from placeholder arrays
        self._az_pos_keys = [arr.output_key for arr in self._arrays]

        # Backup zarr.json files created by yaozarrs before acquire-zarr
        # potentially overwrites them (will be restored in finalize)
        # https://github.com/acquire-project/acquire-zarr/issues/186
        for zarr_json in self._root.rglob("zarr.json"):
            self._zarr_json_backup[zarr_json] = zarr_json.read_bytes()

        # Create acquire-zarr stream
        ndims = len(settings.array_storage_dimensions)
        az_dims = [
            _to_acquire_dim(dim, frame_dim=(i >= ndims - 2))
            for i, dim in enumerate(settings.array_storage_dimensions)
        ]

        compression_settings = self._resolve_compression(settings.compression)
        self._stream = az.ZarrStream(
            az.StreamSettings(
                arrays=[
                    az.ArraySettings(
                        output_key=key,
                        dimensions=az_dims,
                        data_type=settings.dtype,
                        compression=compression_settings,
                    )
                    for key in self._az_pos_keys
                ],
                store_path=str(self._root),
                version=az.ZarrVersion.V3,
            )
        )

    def write(
        self,
        position_index: int,
        index: tuple[int, ...],
        frame: np.ndarray,
    ) -> None:
        """Write frame sequentially via acquire-zarr stream."""
        if self._stream is None:  # pragma: no cover
            raise RuntimeError("Backend not prepared.")

        output_key = self._az_pos_keys[position_index]
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

    def _resolve_compression(
        self, compression: str | None
    ) -> az.CompressionSettings | None:
        """Resolve compression setting for acquire-zarr backend.

        Returns
        -------
        az.CompressionSettings | None
            acquire-zarr compression settings
        """
        if compression in ("none", None):
            # https://github.com/acquire-project/acquire-zarr/pull/187
            # this seems like it should work... but it causes a ton of errors:
            # return az.CompressionSettings(
            #     compressor=az.Compressor.NONE,
            #     codec=az.CompressionCodec.NONE,
            # )
            return None

        if compression == "blosc-zstd":
            return az.CompressionSettings(
                compressor=az.Compressor.BLOSC1,
                codec=az.CompressionCodec.BLOSC_ZSTD,
                level=3,
                shuffle=1,
            )
        if compression == "blosc-lz4":
            return az.CompressionSettings(
                compressor=az.Compressor.BLOSC1,
                codec=az.CompressionCodec.BLOSC_LZ4,
                level=5,
                shuffle=1,
            )
        # Unsupported compressions
        if compression == "zstd":
            raise ValueError(
                "Standalone 'zstd' compression (without blosc) is not supported by "
                "acquire-zarr. Use 'blosc-zstd' instead."
            )
        if compression == "lzw":
            raise ValueError(
                "LZW compression is not supported for Zarr format. "
                "Use 'blosc-zstd' or 'blosc-lz4' instead."
            )
        raise ValueError(f"Unsupported compression for acquire-zarr: {compression}")


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
        shard_size_chunks=dim.shard_size_chunks or 1,
    )


class _ArrayPlaceholder(NamedTuple):
    """Placeholder returned by custom writer - writes are routed through ZarrStream."""

    output_key: str
    shape: tuple[int, ...]
