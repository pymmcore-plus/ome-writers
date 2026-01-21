"""Yaozarrs-based backends for OME-Zarr v0.5."""

from __future__ import annotations

import json
import warnings
from abc import abstractmethod
from copy import deepcopy
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, cast

from ome_writers._backend import ArrayBackend

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    import numpy as np

    from ome_writers._router import FrameRouter, PositionInfo
    from ome_writers.schema import AcquisitionSettings, Dimension, Plate, Position

try:
    from yaozarrs import DimSpec, v05
    from yaozarrs.write.v05 import Bf2RawBuilder, PlateBuilder, prepare_image
except ImportError as e:
    raise ImportError(
        f"{__name__} requires yaozarrs with write support: "
        "`pip install yaozarrs[write-zarr]`."
    ) from e


class ChunkBuffer:
    """Manages in-memory chunk buffers for a single position's array.

    Supports multiple active chunks to handle all acquisition patterns,
    including transposed storage order and non-contiguous chunked dimensions.
    """

    def __init__(
        self,
        index_shape: tuple[int | None, ...],  # Shape of index dims only
        chunk_shape: tuple[int, ...],  # Chunk sizes for index dims
        frame_shape: tuple[int, int],  # Y, X dimensions
        dtype: str | np.dtype[Any],
    ) -> None:
        """Initialize chunk buffer.

        Parameters
        ----------
        index_shape : tuple[int | None, ...]
            Shape of index dimensions (excluding frame dimensions).
            None values indicate unlimited dimensions.
        chunk_shape : tuple[int, ...]
            Chunk sizes for each index dimension.
        frame_shape : tuple[int, int]
            Shape of frame dimensions (Y, X).
        dtype : str | np.dtype[Any]
            Data type for the array.
        """
        self.index_shape = list(index_shape)  # Make mutable for updates
        self.chunk_shape = chunk_shape
        self.frame_shape = frame_shape
        self.dtype = dtype

        # Active chunk buffers: {chunk_coords: ndarray}
        # chunk_coords = tuple of chunk indices in storage space
        # buffer shape = chunk_shape + frame_shape
        self._active_chunks: dict[tuple[int, ...], np.ndarray] = {}

        # Track filled frames per chunk: {chunk_coords: set of frame_indices}
        # frame_indices are positions within the chunk (not global)
        self._filled_frames: dict[tuple[int, ...], set[tuple[int, ...]]] = {}

    def add_frame(
        self,
        storage_index: tuple[int, ...],  # Index in storage space (no frame dims)
        frame: np.ndarray,  # 2D frame (Y, X)
    ) -> tuple[int, ...] | None:
        """Add frame to buffer. Returns chunk_coords if chunk is complete."""
        chunk_coords = self._get_chunk_coords(storage_index)
        frame_within_chunk = self._get_frame_within_chunk(storage_index)

        # Initialize chunk buffer if needed
        if chunk_coords not in self._active_chunks:
            self._allocate_chunk(chunk_coords)

        # Store frame in chunk buffer
        buffer = self._active_chunks[chunk_coords]
        buffer[frame_within_chunk] = frame
        self._filled_frames[chunk_coords].add(frame_within_chunk)

        # Check if chunk is complete
        if self._is_chunk_complete(chunk_coords):
            return chunk_coords
        return None

    def _get_chunk_coords(self, storage_index: tuple[int, ...]) -> tuple[int, ...]:
        """Convert storage index to chunk coordinates."""
        return tuple(
            idx // chunk_size
            for idx, chunk_size in zip(storage_index, self.chunk_shape, strict=False)
        )

    def _get_frame_within_chunk(
        self, storage_index: tuple[int, ...]
    ) -> tuple[int, ...]:
        """Get frame's position within its chunk."""
        return tuple(
            idx % chunk_size
            for idx, chunk_size in zip(storage_index, self.chunk_shape, strict=False)
        )

    def _allocate_chunk(self, chunk_coords: tuple[int, ...]) -> None:
        """Allocate buffer for a new chunk."""
        import numpy as np

        # Determine actual chunk shape (may be partial at array boundaries)
        actual_chunk_shape = self._get_actual_chunk_shape(chunk_coords)
        full_shape = actual_chunk_shape + self.frame_shape

        self._active_chunks[chunk_coords] = np.zeros(full_shape, dtype=self.dtype)
        self._filled_frames[chunk_coords] = set()

    def _get_actual_chunk_shape(self, chunk_coords: tuple[int, ...]) -> tuple[int, ...]:
        """Compute actual chunk shape (handles partial chunks at boundaries)."""
        actual_shape = []
        for i, (cc, cs) in enumerate(zip(chunk_coords, self.chunk_shape, strict=False)):
            start = cc * cs
            if self.index_shape[i] is None:
                # Unlimited dimension - use full chunk size
                actual_shape.append(cs)
            else:
                # Limited dimension - may be partial at boundary
                end = min(start + cs, self.index_shape[i])
                actual_shape.append(end - start)
        return tuple(actual_shape)

    def _is_chunk_complete(self, chunk_coords: tuple[int, ...]) -> bool:
        """Check if all expected frames in chunk have been written."""
        import numpy as np

        actual_shape = self._get_actual_chunk_shape(chunk_coords)
        expected_count = int(np.prod(actual_shape))
        return len(self._filled_frames[chunk_coords]) == expected_count

    def get_chunk_for_flush(
        self, chunk_coords: tuple[int, ...]
    ) -> tuple[tuple[int, ...], np.ndarray]:
        """Extract chunk buffer and compute storage location for writing."""
        buffer = self._active_chunks.pop(chunk_coords)
        self._filled_frames.pop(chunk_coords)

        # Calculate starting storage index for this chunk
        storage_start = tuple(
            cc * cs for cc, cs in zip(chunk_coords, self.chunk_shape, strict=False)
        )

        return storage_start, buffer

    def flush_all_partial(self) -> list[tuple[tuple[int, ...], np.ndarray]]:
        """Flush all remaining chunks during finalize (may be incomplete)."""
        chunks_to_flush = []
        for chunk_coords in list(self._active_chunks.keys()):
            storage_start, buffer = self.get_chunk_for_flush(chunk_coords)
            chunks_to_flush.append((storage_start, buffer))
        return chunks_to_flush

    def estimate_memory_usage(self) -> int:
        """Estimate current memory usage in bytes."""
        return sum(buf.nbytes for buf in self._active_chunks.values())


class YaozarrsBackend(ArrayBackend):
    """Base backend using yaozarrs for OME-Zarr v0.5 structure.

    Subclasses must define the `writer` class attribute to specify which
    yaozarrs writer to use (e.g., "zarr", "tensorstore").
    """

    def __init__(self) -> None:
        self._arrays: list[Any] = []
        self._image_group_paths: list[str] = []  # Parallel to _arrays, for metadata
        self._metadata_cache: dict[str, dict] = {}  # group path -> attrs dict
        self._finalized = False
        self._root: Path | None = None
        self._use_chunk_buffering = False
        self._chunk_buffers: list[ChunkBuffer] | None = None

    @abstractmethod
    def _get_writer(self) -> Literal["zarr", "tensorstore"] | Callable[..., Any]:
        """Return the writer to use for array creation.

        Subclasses can override to provide a custom CreateArrayFunc.
        """

    def _post_prepare(self, settings: AcquisitionSettings) -> None:
        """Hook called after yaozarrs creates the structure, before metadata caching.

        Subclasses can override to do additional setup (e.g., create streams).
        """

    def prepare(self, settings: AcquisitionSettings, router: FrameRouter) -> None:
        """Initialize OME-Zarr storage structure."""

        self._finalized = False
        root = Path(settings.root_path).expanduser().resolve()
        self._root = root
        positions = settings.positions

        # Build storage metadata
        storage_dims = settings.array_storage_dimensions
        shape = tuple(d.count if d.count is not None else 1 for d in storage_dims)
        dtype = settings.dtype
        chunks, shards = _get_chunks_and_shards(storage_dims)
        # this single image model is reused for all positions
        # (the underlying assumption is that we currently don't support inhomogeneous
        # shapes/dtypes across positions)
        image = _build_yaozarrs_image_model(storage_dims)

        # Get writer from hook (subclasses can override)
        writer = self._get_writer()

        # Plate mode
        if settings.plate is not None:
            # mapping of {(row, column): [(position_index, Position), ...]}
            well_positions = {}
            for i, pos in enumerate(positions):
                key = (pos.row, pos.column)
                well_positions.setdefault(key, []).append((i, pos))

            # Build plate metadata and arrays
            builder = PlateBuilder(
                root,
                plate=_build_yaozarrs_plate_model(settings.plate, well_positions),
                overwrite=settings.overwrite,
                chunks=chunks,
                shards=shards,
                writer=writer,
            )

            for (row, col), pos_list in well_positions.items():
                images_dict = {
                    pos.name: (image, [(shape, dtype)]) for _idx, pos in pos_list
                }
                builder.add_well(row=row, col=col, images=images_dict)

            _, all_arrays = builder.prepare()
            # Map position index to array path and store arrays
            self._image_group_paths = [
                f"{pos.row}/{pos.column}/{pos.name}" for pos in positions
            ]
            self._arrays = [
                all_arrays[f"{parent}/0"] for parent in self._image_group_paths
            ]

        # Single position
        elif len(positions) == 1:
            _, all_arrays = prepare_image(
                root,
                image,
                datasets=[(shape, dtype)],
                overwrite=settings.overwrite,
                chunks=chunks,
                shards=shards,
                writer=writer,  # type: ignore[arg-type]
            )
            self._image_group_paths = ["."]
            self._arrays = [all_arrays["0"]]

        # Multi-position (bf2raw layout)
        else:
            builder = Bf2RawBuilder(
                root,
                overwrite=settings.overwrite,
                chunks=chunks,
                shards=shards,
                writer=writer,
            )
            for pos in positions:
                builder.add_series(pos.name, image, [(shape, dtype)])

            _, all_arrays = builder.prepare()
            self._image_group_paths = [pos.name for pos in positions]
            self._arrays = [
                all_arrays[f"{parent}/0"] for parent in self._image_group_paths
            ]

        # Post-prepare hook (subclasses can do additional work)
        self._post_prepare(settings)

        # Cache metadata immediately after creation
        # This is used later for get_metadata() and update_metadata()
        self._cache_metadata_from_arrays(root)

        # Initialize chunk buffering if beneficial
        self._use_chunk_buffering = self._should_use_chunk_buffering(storage_dims)

        if self._use_chunk_buffering:
            # Extract shapes for buffer initialization
            index_dims = storage_dims[:-2]  # Exclude frame dims
            frame_dims = storage_dims[-2:]  # Y, X

            index_shape = tuple(d.count for d in index_dims)
            chunk_shape = tuple(
                d.chunk_size if d.chunk_size is not None else 1 for d in index_dims
            )
            frame_shape = tuple(d.count for d in frame_dims)

            # Create one buffer per position
            self._chunk_buffers = [
                ChunkBuffer(
                    index_shape=index_shape,
                    chunk_shape=chunk_shape,
                    frame_shape=frame_shape,
                    dtype=dtype,
                )
                for _ in range(len(positions))
            ]

            # Check for potentially high memory usage and warn
            self._check_buffer_memory_warning(chunk_shape, frame_shape, dtype)
        else:
            self._chunk_buffers = None

    def write(
        self,
        position_info: PositionInfo,
        index: tuple[int, ...],
        frame: np.ndarray,
    ) -> None:
        """Write frame to specified location with auto-resize for unlimited dims."""
        if self._finalized:  # pragma: no cover
            raise RuntimeError("Cannot write after finalize().")
        if not self._arrays:  # pragma: no cover
            raise RuntimeError("Backend not prepared. Call prepare() first.")

        pos_idx = position_info[0]
        array = self._arrays[pos_idx]

        # Resize if needed (index may be shorter than shape due to spatial dims)
        new_shape = list(array.shape)
        for i, idx_val in enumerate(index):
            new_shape[i] = max(new_shape[i], idx_val + 1)

        if new_shape != list(array.shape):
            self._resize(array, new_shape)
            # Update buffer shape if using buffering
            if self._use_chunk_buffering and self._chunk_buffers is not None:
                self._chunk_buffers[pos_idx].index_shape = new_shape[:-2]

        # Route to buffered or direct write
        if self._use_chunk_buffering:
            self._write_with_buffering(pos_idx, array, index, frame)
        else:
            self._write(array, index, frame)

    def _write_with_buffering(
        self,
        position_idx: int,
        array: Any,  # Backend-specific array (zarr.Array or tensorstore.TensorStore)
        index: tuple[int, ...],
        frame: np.ndarray,
    ) -> None:
        """Write frame using chunk buffering."""
        if self._chunk_buffers is None:  # pragma: no cover
            raise RuntimeError("Chunk buffers not initialized.")

        buffer = self._chunk_buffers[position_idx]

        # Add frame to buffer
        chunk_coords = buffer.add_frame(index, frame)

        # If chunk is complete, flush it
        if chunk_coords is not None:
            storage_start, chunk_data = buffer.get_chunk_for_flush(chunk_coords)
            self._write_chunk(array, storage_start, chunk_data)

    def _write_chunk(
        self,
        array: Any,  # Backend-specific array (zarr.Array or tensorstore.TensorStore)
        start_index: tuple[int, ...],
        chunk_data: np.ndarray,
    ) -> None:
        """Write a complete chunk to the array.

        This is the key method that writes entire chunks at once,
        reducing I/O operations significantly.
        """
        # Build slice for the chunk region
        chunk_shape = chunk_data.shape
        slices = tuple(
            slice(start, start + size)
            for start, size in zip(start_index, chunk_shape, strict=False)
        )
        # Write entire chunk in one operation
        array[slices] = chunk_data

    def get_metadata(self) -> dict[str, dict]:
        """Get metadata from all array groups in the zarr hierarchy.

        Returns a dict mapping group paths to their .zattrs contents.
        Each .zattrs dict typically contains:
        - "ome": yaozarrs v05.Image model (structured OME metadata)
        - "omero": dict with channel colors/names (if applicable)
        - Custom keys for non-standard metadata (timestamps, etc.)

        Returns
        -------
        dict[str, dict]
            Mapping of group paths to .zattrs dicts, or empty dict if not prepared.

        Examples
        --------
        For single position:
            {".": {"ome": {...}}}  # root group attrs

        For multi-position:
            {"pos0": {"ome": {...}}, "pos1": {"ome": {...}}}

        For plates:
            {"A/1/field_0": {"ome": {...}}, "A/2/field_0": {"ome": {...}}}
        """
        if not self._metadata_cache:  # pragma: no cover
            return {}

        return deepcopy(self._metadata_cache)

    def update_metadata(self, metadata: dict[str, dict]) -> None:
        """Update metadata for all array groups in the zarr hierarchy.

        Parameters
        ----------
        metadata : dict[str, dict]
            Mapping of group paths to .zattrs dicts. Keys should match those
            returned by get_metadata(). Values are dicts that will be written
            to the corresponding .zattrs files.

        Raises
        ------
        KeyError
            If a path in metadata doesn't correspond to a group.
        RuntimeError
            If backend is not prepared.
        """
        if not self._metadata_cache:  # pragma: no cover
            raise RuntimeError("Backend not prepared. Call prepare() first.")

        if self._root is None:  # pragma: no cover
            warnings.warn("Root path is unknown. Cannot update metadata.", stacklevel=2)
            return

        for path, attrs in metadata.items():
            if path not in self._metadata_cache:
                raise KeyError(f"Unknown path: {path}")

            if (zarr_json := self._root / path / "zarr.json").exists():
                data = cast("dict", json.loads(zarr_json.read_text()))
                data.setdefault("attributes", {}).update(attrs)
                zarr_json.write_text(json.dumps(data, indent=2))
                self._metadata_cache[path] = deepcopy(attrs)

    def finalize(self) -> None:
        """Flush and release resources."""
        if not self._finalized:
            # Flush any remaining partial chunks
            if self._use_chunk_buffering and self._chunk_buffers is not None:
                for pos_idx, buffer in enumerate(self._chunk_buffers):
                    if self._arrays:  # Arrays may have been cleared already
                        array = self._arrays[pos_idx]
                        for storage_start, chunk_data in buffer.flush_all_partial():
                            self._write_chunk(array, storage_start, chunk_data)

            self._arrays.clear()
            self._image_group_paths.clear()
            self._chunk_buffers = None
            self._finalized = True

    def _write(self, array: Any, index: tuple[int, ...], frame: np.ndarray) -> None:
        """Write frame to array at specified index."""
        array[index] = frame

    def _resize(self, array: Any, new_shape: Sequence[int]) -> None:
        """Resize array to new shape."""
        array.resize(new_shape)

    def _cache_metadata_from_arrays(self, root: Path) -> None:
        """Cache metadata from parent groups.

        Reads zarr.json and caches their attributes in position order.
        """
        for parent_path in self._image_group_paths:
            if (zarr_json := root / parent_path / "zarr.json").exists():
                data = json.loads(zarr_json.read_text())
                self._metadata_cache[parent_path] = data.get("attributes", {})

    def _should_use_chunk_buffering(self, storage_dims: list[Dimension]) -> bool:
        """Check if chunk buffering would be beneficial.

        Returns True if any index dimension (excluding last 2 frame dims)
        has chunk_size > 1.
        """
        # Only check index dimensions (exclude frame dimensions)
        for dim in storage_dims[:-2]:
            if dim.chunk_size is not None and dim.chunk_size > 1:
                return True
        return False

    def _check_buffer_memory_warning(
        self,
        chunk_shape: tuple[int, ...],
        frame_shape: tuple[int, int],
        dtype: str,
    ) -> None:
        """Warn if chunk buffering may require high memory."""
        import numpy as np

        bytes_per_chunk = (
            int(np.prod(chunk_shape + frame_shape)) * np.dtype(dtype).itemsize
        )

        if bytes_per_chunk > 50_000_000:  # > 50 MB per chunk
            warnings.warn(
                f"Chunk buffering will use ~{bytes_per_chunk / 1e6:.1f} MB per active "
                f"chunk. Consider adjusting chunk sizes to reduce memory usage.",
                stacklevel=3,
            )


class ZarrBackend(YaozarrsBackend):
    """OME-Zarr writer using zarr-python via yaozarrs."""

    def _get_writer(self) -> Literal["zarr"]:
        return "zarr"

    def is_incompatible(self, settings: AcquisitionSettings) -> Literal[False] | str:
        if not settings.root_path.endswith(".zarr"):  # pragma: no cover
            return "Root path must end with .zarr for ZarrBackend."
        return False


class TensorstoreBackend(YaozarrsBackend):
    """OME-Zarr writer using tensorstore via yaozarrs."""

    def _get_writer(self) -> Literal["tensorstore"]:
        return "tensorstore"

    def __init__(self) -> None:
        super().__init__()
        self._futures: list[Any] = []

    def is_incompatible(self, settings: AcquisitionSettings) -> Literal[False] | str:
        if not settings.root_path.endswith(".zarr"):  # pragma: no cover
            return "Root path must end with .zarr for TensorstoreBackend."
        return False

    def _write(self, array: Any, index: tuple[int, ...], frame: np.ndarray) -> None:
        """Write frame to array at specified index, async for tensorstore."""
        self._futures.append(array[index].write(frame))

    def _write_chunk(
        self,
        array: Any,  # tensorstore.TensorStore
        start_index: tuple[int, ...],
        chunk_data: np.ndarray,
    ) -> None:
        """Write chunk asynchronously with future tracking."""
        chunk_shape = chunk_data.shape
        slices = tuple(
            slice(start, start + size)
            for start, size in zip(start_index, chunk_shape, strict=False)
        )
        # Async write for tensorstore
        self._futures.append(array[slices].write(chunk_data))

    def _resize(self, array: Any, new_shape: Sequence[int]) -> None:
        """Resize array to new shape, using exclusive_max for tensorstore."""
        array.resize(exclusive_max=new_shape).result()

    def finalize(self) -> None:
        """Flush and release resources."""
        while self._futures:
            self._futures.pop().result()
        super().finalize()


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------


def _get_chunks_and_shards(
    dims: list[Dimension],
) -> tuple[tuple[int, ...], tuple[int, ...] | None]:
    """Compute chunk and shard sizes from dimensions."""
    chunks = []
    shards = []
    has_shards = False
    n = len(dims)

    for i, dim in enumerate(dims):
        # Chunks
        if dim.chunk_size is not None:
            chunks.append(dim.chunk_size)
        elif i >= n - 2:  # Last 2 dims (spatial)
            chunks.append(dim.count)
        else:
            chunks.append(1)

        # Shards
        if dim.shard_size is not None:
            shards.append(dim.shard_size)
            has_shards = True
        else:
            shards.append(chunks[-1])  # Default to chunk size

    return tuple(chunks), tuple(shards) if has_shards else None


def _build_yaozarrs_image_model(dims: list[Dimension]) -> v05.Image:
    """Build yaozarrs v05 Image metadata from Dimensions."""
    dim_specs = [
        DimSpec(
            name=dim.name,
            size=dim.count,
            type=dim.type,
            unit=dim.unit,
            scale=1.0 if dim.scale is None else dim.scale,
            translation=dim.translation,
        )
        for dim in dims
    ]
    return v05.Image(multiscales=[v05.Multiscale.from_dims(dim_specs)])


def _build_yaozarrs_plate_model(
    plate: Plate, well_positions: dict[tuple[str, str], list[tuple[int, Position]]]
) -> v05.Plate:
    """Build yaozarrs v05 Plate metadata from ome-writers Plate schema."""
    return v05.Plate(
        plate=v05.PlateDef(
            name=plate.name,
            rows=[v05.Row(name=name) for name in plate.row_names],
            columns=[v05.Column(name=name) for name in plate.column_names],
            wells=[
                v05.PlateWell(
                    path=f"{row}/{col}",
                    rowIndex=plate.row_names.index(row),
                    columnIndex=plate.column_names.index(col),
                )
                for row, col in well_positions
            ],
        )
    )
