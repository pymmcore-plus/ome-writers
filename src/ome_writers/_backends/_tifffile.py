"""OME-TIFF backend using tifffile for sequential writes."""

from __future__ import annotations

import math
import threading
import warnings
from dataclasses import dataclass
from itertools import count
from queue import Queue
from typing import TYPE_CHECKING, Literal

from ome_writers._backends._backend import ArrayBackend
from ome_writers._backends._ome_xml import prepare_metadata

if TYPE_CHECKING:
    from collections.abc import Iterator
    from typing import Any

    import numpy as np

    from ome_writers._backends._ome_xml import OmeXMLMirror
    from ome_writers._router import FrameRouter
    from ome_writers._schema import AcquisitionSettings, Dimension

try:
    import ome_types.model as ome
    import tifffile
except ImportError as e:
    raise ImportError(
        f"{__name__} requires tifffile and ome-types: "
        "`pip install ome-writers[tifffile]`."
    ) from e

PLANE_KEYS = {
    "delta_t",
    "exposure_time",
    "position_x",
    "position_y",
    "position_z",
}


@dataclass
class PositionManager:
    """Per-position writer/metadata state for TIFF backend."""

    file_path: str
    thread: WriterThread | None
    queue: Queue[np.ndarray | None]
    metadata_mirror: OmeXMLMirror

    def __post_init__(self) -> None:
        self._lock = threading.Lock()
        self._metadata_dirty: bool = False

    def update_metadata(self, metadata: ome.OME, flush: bool = False) -> None:
        """Update cached metadata and mark as dirty.  Optionally flush to file."""
        with self._lock:
            self.metadata_mirror.model = metadata
            self.metadata_mirror.mark_dirty()

        # careful... our lock is not re-entrant, so avoid deadlock
        self.metadata_mirror.flush(force=flush)

    def signal_stop(self) -> None:
        """Signal the writer thread to stop by sending None sentinel."""
        if self.queue is not None:
            self.queue.put(None)

    def finalize(self, index_dims: tuple[Dimension, ...] | None) -> None:
        """Wait for thread completion and update metadata with actual frames written.

        Parameters
        ----------
        index_dims : tuple[Dimension, ...] | None
            Dimensions used for storage indexing, or None if unavailable.
            (usually just T, C, Z)
        """
        # Wait for thread to finish
        if self.thread:
            self.thread.join(timeout=5)

        # Update metadata based on actual frames written
        if self.thread is None:
            # No thread means no TIFF file (e.g., companion OME-XML only)
            self.metadata_mirror.flush(force=True)
            return

        # Update dimension sizes and plane count based on actual frames written
        images = self.metadata_mirror.model.images
        if self.thread.frames_written and index_dims and images:
            pixels = images[0].pixels

            # Update the outermost dimension's size based on frames written
            # This handles both unbounded dims and incomplete bounded dims
            if index_dims:
                first, *inner = index_dims
                # Calculate actual size of outermost dimension
                inner_prod = math.prod([d.count or 1 for d in inner])
                actual_outer_size = self.thread.frames_written // inner_prod
                setattr(pixels, f"size_{first.name.lower()}", actual_outer_size)

            # Update plane count
            if data_blocks := pixels.tiff_data_blocks:
                data_blocks[0].plane_count = self.thread.frames_written
            self.metadata_mirror.flush(force=True)


class TiffBackend(ArrayBackend):
    """OME-TIFF backend using tifffile for sequential writes.

    TIFF files are written sequentially, with one file per position.
    The index parameter in write() is ignored since TIFF only supports
    sequential writing.
    """

    def __init__(self) -> None:
        self._finalized = False
        self._position_managers: dict[int, PositionManager] = {}
        self._storage_dims: tuple[Dimension, ...] | None = None
        self._dtype: str = ""
        self._frame_metadata: dict[int, list[dict[str, Any]]] = {}

    def is_incompatible(self, settings: AcquisitionSettings) -> Literal[False] | str:
        """Check if settings are compatible with TIFF backend."""
        # for now, assume we use the same compression strings as tifffile
        if settings.compression not in (None, "none") and not hasattr(
            tifffile.COMPRESSION, settings.compression.upper()
        ):  # pragma: no cover
            supported = {"none"} | set(tifffile.COMPRESSION.__members__.keys())
            return (
                f"Compression '{settings.compression}' is not supported by "
                f"TiffBackend. Supported: {supported}."
            )

        # Validate storage dimension names for OME-TIFF compatibility
        # OME-TIFF supports x, y, z, c, t (all StandardAxis except position)
        ome_dims = set("xyzct")
        for dim in settings.array_storage_dimensions:
            if dim.name.lower() not in ome_dims:  # pragma: no cover
                return (
                    f"Invalid dimension name '{dim.name}' for OME-TIFF. "
                    f"Valid names are: {', '.join(sorted(ome_dims))} "
                    f"(case-insensitive)."
                )

        return False

    def prepare(self, settings: AcquisitionSettings, router: FrameRouter) -> None:
        """Initialize OME-TIFF files and writer threads."""
        self._finalized = False
        self._storage_dims = storage_dims = settings.array_storage_dimensions
        # Extract index keys, excluding Y and X, example: ['t', 'c', 'z']
        self._index_keys = [d.name for d in storage_dims[:-2]]
        self._dtype = settings.dtype

        # Extract and validate compression
        compression = None
        if settings.compression not in (None, "none"):
            compression = getattr(tifffile.COMPRESSION, settings.compression.upper())

        # Compute shape from storage dimensions
        shape = tuple(d.count if d.count is not None else 1 for d in storage_dims)

        # Check if any dimension is unbounded
        has_unbounded = any(d.count is None for d in storage_dims)

        # Prepare OME-XML metadata mirrors
        # mapping of filepath -> OmeXMLMirror
        metas = prepare_metadata(settings)

        # Create writer thread for each position
        for fname, meta_mirror in metas.items():
            thread = q = None
            if meta_mirror.is_tiff:
                q = Queue()
                thread = WriterThread(
                    path=fname,
                    shape=shape,
                    dtype=self._dtype,
                    image_queue=q,
                    ome_xml=meta_mirror.model.to_xml(),
                    has_unbounded=has_unbounded,
                    compression=compression,
                )
                thread.start()
            self._position_managers[meta_mirror.pos_idx] = PositionManager(
                file_path=fname,
                thread=thread,
                queue=q,
                metadata_mirror=meta_mirror,
            )

    def write(
        self,
        position_index: int,
        index: tuple[int, ...],
        frame: np.ndarray,
        *,
        frame_metadata: dict[str, Any] | None = None,
    ) -> None:
        """Write frame sequentially to the appropriate position's TIFF file.

        The index parameter is ignored since TIFF writes are sequential.
        """
        if self._finalized:  # pragma: no cover
            raise RuntimeError("Cannot write after finalize().")
        if not self._position_managers:  # pragma: no cover
            raise RuntimeError("Backend not prepared. Call prepare() first.")

        manager = self._position_managers[position_index]
        manager.queue.put(frame)

        # Accumulate frame metadata with storage index
        if frame_metadata is not None:
            self._append_frame_metadata(position_index, index, frame_metadata)

    def _append_frame_metadata(
        self,
        position_index: int,
        index: tuple[int, ...],
        frame_metadata: dict[str, Any],
    ) -> None:
        if position_index not in self._frame_metadata:
            self._frame_metadata[position_index] = []
        # self._frame_metadata[position_index].append(meta_with_idx)

        mirror = self._position_managers[position_index].metadata_mirror
        model = mirror.model
        map_annotations = model.structured_annotations.map_annotations  # type: ignore
        if images := model.images:
            # {"the_z": 0, "the_c": 1, ...}
            plane_kwargs = {
                f"the_{k}": v for k, v in zip(self._index_keys, index, strict=False)
            }
            plane_kwargs.update(
                {f"the_{k}": 0 for k in "tcz" if k not in self._index_keys}
            )

            extra_kwargs = {}
            for key, value in frame_metadata.items():
                if key in PLANE_KEYS:
                    plane_kwargs[key] = value
                else:
                    extra_kwargs[key] = value

            # meta_with_idx = {**frame_metadata, "storage_index": index}
            annotation = ome.MapAnnotation(value=ome.Map.model_validate(extra_kwargs))
            map_annotations.append(annotation)
            planes = images[position_index].pixels.planes
            planes.append(
                ome.Plane(
                    **plane_kwargs,
                    annotation_refs=[ome.AnnotationRef(id=annotation.id)],
                )
            )
            mirror.mark_dirty()

    def finalize(self) -> None:
        """Flush and close all TIFF writers."""
        if not self._finalized:
            # Signal all threads to stop (parallel shutdown begins)
            for manager in self._position_managers.values():
                manager.signal_stop()

            # Finalize each position (wait for thread and update metadata)
            for manager in self._position_managers.values():
                index_dims = self._storage_dims[:-2] if self._storage_dims else None
                manager.finalize(index_dims)

            self._finalized = True

    def get_metadata(self) -> dict[int, ome.OME]:
        """Get the base OME metadata generated from acquisition settings.

        Returns a mapping of position indices to `ome_types.OME` objects.  The `OME`
        objects represent the metadata as it would appear in the TIFF or companion
        file for that position.

        !!! note
            The special "position index" of -1, if present, represents metadata
            in the companion.ome file, if applicable.

        Users can modify these objects as needed and pass a mapping of position indices
        to `ome_types.OME` objects back to `update_metadata()`.

        See the `ome-types` documentation for details on modifying OME metadata:
        <https://ome-types.readthedocs.io/en/latest/API/ome_types/>

        Returns
        -------
        dict[int, ome_types.model.OME]
            Mapping of position indices to OME metadata objects, or empty dict if
            prepare() has not been called yet.
        """
        if not self._position_managers:  # pragma: no cover
            return {}

        return {
            p_idx: manager.metadata_mirror.model.model_copy(deep=True)
            for p_idx, manager in self._position_managers.items()
        }

    def update_metadata(self, metadata: dict[int, ome.OME]) -> None:
        """Update the OME metadata in the TIFF files.

        The metadata argument MUST be a dict mapping position indices to
        `ome_types.OME` instances, with the special index -1 representing the
        companion.ome file, if applicable.

        This method must be called AFTER exiting the stream context (after
        finalize() completes), as TIFF files must be closed before metadata
        can be updated.

        Parameters
        ----------
        metadata : dict[int, ome_types.model.OME]
            Mapping of position indices to OME metadata objects. Keys should match
            those returned by get_metadata().

        Raises
        ------
        TypeError
            If metadata is not a dict or values are not ome_types.model.OME instances.
        KeyError
            If a position index in metadata doesn't correspond to a position.
        RuntimeError
            If called before finalize() completes, or if metadata update fails.
        """
        if not self._finalized:  # pragma: no cover
            raise RuntimeError(
                "update_metadata() must be called after the stream context exits. "
                "TIFF files must be closed before metadata can be updated."
            )

        if not isinstance(metadata, dict):
            raise TypeError(
                "Expected dict[int, ome_types.model.OME] metadata, "
                f"got {type(metadata)}"
            )

        for pos_idx, meta in metadata.items():
            if not isinstance(meta, ome.OME):
                raise TypeError(
                    f"Expected ome_types.model.OME for position {pos_idx}, "
                    f"got {type(meta)}"
                )

            try:
                # not calling deep copy here, since this is currently only ever called
                # after finalize().  i.e. we're done.
                self._position_managers[pos_idx].update_metadata(meta, flush=True)
            except KeyError as e:  # pragma: no cover
                raise KeyError(f"Unknown position index: {pos_idx}") from e


class WriterThread(threading.Thread):
    """Background thread for sequential TIFF writing."""

    def __init__(
        self,
        path: str,
        shape: tuple[int, ...],
        dtype: str,
        image_queue: Queue[np.ndarray | None],
        ome_xml: str = "",
        pixelsize: float = 1.0,
        has_unbounded: bool = False,
        compression: tifffile.COMPRESSION | None = None,
    ) -> None:
        super().__init__(daemon=True, name=f"TiffWriterThread-{next(_thread_counter)}")
        self._path = path
        self._shape = shape
        self._dtype = dtype
        self._image_queue = image_queue
        # Encode to UTF-8 bytes
        # critical: if you pass a str to tifffile.tiffcomment, it requires ASCII
        # which limits the ability to properly express characters like 'µ' in
        # physical units.  The OME-TIFF spec, however, explicitly requests UTF-8.
        # passing in bytes directly circumvents tifffile conversion and preserves
        # encoding.
        self._ome_xml_bytes = ome_xml.encode("utf-8")
        self._res = 1 / pixelsize
        self._has_unbounded = has_unbounded
        self._compression = compression
        self.frames_written = 0  # Track actual frames written for unbounded dims

    def run(self) -> None:
        """Write frames from queue to TIFF file sequentially."""
        # Wait for first frame before opening file - if close is called before
        # any frames are written, we get None and can return early
        first_frame = self._image_queue.get()
        if first_frame is None:
            return

        def _queue_iterator() -> Iterator[np.ndarray]:
            """Yield first frame, then frames from queue until None."""
            self.frames_written += 1
            yield first_frame
            while True:
                frame = self._image_queue.get()
                if frame is None:
                    break
                self.frames_written += 1
                yield frame

        try:
            with tifffile.TiffWriter(
                self._path, bigtiff=True, ome=False, shaped=False
            ) as writer:
                # Write frames individually for both bounded and unbounded dimensions.
                # This approach:
                # - Doesn't promise a frame count upfront (no shape parameter)
                # - Handles incomplete writes gracefully (iterator can end early)
                # - Lets tifffile discover the actual count as frames arrive
                # Note: contiguous=True is incompatible with compression, so we only
                # use it when compression is disabled
                use_contiguous = self._compression is None
                for i, frame in enumerate(_queue_iterator()):
                    writer.write(
                        frame,
                        contiguous=use_contiguous,
                        dtype=self._dtype,
                        resolution=(self._res, self._res),
                        resolutionunit=tifffile.RESUNIT.MICROMETER,
                        photometric=tifffile.PHOTOMETRIC.MINISBLACK,
                        description=self._ome_xml_bytes if i == 0 else None,
                        compression=self._compression,
                    )
        except Exception as e:  # pragma: no cover
            # Unexpected errors - log and continue
            warnings.warn(
                f"Unexpected error during TIFF write: {e}",
                RuntimeWarning,
                stacklevel=2,
            )


_thread_counter = count()
