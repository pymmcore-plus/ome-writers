"""OME-TIFF backend using tifffile for sequential writes."""

from __future__ import annotations

import json
import math
import threading
import time
import traceback
import warnings
import weakref
from dataclasses import dataclass
from itertools import count
from queue import Queue
from typing import TYPE_CHECKING, Literal, cast

import numpy as np

from ome_writers._backends._backend import ArrayBackend
from ome_writers._backends._ome_xml import prepare_metadata, tiff_xmls
from ome_writers._backends._tiff_array import FinalizedTiffArray, LiveTiffArray

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator, Mapping, Sequence
    from typing import Any

    from ome_writers._backends._backend import ArrayLike
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

# Minimum seconds without any writer writing a frame before `finalize` gives up
# (detection can take up to ~2x this; see `_join_writers`)
WRITER_STALL_TIMEOUT = 60.0


@dataclass
class PositionManager:
    """Per-position writer/metadata state for TIFF backend."""

    file_path: str
    thread: WriterThread | None
    queue: Queue[np.ndarray | None]
    metadata_mirror: OmeXMLMirror
    writer: tifffile.TiffWriter | None
    n_queued: int = 0  # frames (incl. placeholders) sent to the writer thread

    def __post_init__(self) -> None:
        self._lock = threading.Lock()

    def update_metadata(self, metadata: ome.OME, flush: bool = False) -> None:
        """Update cached metadata and mark as dirty.  Optionally flush to file."""
        with self._lock:
            self.metadata_mirror.model = metadata
            self.metadata_mirror.mark_dirty()

        # careful... our lock is not re-entrant, so avoid deadlock
        self.flush(force=flush)

    def signal_stop(self) -> None:
        """Signal the writer thread to stop by sending None sentinel."""
        if self.queue is not None:
            self.queue.put(None)

    def flush(self, force: bool = False, xml: str | None = None) -> None:
        """Write the OME-XML to file if it has been modified (or if forced)."""
        # A TIFF file must be closed, with a frame (IFD) to hold the OME-XML
        thread = self.thread
        if thread is None or (thread.frames_written and not thread.is_alive()):
            try:
                self.metadata_mirror.flush(force=force, xml=xml)
            except Exception as e:
                # writing to the file of a failed writer may fail too (e.g. disk full)
                if thread is None or thread.error is None:
                    raise
                warnings.warn(
                    f"OME-XML not written to {self.file_path}: {e}", stacklevel=3
                )
        elif force:
            warnings.warn(
                f"OME-XML not written to {self.file_path}: the file has no frames, "
                "or its writer thread is still running.",
                stacklevel=3,
            )


class TiffBackend(ArrayBackend):
    """OME-TIFF backend using tifffile for sequential writes.

    TIFF files are written sequentially, with one file per position.
    The index parameter in write() is ignored since TIFF only supports
    sequential writing.
    """

    def __init__(self) -> None:
        self._finalized = False
        # Lock ordering: always acquire self._state_lock BEFORE any
        # PositionManager._lock. set_global_metadata() and finalize() follow
        # this order; inverting it in new code will deadlock.
        self._state_lock = threading.Lock()
        self._position_managers: dict[int, PositionManager] = {}
        self._storage_dims: tuple[Dimension, ...] | None = None
        self._dtype: str = ""
        self._failed: WriterThread | None = None  # the first writer that failed

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
        self._failed = None
        self._storage_dims = storage_dims = settings.array_storage_dimensions
        # Extract index keys, excluding Y and X, example: ['t', 'c', 'z']
        self._index_keys = [d.name for d in storage_dims[:-2]]
        # frames (IFDs) are written sequentially, in acquisition order
        self._acq_index_dims = settings.index_dimensions
        self._dtype = settings.dtype
        self._frame_shape = settings.frame_shape

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
        xmls = tiff_xmls(metas)

        # Create writer thread for each position
        for fname, meta_mirror in metas.items():
            thread = q = writer = None
            if meta_mirror.is_tiff:
                # Create TiffWriter immediately - file exists with valid header
                writer = tifffile.TiffWriter(
                    fname, bigtiff=True, ome=False, shaped=False
                )

                q = Queue()
                thread = WriterThread(
                    writer=writer,
                    shape=shape,
                    dtype=self._dtype,
                    image_queue=q,
                    ome_xml=xmls[fname],
                    has_unbounded=has_unbounded,
                    compression=compression,
                    on_error=self._writer_failed,
                )
                thread.start()
            else:  # (no writer thread writes the companion file's initial OME-XML)
                meta_mirror.mark_dirty()
            self._position_managers[meta_mirror.pos_idx] = PositionManager(
                file_path=fname,
                thread=thread,
                queue=q,
                metadata_mirror=meta_mirror,
                writer=writer,
            )
        self._full_pos_idxs = self._global_target_pos_idxs()
        # full OME models differ only by root UUID until a user `update_metadata`
        self._models_in_sync = True

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
        if self._failed is not None:
            raise self._writer_error([self._failed]) from self._failed.error

        manager = self._position_managers[position_index]
        manager.queue.put(frame)
        manager.n_queued += 1

        # Accumulate frame metadata with storage index
        if frame_metadata is not None:
            self._append_frame_metadata(position_index, index, frame_metadata)

    def advance(self, indices: Sequence[tuple[int, tuple[int, ...]]]) -> None:
        """Write zero-filled placeholder frames to maintain sequential TIFF structure.

        TIFF files must be written sequentially. When frames are skipped during
        acquisition (e.g., autofocus failure), we write zero-filled placeholder
        frames to preserve the IFD order and structure.
        """
        if self._finalized:  # pragma: no cover
            raise RuntimeError("Cannot advance after finalize().")
        if not self._position_managers:  # pragma: no cover
            raise RuntimeError("Backend not prepared. Call prepare() first.")

        if self._failed is not None:
            raise self._writer_error([self._failed]) from self._failed.error
        if not indices:  # pragma: no cover
            return

        placeholder = np.zeros(self._frame_shape, dtype=self._dtype)
        # Write placeholder for each skipped frame
        for pos_idx, _storage_idx in indices:
            manager = self._position_managers[pos_idx]
            # Send to WriterThread queue (same path as regular writes)
            manager.queue.put(placeholder)
            manager.n_queued += 1

    def _writer_failed(self, thread: WriterThread) -> None:
        """Record a writer thread that failed (called from that thread)."""
        if self._failed is None:
            self._failed = thread

    def _writer_error(
        self, threads: Sequence[WriterThread], also: Sequence[str] = ()
    ) -> RuntimeError:
        """Return the error for failed writer `threads`, and mark them reported."""
        msgs = []
        for t in threads:
            t.error_reported = True
            msgs.append(
                f"TIFF writer for {t.path} failed after {t.frames_written} frames: "
                f"{t.error!r}"
            )
        return RuntimeError("; ".join([*msgs, *also]))

    def _append_frame_metadata(
        self,
        position_index: int,
        index: tuple[int, ...],
        frame_metadata: dict[str, Any],
    ) -> None:
        # {"the_z": 0, "the_c": 1, ...}
        plane_kwargs = {
            f"the_{k}": v for k, v in zip(self._index_keys, index, strict=False)
        }
        plane_kwargs.update({f"the_{k}": 0 for k in "tcz" if k not in self._index_keys})

        extra_kwargs = {}
        for key, value in frame_metadata.items():
            if key in PLANE_KEYS:
                plane_kwargs[key] = value
            else:
                extra_kwargs[key] = value

        annotation = None
        if extra_kwargs:
            annotation = ome.MapAnnotation(value=ome.Map.model_validate(extra_kwargs))
            plane_kwargs["annotation_refs"] = [ome.AnnotationRef(id=annotation.id)]
        plane = ome.Plane(**plane_kwargs)

        # Every mirror holding the full OME model describes this position's Image
        for pos_idx in self._full_pos_idxs:
            mirror = self._position_managers[pos_idx].metadata_mirror
            model = mirror.model
            if annotation is not None:
                if model.structured_annotations is None:
                    model.structured_annotations = ome.StructuredAnnotations()
                model.structured_annotations.map_annotations.append(annotation)
            model.images[position_index].pixels.planes.append(plane)
            mirror.mark_dirty()

    def _global_target_pos_idxs(self) -> list[int]:
        """Return pos_idxs of managers whose mirror holds the full OME model.

        Across every `multi_file_metadata` mode, the "full OME" mirrors are
        exactly the files that should carry global `MapAnnotation`s: the
        companion in companion-file mode, the master in master-tiff, the
        sole file in single-file, and every file in redundant mode.
        `prepare_metadata` marks stub mirrors by setting
        ``model.binary_only``; full-OME mirrors leave it unset.
        """
        if not self._position_managers:  # pragma: no cover
            raise RuntimeError("Backend not prepared. Call prepare() first.")

        return sorted(
            pos_idx
            for pos_idx, mgr in self._position_managers.items()
            if mgr.metadata_mirror.model.binary_only is None
        )

    def set_global_metadata(self, namespace: str, metadata: Mapping[str, Any]) -> None:
        """Attach acquisition-level metadata as a `MapAnnotation`.

        The annotation is placed under the target file's
        `OME.structured_annotations.map_annotations` with `Namespace=namespace` and is
        NOT referenced by any plane (distinguishing it from per-frame metadata). Same
        namespace replaces any prior value; different namespaces are siblings.

        Where the annotation lands on disk is driven by the OME-TIFF
        `multi_file_metadata` mode: single-file and `master-tiff` write to one file,
        `companion-file` writes to the companion, and `redundant` writes a copy into
        every per-position TIFF (matching that mode's "full OME in every file"
        guarantee).

        Can be called any time after `prepare()`, including after `close()`.
        Pre-close the update is deferred to `finalize()` (`tiffcomment` cannot
        safely run while a writer thread is still appending frames); post-close
        the update is flushed immediately via `tiffcomment` for TIFF mirrors
        or a direct file write for the companion file.
        """
        with self._state_lock:
            value = ome.Map.model_validate(
                {k: json.dumps(v) for k, v in metadata.items()}
            )
            # while in sync, one annotation (and ID) is shared by every full model, so
            # they stay identical apart from their root UUID. After `update_metadata`
            # (which keeps the caller's objects, uncopied), each gets its own.
            annotation = ome.MapAnnotation(namespace=namespace, value=value)
            managers = [
                self._position_managers[i] for i in self._global_target_pos_idxs()
            ]
            for manager in managers:
                if not self._models_in_sync:
                    annotation = ome.MapAnnotation(namespace=namespace, value=value)
                with manager._lock:
                    manager.metadata_mirror.set_map_annotation(annotation)

            # Post-finalize: writer threads are joined and files are closed, so
            # flush directly, serializing once while no user update has made the
            # full models differ (as in `finalize`)
            if self._finalized:
                mirrors = {m.file_path: m.metadata_mirror for m in managers}
                xmls = tiff_xmls(mirrors) if self._models_in_sync else {}
                for manager in managers:
                    with manager._lock:
                        manager.flush(force=True, xml=xmls.get(manager.file_path))

    def finalize(self) -> None:
        """Flush and close all TIFF writers."""
        with self._state_lock:
            managers = self._position_managers
            if self._finalized or not managers:
                return

            # Pad a partial last outer index (e.g. timepoint) with zero frames:
            # Bio-Formats can't read a single-file OME-TIFF that ends mid-timepoint
            index_dims = self._acq_index_dims
            inner_prod = math.prod(d.count or 1 for d in index_dims[1:])
            placeholder = np.zeros(self._frame_shape, dtype=self._dtype)
            for manager in managers.values():
                if manager.thread is not None:
                    for _ in range(-manager.n_queued % inner_prod):
                        manager.queue.put(placeholder)

            # Signal all threads to stop (parallel shutdown begins), then wait for
            # every one: each full OME model describes the frames of all positions.
            for manager in managers.values():
                manager.signal_stop()
            stalled = _join_writers(
                {m.file_path: m.thread for m in managers.values() if m.thread}
            )

            flush_errors: list[tuple[str, Exception]] = []
            try:
                # Record the frames actually written to each position (handles both
                # unbounded dims and incomplete bounded dims) in each full OME model
                outer = f"size_{index_dims[0].name.lower()}" if index_dims else ""
                frames_written = {
                    p: _frames_to_record(m.thread, inner_prod)
                    for p, m in managers.items()
                    if m.thread
                }
                full_mirrors: dict[str, OmeXMLMirror] = {}
                for target in self._global_target_pos_idxs():
                    mirror = managers[target].metadata_mirror
                    for p, n_frames in frames_written.items():
                        pixels = mirror.model.images[p].pixels
                        block = pixels.tiff_data_blocks[0]
                        size = max(1, -(-n_frames // inner_prod))  # (partial counts)
                        if block.plane_count != n_frames or (
                            outer and getattr(pixels, outer) != size
                        ):
                            block.plane_count = n_frames
                            if outer:
                                setattr(pixels, outer, size)
                            mirror.mark_dirty()
                    if mirror.dirty:
                        full_mirrors[managers[target].file_path] = mirror

                # no metadata update has made the full OME models differ yet, so
                # serialize them once (unchanged models are not written again)
                xmls = tiff_xmls(full_mirrors)
                for manager in managers.values():
                    # (skips the files of stalled writers)
                    try:
                        manager.flush(xml=xmls.get(manager.file_path))
                    except Exception as e:  # e.g. disk full: still try the others
                        flush_errors.append((manager.file_path, e))
            finally:
                # writers are closed, so later metadata updates can flush directly
                self._finalized = True

            # (errors already raised by `write` or `advance` are not raised again)
            failed = [
                m.thread
                for m in managers.values()
                if m.thread and m.thread.error and not m.thread.error_reported
            ]
            not_written = [
                f"OME-XML not written to {p}: {e!r}" for p, e in flush_errors
            ]
            first_flush_error = flush_errors[0][1] if flush_errors else None
            if failed:
                if stalled:
                    not_written.append(f"and these TIFF writers stalled: {stalled}")
                raise self._writer_error(failed, not_written) from failed[0].error
            if stalled:
                msg = (
                    f"TIFF writers wrote no frame for {WRITER_STALL_TIMEOUT} s, "
                    f"so these files may be incomplete: {stalled}"
                )
                raise TimeoutError(
                    "; ".join([msg, *not_written])
                ) from first_flush_error
            if first_flush_error is not None:
                raise first_flush_error

    def get_arrays(self) -> list[ArrayLike]:
        """Return array-like objects backed by TIFF files.

        If finalized and fully written: returns `FinalizedTiffArray` objects
        that read through *tifffile*'s page API (supports compression).

        Otherwise (live viewing, finalized-partial, or zero-frame): returns
        `LiveTiffArray` objects that read raw bytes at calculated offsets
        (requires uncompressed / contiguous writes).

        Returns
        -------
        list[ArrayLike]
            List of array-like objects (one per TIFF file).
        """

        if not self._position_managers:  # pragma: no cover
            raise RuntimeError("Backend not prepared. Call prepare() first.")

        arrays: list[ArrayLike] = []
        storage_dims = cast("tuple[Dimension]", self._storage_dims)
        for _, manager in sorted(self._position_managers.items()):
            if not manager.metadata_mirror.is_tiff:  # pragma: no cover
                continue  # Skip companion-only entries

            path = manager.file_path
            thread = manager.thread
            assert thread is not None, f"No WriterThread for {path}"

            if self._finalized:
                frames_written = thread.frames_written
                shape = tuple(d.count or 1 for d in storage_dims)
                expected_frames = math.prod(shape[:-2] or (1,))
                if frames_written >= expected_frames and frames_written > 0:
                    # Fully written: read via tifffile pages (compression OK)
                    tf = tifffile.TiffFile(path)
                    arr = FinalizedTiffArray(tf, shape, self._dtype)
                    weakref.finalize(arr, tf.close)
                    arrays.append(arr)
                    continue

            # Everything else needs raw byte access (uncompressed)
            if thread._compression is not None:
                raise NotImplementedError(
                    "Tiff viewing is not supported with compression enabled."
                )

            arrays.append(
                LiveTiffArray(
                    writer_thread=thread,
                    file_path=path,
                    storage_dims=storage_dims,
                    dtype=self._dtype,
                )
            )

        return arrays

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

        with self._state_lock:
            self._models_in_sync = False
            for pos_idx, meta in metadata.items():
                if not isinstance(meta, ome.OME):
                    raise TypeError(
                        f"Expected ome_types.model.OME for position {pos_idx}, "
                        f"got {type(meta)}"
                    )

                try:
                    # not calling deep copy here, since this is currently only ever
                    # called after finalize().  i.e. we're done.
                    self._position_managers[pos_idx].update_metadata(meta, flush=True)
                except KeyError as e:  # pragma: no cover
                    raise KeyError(f"Unknown position index: {pos_idx}") from e


class WriterThread(threading.Thread):
    """Background thread for sequential TIFF writing."""

    def __init__(
        self,
        writer: tifffile.TiffWriter,
        shape: tuple[int, ...],
        dtype: str,
        image_queue: Queue[np.ndarray | None],
        ome_xml: str = "",
        pixelsize: float = 1.0,
        has_unbounded: bool = False,
        compression: tifffile.COMPRESSION | None = None,
        on_error: Callable[[WriterThread], None] | None = None,
    ) -> None:
        super().__init__(daemon=True, name=f"TiffWriterThread-{next(_thread_counter)}")
        self._writer = writer
        self.path = writer.filehandle.path
        self._on_error = on_error
        self.error: Exception | None = None  # the exception that stopped writing
        self.error_reported = False
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
        self.state_lock = threading.Lock()  # Synchronize with readers
        self.data_offset: int | None = None  # Byte offset where frame data starts

    def run(self) -> None:
        """Write frames from queue to TIFF file sequentially."""
        # Wait for first frame - if None, close writer and return
        first_frame = self._image_queue.get()
        if first_frame is None:
            self._close_writer()
            return

        def _queue_iterator() -> Iterator[np.ndarray]:
            """Yield first frame, then frames from queue until None."""
            yield first_frame
            while True:
                frame = self._image_queue.get()
                if frame is None:
                    break
                yield frame

        try:
            # Write frames individually for both bounded and unbounded dimensions.
            # This approach:
            # - Doesn't promise a frame count upfront (no shape parameter)
            # - Handles incomplete writes gracefully (iterator can end early)
            # - Lets tifffile discover the actual count as frames arrive
            # Note: contiguous=True is incompatible with compression, so we only
            # use it when compression is disabled
            use_contiguous = self._compression is None
            for i, frame in enumerate(_queue_iterator()):
                # Write frame without holding lock - only this thread writes
                self._writer.write(
                    frame,
                    contiguous=use_contiguous,
                    dtype=self._dtype,
                    resolution=(self._res, self._res),
                    resolutionunit=tifffile.RESUNIT.MICROMETER,
                    photometric=tifffile.PHOTOMETRIC.MINISBLACK,
                    description=self._ome_xml_bytes if i == 0 else None,
                    compression=self._compression,
                )

                # Only hold lock when updating shared state
                with self.state_lock:
                    # Capture data offset after first frame (where frames start in file)
                    if i == 0 and self.data_offset is None:
                        try:
                            # ! private attribute access - relies on tifffile internals
                            self.data_offset = self._writer._dataoffset
                        except AttributeError:  # pragma: no cover
                            raise RuntimeError(
                                "tifffile.TiffWriter has no _dataoffset attribute. "
                                "Cannot determine frame data offset for live viewing."
                                "Please report this issue with the version of tifffile "
                                "you're using."
                            ) from None

                    # Increment counter AFTER write and flush to ensure readers
                    # only see frames that are fully written
                    self.frames_written += 1

        except Exception as e:
            # e.g. disk full: the backend raises it from the next write or finalize
            self._fail(e)
        finally:
            sentinel_queued = self.error is not None  # (stopped before the sentinel)
            self._close_writer()

        if sentinel_queued:
            # discard frames until the stop sentinel, so they don't pile up in memory
            # (and so a producer never blocks on a full queue)
            while self._image_queue.get() is not None:
                pass

    def _close_writer(self) -> None:
        """Close the TIFF file, recording an error if it's the first."""
        try:
            # (in contiguous mode, all IFDs but the first are written here)
            self._writer.close()
        except Exception as e:
            if self.error is None:
                self._fail(e)

    def _fail(self, error: Exception) -> None:
        """Record the `error` that stopped writing, and report it to the backend."""
        # free the locals of its traceback's frames (e.g. frame buffers), which the
        # stored error would otherwise keep alive (the traceback itself is kept)
        traceback.clear_frames(error.__traceback__)
        self.error = error
        if self._on_error is not None:
            self._on_error(self)


def _frames_to_record(thread: WriterThread, inner_prod: int) -> int:
    """Return the frames to record in the OME-XML for a writer `thread`."""
    # a failed writer can't be padded to a whole outer index (e.g. timepoint), so
    # only its whole ones are recorded: Bio-Formats misreads every plane of a
    # single-file OME-TIFF that ends mid-timepoint
    n = thread.frames_written
    return n - n % inner_prod if thread.error is not None else n


def _join_writers(threads: Mapping[str, WriterThread]) -> list[str]:
    """Wait for writer threads to finish, and return the keys of stalled ones.

    Writers are waited on for as long as any of them keeps writing frames (however
    long the queues take to drain), but once none has written a frame for at least
    `WRITER_STALL_TIMEOUT` seconds, all remaining writers are given up on together
    (e.g. stuck on an unresponsive network drive). Progress is only checked when a
    wait ends, so that can take up to ~2x `WRITER_STALL_TIMEOUT`.
    """
    total, deadline = -1, 0.0
    for thread in threads.values():
        while thread.is_alive():
            if (n := sum(t.frames_written for t in threads.values())) != total:
                total, deadline = n, time.monotonic() + WRITER_STALL_TIMEOUT
            if (remaining := deadline - time.monotonic()) <= 0:
                break
            thread.join(remaining)
    return [key for key, thread in threads.items() if thread.is_alive()]


_thread_counter = count()
