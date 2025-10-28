from __future__ import annotations

import importlib
import importlib.util
import threading
import warnings
from contextlib import suppress
from itertools import count
from pathlib import Path
from queue import Queue
from typing import TYPE_CHECKING, Any, cast

import numpy as np
from typing_extensions import Self

from ome_writers._dimensions import dims_to_ome
from ome_writers._stream_base import MultiPositionOMEStream

if TYPE_CHECKING:
    from collections.abc import Iterator, Sequence

    import ome_types.model as ome

    from ome_writers._dimensions import Dimension

else:
    with suppress(ImportError):
        import ome_types.model as ome


class TifffileStream(MultiPositionOMEStream):
    """A concrete OMEStream implementation for writing to OME-TIFF files.

    This writer supports two modes controlled by the `use_memmap` parameter:

    1. **Memmap mode** (`use_memmap=True`): Low RAM usage, disk-backed storage.
       Frames are written directly to memory-mapped files during acquisition,
       then converted to final OME-TIFF on flush(). Best for large acquisitions.

    2. **RAM mode** (`use_memmap=False`, default): Frames buffered in RAM during
       acquisition, then written to TIFF via background threads on flush().
       Best for smaller acquisitions where you want to minimize disk I/O.

    Both modes are designed for deterministic acquisitions where the full
    experiment shape is known ahead of time.

    If a 'p' (position) dimension is included in the dimensions, a separate
    OME-TIFF file will be created for each position.

    Parameters
    ----------
    use_memmap : bool, optional
        If True, use memory-mapped files for low-RAM disk-backed storage.
        If False (default), buffer frames in RAM until flush().

    Attributes
    ----------
    _memmaps : Dict[int, np.memmap] (memmap mode only)
        Per-position memory-mapped arrays for disk-backed storage.
    _frame_buffer : Dict[int, Dict[tuple, np.ndarray]] (RAM mode only)
        Per-position frame buffers keyed by storage index.
    """

    @classmethod
    def is_available(cls) -> bool:
        """Check if the tifffile package is available."""
        return bool(
            importlib.util.find_spec("tifffile") is not None
            and importlib.util.find_spec("ome_types") is not None
        )

    def __init__(self, *, use_memmap: bool = False) -> None:
        super().__init__()
        try:
            import ome_types.model
            import tifffile
        except ImportError as e:
            msg = (
                "TifffileStream requires tifffile and ome-types: "
                "`pip install ome-writers[tifffile]`."
            )
            raise ImportError(msg) from e

        self._tf = tifffile
        self._ome = ome_types.model
        self._use_memmap = use_memmap
        self._is_active = False
        self._tiff_shape: tuple[int, ...] = ()

        if use_memmap:
            # Memmap-based storage (low RAM, disk-backed)
            self._memmaps: dict[int, np.memmap] = {}
            self._memmap_paths: dict[int, str] = {}
            self._frame_count = 0
            self._flush_interval = 100  # Flush every N frames (maybe add to init?)
            self._position_ome_xml: dict[int, str] = {}
        else:
            # RAM-buffered storage with writer threads
            self._threads: dict[int, WriterThread] = {}
            self._queues: dict[int, Queue[np.ndarray | None]] = {}
            # Buffer to store frames until flush, allowing reordering
            self._frame_buffer: dict[int, dict[tuple[int, ...], np.ndarray]] = {}

    def __del__(self) -> None:
        """Cleanup memmap files on object deletion (memmap mode only)."""
        if self._use_memmap and hasattr(self, "_memmap_paths"):
            for mp in self._memmap_paths.values():
                try:
                    Path(mp).unlink(missing_ok=True)
                except Exception:
                    pass

    # ------------------------PUBLIC METHODS------------------------ #

    def create(
        self,
        path: str,
        dtype: np.dtype,
        dimensions: Sequence[Dimension],
        *,
        overwrite: bool = False,
    ) -> Self:
        # For TIFF, we need to handle the acquisition order specially because
        # TIFF files are always stored in [t, c, z, y, x] order regardless of
        # acquisition order. We'll build the index mapping to respect acquisition
        # order while ensuring frames are written in TIFF storage order.
        num_positions, tczyx_dims = self._init_positions_tiff(dimensions)

        self._delete_existing = overwrite
        self._path = Path(self._normalize_path(path))
        self._tiff_shape = tuple(d.size for d in tczyx_dims)

        fnames = self._prepare_files(self._path, num_positions, overwrite)

        if self._use_memmap:
            # Memmap mode: allocate disk-backed memmaps
            self._dtype = np.dtype(dtype)
            for p_idx, fname in enumerate(fnames):
                # create memmap file for this position with shape [t,c,z,y,x]
                mm_path = Path(f"{fname}.memmap")
                mm = np.memmap(
                    str(mm_path),
                    dtype=self._dtype,
                    mode="w+",
                    shape=self._tiff_shape,
                    order="C",
                )
                self._memmaps[p_idx] = mm
                self._memmap_paths[p_idx] = str(mm_path)

                ome = dims_to_ome(tczyx_dims, dtype=dtype, tiff_file_name=fname)
                self._position_ome_xml[p_idx] = ome.to_xml()

            # keep filenames for flush
            self._filenames = fnames
        else:
            # RAM mode: initialize frame buffers and writer threads
            for p_idx in range(num_positions):
                self._frame_buffer[p_idx] = {}
                self._queues[p_idx] = Queue()

            # Create threads but don't start them yet - we'll start them in flush()
            for p_idx, fname in enumerate(fnames):
                ome = dims_to_ome(tczyx_dims, dtype=dtype, tiff_file_name=fname)
                self._threads[p_idx] = WriterThread(
                    fname,
                    shape=self._tiff_shape,
                    dtype=dtype,
                    image_queue=self._queues[p_idx],
                    ome_xml=ome.to_xml(),
                )

        self._is_active = True
        return self

    def is_active(self) -> bool:
        """Return True if the stream is currently active."""
        return self._is_active

    def flush(self) -> None:
        """Flush all pending writes to the underlying TIFF files.

        In memmap mode: converts memmaps to final OME-TIFF files and cleans up.
        In RAM mode: starts writer threads to consume buffered frames in order.
        """
        if self._use_memmap:
            # Memmap mode: flush memmaps to OME-TIFF files
            for p_idx in list(self._memmaps.keys()):
                mm = self._memmaps[p_idx]
                # ensure memmap is flushed to disk
                mm.flush()

                fname = self._filenames[p_idx]
                # write memmap array directly; tifffile will read from the memmap
                self._tf.imwrite(
                    fname,
                    mm,
                    bigtiff=True,
                    ome=False,
                    resolutionunit=self._tf.RESUNIT.MICROMETER,
                    photometric=self._tf.PHOTOMETRIC.MINISBLACK,
                    description=self._position_ome_xml.get(p_idx, ""),
                )

                # remove underlying memmap file
                mp = self._memmap_paths.get(p_idx)
                # delete memmap object and file
                del self._memmaps[p_idx]
                if mp:
                    Path(mp).unlink(missing_ok=True)

            self._memmap_paths.clear()
        else:
            # RAM mode: write buffered frames in correct order via threads
            for p_idx, frame_dict in self._frame_buffer.items():
                # Start the writer thread for this position
                self._threads[p_idx].start()

                # Generate all possible indices in row-major order
                from itertools import product

                for storage_idx in product(*[range(s) for s in self._tiff_shape[:-2]]):
                    if storage_idx in frame_dict:
                        self._queues[p_idx].put(frame_dict[storage_idx])

                # Signal completion
                self._queues[p_idx].put(None)

            # Wait for threads to finish
            for thread in self._threads.values():
                thread.join(timeout=5)

            # Clear buffers
            self._frame_buffer.clear()

        self._is_active = False

    def update_ome_metadata(self, metadata: ome.OME) -> None:
        """Update the OME metadata in the TIFF files.

        The metadata argument MUST be an instance of ome_types.OME.

        This method should be called after flush() to update the OME-XML
        description in the already-written TIFF files with complete metadata.

        Parameters
        ----------
        metadata : OME
            The OME metadata object to write to the TIFF files.
        """
        if not isinstance(metadata, self._ome.OME):  # pragma: no cover
            raise TypeError(f"Expected OME metadata, got {type(metadata)}")

        if self._use_memmap:
            # Update metadata for each position file (memmap mode)
            for position_idx in range(len(self._filenames)):
                self._update_position_metadata(position_idx, metadata)
        else:
            # Update metadata for each position file (RAM mode)
            for position_idx in self._threads:
                self._update_position_metadata(position_idx, metadata)

    # -----------------------PRIVATE METHODS------------------------ #

    def _init_positions_tiff(
        self, dimensions: Sequence[Dimension]
    ) -> tuple[int, Sequence[Dimension]]:
        """Initialize positions with TIFF-specific dimension ordering.

        TIFF files must be stored in [t, c, z, y, x] order. This method builds
        an index mapping that respects the acquisition order (dimension sequence)
        while ensuring frames are queued in TIFF storage order.

        Parameters
        ----------
        dimensions : Sequence[Dimension]
            All dimensions including position, in acquisition order.

        Returns
        -------
        tuple[int, Sequence[Dimension]]
            Number of positions and dimensions in TIFF order [t, c, z, y, x].
        """
        from itertools import product

        # Separate position dimension from other dimensions
        position_dims = [d for d in dimensions if d.label == "p"]
        non_position_dims = [d for d in dimensions if d.label != "p"]
        num_positions = position_dims[0].size if position_dims else 1

        # Build dimension ranges (excluding x, y which are not iterated)
        non_spatial_dims = [d for d in non_position_dims if d.label not in "yx"]

        # Use the order of dimensions to determine acquisition order

        # Create a mapping from label to range
        dim_ranges = {d.label: range(d.size) for d in non_spatial_dims}

        # Build ranges in the order dimensions appear
        acq_ordered_ranges = []
        acq_ordered_labels = []
        for dim in dimensions:
            if dim.label == "p":
                acq_ordered_ranges.append(range(num_positions))
                acq_ordered_labels.append("p")
            elif dim.label in dim_ranges and dim.label not in "yx":
                acq_ordered_ranges.append(dim_ranges[dim.label])
                acq_ordered_labels.append(dim.label)

        # TIFF storage order (excluding position and spatial dims)
        tiff_storage_labels = [d for d in "tcz" if d in dim_ranges]

        # Create index mapping from acquisition order to TIFF storage order.
        # Store a storage-tuple (in TIFF order) so buffering code can use the
        # tuple directly as the key.
        self._indices = {}
        for i, acq_indices in enumerate(product(*acq_ordered_ranges)):
            acq_dict = dict(zip(acq_ordered_labels, acq_indices, strict=False))
            pos = acq_dict.get("p", 0)

            # Build storage tuple in TIFF order [t, c, z]
            storage_tuple: tuple[int, ...] = tuple(
                acq_dict[label] for label in tiff_storage_labels
            )

            self._indices[i] = (str(pos), storage_tuple)

        # Keep the TIFF storage label ordering handy for later conversion when
        # buffering frames (flush expects tuple keys in TIFF order)
        self._tiff_storage_labels = tiff_storage_labels

        self._position_dim = position_dims[0] if position_dims else None
        self._append_count = 0
        self._num_positions = num_positions
        self._non_position_dims = non_position_dims

        # Return dimensions in TIFF order [t, c, z, y, x]
        dim_map = {d.label: d for d in non_position_dims}
        tczyx_dims: list[Dimension] = []
        for label in "tczyx":
            if label in dim_map:
                tczyx_dims.append(dim_map[label])  # type: ignore

        return num_positions, tczyx_dims

    def _prepare_files(
        self, path: Path, num_positions: int, overwrite: bool
    ) -> list[str]:
        path_root = str(path)
        for possible_ext in [".ome.tiff", ".ome.tif", ".tiff", ".tif"]:
            if path_root.endswith(possible_ext):
                ext = possible_ext
                path_root = path_root[: -len(possible_ext)]
                break
        else:
            ext = path.suffix

        fnames = []
        for p_idx in range(num_positions):
            # only append position index if there are multiple positions
            if num_positions > 1:
                p_path = Path(f"{path_root}_p{p_idx:03d}{ext}")
            else:
                p_path = self._path

            # Check if file exists and handle overwrite logic
            if p_path.exists():
                if not overwrite:
                    raise FileExistsError(
                        f"File {p_path} already exists. "
                        "Use overwrite=True to overwrite it."
                    )
                p_path.unlink()

            # Ensure the parent directory exists
            p_path.parent.mkdir(parents=True, exist_ok=True)
            fnames.append(str(p_path))

        return fnames

    def _write_to_backend(
        self, array_key: str, index: tuple[int, ...], frame: np.ndarray
    ) -> None:
        """TIFF-specific write implementation.

        In memmap mode: writes frame directly to memmap at TIFF storage index.
        In RAM mode: buffers frames for reordering at flush time.
        """
        if self._use_memmap:
            # Memmap mode: write directly to disk-backed array
            p_idx = int(array_key)
            storage_idx = index
            mm = self._memmaps[p_idx]
            # place 2D frame into memmap at storage_idx
            storage_idx_tuple = tuple(int(i) for i in storage_idx)
            idx = (*storage_idx_tuple, slice(None), slice(None))
            mm[cast("tuple[Any, ...]", idx)] = frame

            # Periodic flush for durability without per-frame overhead
            self._frame_count += 1
            if self._frame_count % self._flush_interval == 0:
                for memmap in self._memmaps.values():
                    try:
                        memmap.flush()
                    except Exception:
                        pass
        else:
            # RAM mode: buffer frame in memory
            self._frame_buffer[int(array_key)][index] = frame

    def _update_position_metadata(self, position_idx: int, metadata: ome.OME) -> None:
        """Add OME metadata to TIFF file efficiently without rewriting image data."""
        if self._use_memmap:
            fname = self._filenames[position_idx]
        else:
            thread = self._threads[position_idx]
            fname = thread._path

        if not Path(fname).exists():  # pragma: no cover
            warnings.warn(
                f"TIFF file for position {position_idx} does not exist at "
                f"{fname}. Not writing metadata.",
                stacklevel=2,
            )
            return

        try:
            position_ome = _create_position_specific_ome(position_idx, metadata)
            # Create ASCII version for tifffile.tiffcomment since tifffile.tiffcomment
            # requires ASCII strings
            ascii_xml = position_ome.to_xml().replace("µ", "&#x00B5;").encode("ascii")
        except Exception as e:
            raise RuntimeError(
                f"Failed to create position-specific OME metadata for position "
                f"{position_idx}. {e}"
            ) from e

        try:
            # TODO:
            # consider a lock on the tiff file itself to prevent concurrent writes?
            self._tf.tiffcomment(fname, comment=ascii_xml)
        except Exception as e:
            raise RuntimeError(f"Failed to update OME metadata in {fname}") from e


class WriterThread(threading.Thread):
    def __init__(
        self,
        path: str,
        shape: tuple[int, ...],
        dtype: np.dtype,
        image_queue: Queue[np.ndarray | None],
        ome_xml: str = "",
        pixelsize: float = 1.0,
    ) -> None:
        super().__init__(daemon=True, name=f"TiffWriterThread-{next(thread_counter)}")
        self._path: str = path
        self._shape = shape
        self._dtype = dtype
        self._image_queue = image_queue
        self._res = 1 / pixelsize
        self._bytes_written = 0
        self._frames_written = 0
        self._ome_xml = ome_xml

    def run(self) -> None:
        # would be nice if we could just use `iter(queue, None)`...
        # but that doesn't work with numpy arrays which don't support __eq__
        import tifffile

        def _queue_iterator() -> Iterator[np.ndarray]:
            """Generator to yield frames from the queue."""
            while True:
                frame = self._image_queue.get()
                if frame is None:
                    break
                yield frame
                self._bytes_written += frame.nbytes
                self._frames_written += 1

        try:
            # Create TiffWriter and write the data
            # Since we're using tiffcomment for metadata updates,
            # we can close immediately
            with tifffile.TiffWriter(self._path, bigtiff=True, ome=False) as writer:
                writer.write(
                    _queue_iterator(),
                    shape=self._shape,
                    dtype=self._dtype,
                    resolution=(self._res, self._res),
                    resolutionunit=tifffile.RESUNIT.MICROMETER,
                    photometric=tifffile.PHOTOMETRIC.MINISBLACK,
                    description=self._ome_xml,
                )

        except Exception as e:
            # suppress an over-eager tifffile exception
            # when the number of bytes written is less than expected
            if "wrong number of bytes" in str(e):
                return
            raise


thread_counter = count()

# ------------------------

# helpers for position-specific OME metadata updates


def _create_position_specific_ome(position_idx: int, metadata: ome.OME) -> ome.OME:
    """Create OME metadata for a specific position from complete metadata.

    Extracts only the Image and related metadata for the given position index.
    Assumes Image IDs follow the pattern "Image:{position_idx}".
    """
    target_image_id = f"Image:{position_idx}"

    # Find an image by its ID in the given list of images
    # will raise StopIteration if not found (caller should catch error)
    position_image = next(img for img in metadata.images if img.id == target_image_id)
    position_plates = _extract_position_plates(metadata, target_image_id)

    return ome.OME(
        uuid=metadata.uuid,
        images=[position_image],
        instruments=metadata.instruments,
        plates=position_plates,
    )


def _extract_position_plates(ome: ome.OME, target_image_id: str) -> list[ome.Plate]:
    """Extract plate metadata for a specific image ID.

    Searches through plates to find the well sample referencing the target
    image ID and returns a plate containing only the relevant well and sample.
    """
    for plate in ome.plates:
        for well in plate.wells:
            if _well_contains_image(well, target_image_id):
                return [_create_position_plate(plate, well, target_image_id)]

    return []


def _well_contains_image(well: ome.Well, target_image_id: str) -> bool:
    """Check if a well contains a sample referencing the target image ID."""
    return any(
        sample.image_ref and sample.image_ref.id == target_image_id
        for sample in well.well_samples
    )


def _create_position_plate(
    original_plate: ome.Plate, well: ome.Well, target_image_id: str
) -> ome.Plate:
    """Create a new plate containing only the relevant well and sample."""
    # Find the specific well sample for this image
    target_sample = next(
        sample
        for sample in well.well_samples
        if sample.image_ref and sample.image_ref.id == target_image_id
    )

    # Create new plate with only the relevant well and sample
    plate_dict = original_plate.model_dump()
    well_dict = well.model_dump()
    well_dict["well_samples"] = [target_sample]
    plate_dict["wells"] = [well_dict]
    return ome.Plate.model_validate(plate_dict)
