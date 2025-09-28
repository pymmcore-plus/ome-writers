from __future__ import annotations

import importlib
import importlib.util
import threading
from itertools import count
from pathlib import Path
from queue import Queue
from typing import TYPE_CHECKING

from typing_extensions import Self

from ome_writers._dimensions import dims_to_ome
from ome_writers._stream_base import MultiPositionOMEStream

if TYPE_CHECKING:
    from collections.abc import Iterator, Sequence

    import numpy as np
    from ome_types import OME
    from ome_types.model import Image, Plate, Well

    from ome_writers._dimensions import Dimension


class TifffileStream(MultiPositionOMEStream):
    """A concrete OMEStream implementation for writing to OME-TIFF files.

    This writer is designed for deterministic acquisitions where the full experiment
    shape is known ahead of time. It works by creating all necessary OME-TIFF
    files at the start of the acquisition and using memory-mapped arrays for
    efficient, sequential writing of incoming frames.

    If a 'p' (position) dimension is included in the dimensions, a separate
    OME-TIFF file will be created for each position.

    Attributes
    ----------
    _writers : Dict[int, np.memmap]
        A dictionary mapping position index to its numpy memmap array.
    """

    @classmethod
    def is_available(cls) -> bool:
        """Check if the tifffile package is available."""
        return importlib.util.find_spec("tifffile") is not None

    def __init__(self) -> None:
        super().__init__()
        try:
            import tifffile  # noqa: F401
        except ImportError as e:
            msg = "TifffileStream requires tifffile: `pip install tifffile`."
            raise ImportError(msg) from e

        try:
            import ome_types  # noqa: F401
        except ImportError as e:
            msg = "TifffileStream requires ome-types: `pip install ome-types`."
            raise ImportError(msg) from e

        # Using dictionaries to handle multi-position ('p') acquisitions
        self._threads: dict[int, WriterThread] = {}
        self._queues: dict[int, Queue[np.ndarray | None]] = {}
        self._is_active = False

    # ------------------------PUBLIC METHODS------------------------ #

    def create(
        self,
        path: str,
        dtype: np.dtype,
        dimensions: Sequence[Dimension],
        *,
        overwrite: bool = False,
    ) -> Self:
        # Use MultiPositionOMEStream to handle position logic
        num_positions, tczyx_dims = self._init_positions(dimensions)
        self._delete_existing = overwrite
        self._path = Path(self._normalize_path(path))
        shape_5d = tuple(d.size for d in tczyx_dims)

        fnames = self._prepare_files(self._path, num_positions, overwrite)

        # Create a memmap for each position
        for p_idx, fname in enumerate(fnames):
            ome = dims_to_ome(tczyx_dims, dtype=dtype, tiff_file_name=fname)
            self._queues[p_idx] = q = Queue()  # type: ignore
            self._threads[p_idx] = thread = WriterThread(
                fname,
                shape=shape_5d,
                dtype=dtype,
                image_queue=q,
                ome_xml=ome.to_xml(),
            )
            thread.start()

        self._is_active = True
        return self

    def is_active(self) -> bool:
        """Return True if the stream is currently active."""
        return self._is_active

    def flush(self) -> None:
        """Flush all pending writes to the underlying TIFF files."""
        # Signal the threads to stop by putting None in each queue
        for queue in self._queues.values():
            queue.put(None)

        # Wait for the thread to finish
        for thread in self._threads.values():
            thread.join(timeout=5)

        # Mark as inactive after flushing - this is consistent with other backends
        self._is_active = False

    def update_metadata(self, metadata: dict) -> None:
        """Update the OME metadata in the TIFF files.

        The dict passed in should be a valid OME structure as a dictionary.

        This method should be called after flush() to update the OME-XML
        description in the already-written TIFF files with complete metadata.
        """
        from ome_types import OME

        try:
            ome_metadata = OME.model_validate(metadata)
        except Exception as e:
            raise UserWarning(f"Failed to validate OME metadata: {e}") from e

        if len(self._threads) == 1:
            self._update_single_file_metadata(0, ome_metadata)
        else:
            self._update_multifile_metadata(ome_metadata)

    # -----------------------PRIVATE METHODS------------------------ #

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
        """TIFF-specific write implementation."""
        self._queues[int(array_key)].put(frame)

    def _update_single_file_metadata(self, position_idx: int, ome: OME) -> None:
        """Add OME metadata to TIFF file efficiently without rewriting image data."""
        import tifffile

        thread = self._threads[position_idx]

        # Create ASCII version for tifffile.tiffcomment since tifffile.tiffcomment
        # requires ASCII strings
        ascii_xml = ome.to_xml().replace("µ", "&#x00B5;").encode("ascii")

        try:
            tifffile.tiffcomment(thread._path, comment=ascii_xml)
        except Exception as e:
            raise RuntimeError(
                f"Failed to update OME metadata in {thread._path}"
            ) from e

    def _update_multifile_metadata(self, ome: OME) -> None:
        """Update metadata for multiple TIFF files in multi-position experiments.

        Each file gets only the OME metadata relevant to its position.

        The ome argument contains the complete metadata for all positions (Images)
        so we need to create separate OME metadata for each file (position) which means
        to extract only the relevant Image and related metadata for each position.
        """
        for position_idx in self._threads:
            position_ome = self._create_position_specific_ome(ome, position_idx)
            if position_ome is not None:
                self._update_single_file_metadata(position_idx, position_ome)

    def _create_position_specific_ome(self, ome: OME, position_idx: int) -> OME | None:
        """Create OME metadata for a specific position from complete metadata.

        Extracts only the Image and related metadata for the given position index.
        Assumes Image IDs follow the pattern "Image:{position_idx}".
        """
        from ome_types import OME

        target_image_id = f"Image:{position_idx}"
        position_image = self._find_image_by_id(ome.images, target_image_id)

        if position_image is None:
            return None

        position_plate = self._extract_position_plate(ome, target_image_id)

        return OME(
            uuid=ome.uuid,
            images=[position_image],
            instruments=ome.instruments,
            plates=[position_plate] if position_plate else [],
        )

    def _find_image_by_id(self, images: list[Image], target_id: str) -> Image | None:
        """Find an image by its ID in the given list of images."""
        for image in images:
            if image.id == target_id:
                return image
        return None

    def _extract_position_plate(self, ome: OME, target_image_id: str) -> Plate | None:
        """Extract plate metadata for a specific image ID.

        Searches through plates to find the well sample referencing the target
        image ID and returns a plate containing only the relevant well and sample.
        """
        if not ome.plates:
            return None

        for plate in ome.plates:
            matching_well = self._find_well_with_image(plate.wells, target_image_id)
            if matching_well is not None:
                return self._create_position_plate(
                    plate, matching_well, target_image_id
                )

        return None

    def _find_well_with_image(
        self, wells: list[Well], target_image_id: str
    ) -> Well | None:
        """Find the well containing a sample that references the target image ID."""
        for well in wells:
            if self._well_contains_image(well, target_image_id):
                return well
        return None

    def _well_contains_image(self, well: Well, target_image_id: str) -> bool:
        """Check if a well contains a sample referencing the target image ID."""
        return any(
            sample.image_ref and sample.image_ref.id == target_image_id
            for sample in well.well_samples
        )

    def _create_position_plate(
        self, original_plate: Plate, well: Well, target_image_id: str
    ) -> Plate:
        """Create a new plate containing only the relevant well and sample."""
        from ome_types.model import Plate

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

        return Plate.model_validate(plate_dict)


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
        self._path = path
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
