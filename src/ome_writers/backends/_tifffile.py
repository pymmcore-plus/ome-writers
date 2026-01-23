"""OME-TIFF backend using tifffile for sequential writes."""

from __future__ import annotations

import math
import threading
import uuid
import warnings
from datetime import datetime, timezone
from itertools import count
from pathlib import Path
from queue import Queue
from typing import TYPE_CHECKING, Literal

from ome_writers.backends._backend import ArrayBackend

if TYPE_CHECKING:
    from collections.abc import Iterator

    import numpy as np

    from ome_writers._router import FrameRouter
    from ome_writers.schema import AcquisitionSettings, Dimension


try:
    import ome_types.model as ome
    import tifffile
except ImportError as e:
    raise ImportError(
        f"{__name__} requires tifffile and ome-types: "
        "`pip install ome-writers[tifffile]`."
    ) from e


class TiffBackend(ArrayBackend):
    """OME-TIFF backend using tifffile for sequential writes.

    TIFF files are written sequentially, with one file per position.
    The index parameter in write() is ignored since TIFF only supports
    sequential writing.
    """

    def __init__(self) -> None:
        self._threads: dict[int, WriterThread] = {}
        self._queues: dict[int, Queue[np.ndarray | None]] = {}
        self._file_paths: dict[int, str] = {}  # Store paths for metadata updates
        self._file_uuids: dict[int, str] = {}  # Store UUIDs for each file
        self._finalized = False
        self._storage_dims: tuple[Dimension, ...] | None = None
        self._cached_metadata: ome.OME | None = None  # Cache for get_metadata()

    def is_incompatible(self, settings: AcquisitionSettings) -> Literal[False] | str:
        """Check if settings are compatible with TIFF backend."""
        path = settings.root_path.lower()
        valid_extensions = [".tif", ".tiff", ".ome.tif", ".ome.tiff"]
        if not any(path.endswith(ext) for ext in valid_extensions):  # pragma: no cover
            return (
                "Root path must end with .tif, .tiff, .ome.tif, or .ome.tiff "
                "for TiffBackend."
            )
        return False

    def prepare(self, settings: AcquisitionSettings, router: FrameRouter) -> None:
        """Initialize OME-TIFF files and writer threads."""
        self._finalized = False
        root = Path(settings.root_path).expanduser().resolve()
        positions = settings.positions
        self._storage_dims = storage_dims = settings.array_storage_dimensions
        self._dtype = settings.dtype

        # Compute shape from storage dimensions
        shape = tuple(d.count if d.count is not None else 1 for d in storage_dims)
        dtype = settings.dtype

        # Check if any dimension is unbounded
        has_unbounded = any(d.count is None for d in storage_dims)

        # Prepare file paths
        fnames = self._prepare_files(root, len(positions), settings.overwrite)

        # Generate UUIDs and OME metadata for all positions
        all_images, num_pos = [], len(positions)
        for p_idx, fname in enumerate(fnames):
            self._file_uuids[p_idx] = file_uuid = str(uuid.uuid4())
            all_images.append(
                _create_ome_image(
                    storage_dims, dtype, Path(fname).name, p_idx, file_uuid, num_pos
                )
            )
        self._cached_metadata = ome.OME(images=all_images)

        # Create writer thread for each position
        for p_idx, fname in enumerate(fnames):
            if num_pos > 1:  # multiposition: write complete OME with all images
                ome_xml = self._cached_metadata.to_xml()
            else:
                ome_xml = ome.OME(images=[all_images[p_idx]]).to_xml()
            ome_xml = ome_xml.replace("µ", "&#x00B5;")  # ASCII compatibility

            self._file_paths[p_idx] = fname
            self._queues[p_idx] = q = Queue()
            self._threads[p_idx] = thread = WriterThread(
                path=fname,
                shape=shape,
                dtype=dtype,
                image_queue=q,
                ome_xml=ome_xml,
                has_unbounded=has_unbounded,
            )
            thread.start()

    def write(
        self,
        position_index: int,
        index: tuple[int, ...],
        frame: np.ndarray,
    ) -> None:
        """Write frame sequentially to the appropriate position's TIFF file.

        The index parameter is ignored since TIFF writes are sequential.
        """
        if self._finalized:  # pragma: no cover
            raise RuntimeError("Cannot write after finalize().")
        if not self._threads:  # pragma: no cover
            raise RuntimeError("Backend not prepared. Call prepare() first.")

        self._queues[position_index].put(frame)

    def finalize(self) -> None:
        """Flush and close all TIFF writers."""
        if not self._finalized:
            # Signal threads to stop
            for queue in self._queues.values():
                queue.put(None)

            # Wait for threads to finish
            for thread in self._threads.values():
                thread.join(timeout=5)

            # Update OME metadata if unbounded dimensions were written
            if self._storage_dims and any(d.count is None for d in self._storage_dims):
                self._update_unbounded_metadata()

            self._threads.clear()
            self._queues.clear()
            self._finalized = True

    def get_metadata(self) -> ome.OME | None:
        """Get the base OME metadata generated from acquisition settings.

        Returns the OME metadata object that was auto-generated during prepare()
        based on dimensions, dtype, and file paths. This object contains all
        positions as Image objects with IDs "Image:0", "Image:1", etc.

        Users can modify this object to add meaningful names, timestamps, etc.,
        and pass it to update_metadata().

        Returns
        -------
        ome_types.model.OME or None
            Complete OME metadata object with all positions, or None if prepare()
            has not been called yet.
        """
        if self._cached_metadata is None:  # pragma: no cover
            return None
        return self._cached_metadata.model_copy(deep=True)

    def update_metadata(self, metadata: ome.OME) -> None:
        """Update the OME metadata in the TIFF files.

        The metadata argument MUST be an instance of ome_types.OME.

        This method must be called AFTER exiting the stream context (after
        finalize() completes), as TIFF files must be closed before metadata
        can be updated.

        Parameters
        ----------
        metadata : ome_types.model.OME
            Complete OME metadata object to write to the TIFF files.

        Raises
        ------
        TypeError
            If metadata is not an ome_types.model.OME instance.
        RuntimeError
            If called before finalize() completes, or if metadata update fails.
        """
        if not self._finalized:  # pragma: no cover
            raise RuntimeError(
                "update_metadata() must be called after the stream context exits. "
                "TIFF files must be closed before metadata can be updated."
            )

        if not isinstance(metadata, ome.OME):
            raise TypeError(
                f"Expected ome_types.model.OME metadata, got {type(metadata)}"
            )

        for position_idx in self._file_paths:
            self._update_position_metadata(position_idx, metadata)

        # Update cache to reflect what's now on disk
        self._cached_metadata = metadata.model_copy(deep=True)

    # -------------------

    def _update_unbounded_metadata(self) -> None:
        """Update OME metadata after writing unbounded dimensions.

        For unbounded dimensions, we write frames without knowing the final count.
        After writing completes, update the OME-XML with the actual frame counts.
        """
        if not self._storage_dims:  # pragma: no cover
            return

        # Get actual frame count from first position's thread
        # (all positions should have written the same number of frames)
        if 0 not in self._threads:  # pragma: no cover
            return

        total_frames_written = self._threads[0].frames_written
        if total_frames_written == 0:  # pragma: no cover
            return

        # Calculate the count for the unbounded dimension
        # total_frames = product of all dimension counts (excluding spatial dims Y,X)
        # For unbounded dim: unbounded_count = total_frames / product(other_dims)
        # Product of all known (non-None) dimension counts except Y,X
        known_product = math.prod(
            d.count for d in self._storage_dims[:-2] if d.count is not None
        )

        # Calculate unbounded dimension count
        unbounded_count = (
            total_frames_written // known_product
            if known_product > 0
            else total_frames_written
        )

        # Create corrected dimensions with actual counts
        corrected_dims = tuple(
            d.model_copy(update={"count": unbounded_count}) if d.count is None else d
            for d in self._storage_dims
        )

        # Regenerate OME metadata for each position
        all_images, num_pos = [], len(self._file_paths)
        for p_idx, fname in sorted(self._file_paths.items()):
            filename, file_uuid = Path(fname).name, self._file_uuids.get(p_idx)
            all_images.append(
                _create_ome_image(
                    corrected_dims, self._dtype, filename, p_idx, file_uuid, num_pos
                )
            )

        # Update cached metadata with corrected dimensions
        self._cached_metadata = ome.OME(images=all_images)

        # Write metadata to each file
        for p_idx, fname in sorted(self._file_paths.items()):
            try:
                if num_pos > 1:  # multiposition: full OME with all images
                    xml = self._cached_metadata.to_xml()
                else:
                    xml = ome.OME(images=[all_images[p_idx]]).to_xml()
                xml = xml.replace("µ", "&#x00B5;")
                tifffile.tiffcomment(fname, comment=xml.encode("ascii"))
            except Exception as e:  # pragma: no cover
                warnings.warn(
                    f"Failed to update OME metadata in {fname}: {e}", stacklevel=2
                )

    def _update_position_metadata(self, position_idx: int, metadata: ome.OME) -> None:
        """Update OME metadata for a specific position's TIFF file."""
        file_path = self._file_paths[position_idx]

        if not Path(file_path).exists():  # pragma: no cover
            warnings.warn(
                f"TIFF file for position {position_idx} does not exist at "
                f"{file_path}. Not writing metadata.",
                stacklevel=3,
            )
            return

        try:
            # Extract position-specific metadata from complete OME
            position_ome = _create_position_specific_ome(position_idx, metadata)

            # Create ASCII version for tifffile.tiffcomment
            # tifffile.tiffcomment requires ASCII strings
            ascii_xml = position_ome.to_xml().replace("µ", "&#x00B5;").encode("ascii")
        except Exception as e:  # pragma: no cover
            raise RuntimeError(
                f"Failed to create position-specific OME metadata for position "
                f"{position_idx}: {e}"
            ) from e

        try:
            tifffile.tiffcomment(file_path, comment=ascii_xml)
        except Exception as e:  # pragma: no cover
            raise RuntimeError(
                f"Failed to update OME metadata in {file_path}: {e}"
            ) from e

    def _prepare_files(
        self, path: Path, num_positions: int, overwrite: bool
    ) -> list[str]:
        """Prepare file paths for each position."""
        path_str = str(path)

        # Strip known extensions
        ext = path_root = None
        for possible_ext in [".ome.tiff", ".ome.tif", ".tiff", ".tif"]:
            if path_str.endswith(possible_ext):
                ext = possible_ext
                path_root = path_str[: -len(possible_ext)]
                break

        if ext is None:
            # No recognized extension, default to .ome.tiff
            path_root = path_str
            ext = ".ome.tiff"

        fnames = []
        for p_idx in range(num_positions):
            # Append position index only if multiple positions
            if num_positions > 1:
                p_path = Path(f"{path_root}_p{p_idx:03d}{ext}")
            else:
                p_path = path

            # Handle overwrite
            if p_path.exists():
                if not overwrite:
                    raise FileExistsError(
                        f"File {p_path} already exists. "
                        "Use overwrite=True to overwrite it."
                    )
                p_path.unlink()

            # Ensure parent directory exists
            p_path.parent.mkdir(parents=True, exist_ok=True)
            fnames.append(str(p_path))

        return fnames


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
    ) -> None:
        super().__init__(daemon=True, name=f"TiffWriterThread-{next(_thread_counter)}")
        self._path = path
        self._shape = shape
        self._dtype = dtype
        self._image_queue = image_queue
        self._ome_xml = ome_xml
        self._res = 1 / pixelsize
        self._has_unbounded = has_unbounded
        self.frames_written = 0  # Track actual frames written for unbounded dims

    def run(self) -> None:
        """Write frames from queue to TIFF file sequentially."""

        def _queue_iterator() -> Iterator[np.ndarray]:
            """Yield frames from the queue until None is received."""
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
                if self._has_unbounded:
                    # For unbounded dimensions, write frames individually
                    # to avoid shape mismatch errors
                    for i, frame in enumerate(_queue_iterator()):
                        writer.write(
                            frame,
                            contiguous=True,
                            dtype=self._dtype,
                            resolution=(self._res, self._res),
                            resolutionunit=tifffile.RESUNIT.MICROMETER,
                            photometric=tifffile.PHOTOMETRIC.MINISBLACK,
                            description=self._ome_xml if i == 0 else None,
                        )
                else:
                    # For bounded dimensions, use iterator with shape
                    writer.write(
                        _queue_iterator(),
                        shape=self._shape,
                        dtype=self._dtype,
                        resolution=(self._res, self._res),
                        resolutionunit=tifffile.RESUNIT.MICROMETER,
                        photometric=tifffile.PHOTOMETRIC.MINISBLACK,
                        description=self._ome_xml,
                    )
        except Exception as e:  # pragma: no cover
            # Suppress over-eager tifffile exception for incomplete writes
            if "wrong number of bytes" in str(e):
                return
            raise


_thread_counter = count()

# ------------------------

# helpers for position-specific OME metadata updates


def _create_ome_image(
    dims: tuple[Dimension, ...],
    dtype: str,
    filename: str,
    image_index: int,
    file_uuid: str | None = None,
    num_positions: int = 1,
) -> ome.Image:
    """Generate OME Image object for TIFF file."""
    shape_dict = {d.name.upper(): d.count or 1 for d in dims}
    size_t = shape_dict.get("T", 1)
    size_c = shape_dict.get("C", 1)
    size_z = shape_dict.get("Z", 1)
    dim_order = next(
        order
        for order in ome.Pixels_DimensionOrder
        if order.value.startswith("".join(reversed(shape_dict.keys())))
    )
    dims_by_name = {d.name.lower(): d for d in dims}

    # Build TiffData: simple for single position, detailed with UUIDs for multi
    if num_positions == 1:
        tiff_data_blocks = [ome.TiffData(plane_count=size_t * size_c * size_z)]
    else:
        tiff_data_blocks, ifd = [], 0
        for t in range(size_t):
            for z in range(size_z):
                for c in range(size_c):
                    td = ome.TiffData(
                        ifd=ifd, plane_count=1, first_c=c, first_z=z, first_t=t
                    )
                    if file_uuid:
                        td.uuid = ome.TiffData.UUID(
                            value=f"urn:uuid:{file_uuid}", file_name=filename
                        )
                    tiff_data_blocks.append(td)
                    ifd += 1

    pixels = ome.Pixels(
        id=f"Pixels:{image_index}",
        dimension_order=dim_order,
        size_t=size_t,
        size_c=size_c,
        size_z=size_z,
        size_y=shape_dict.get("Y", 1),
        size_x=shape_dict.get("X", 1),
        type=dtype,
        big_endian=False,
        channels=[
            ome.Channel(id=f"Channel:{image_index}:{i}", samples_per_pixel=1)
            for i in range(size_c)
        ],
        tiff_data_blocks=tiff_data_blocks,
    )
    # Add physical sizes if available
    for axis in ["x", "y", "z"]:
        if (dim := dims_by_name.get(axis)) and dim.scale is not None:
            setattr(pixels, f"physical_size_{axis}", dim.scale)
            if dim.unit:
                try:
                    setattr(
                        pixels, f"physical_size_{axis}_unit", ome.UnitsLength(dim.unit)
                    )
                except ValueError:
                    pass

    name = Path(filename).stem.removesuffix(".ome")
    return ome.Image(
        id=f"Image:{image_index}",
        acquisition_date=datetime.now(timezone.utc),
        pixels=pixels,
        name=name,
    )


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
