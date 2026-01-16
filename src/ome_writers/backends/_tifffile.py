"""OME-TIFF backend using tifffile for sequential writes."""

from __future__ import annotations

import threading
import warnings
from itertools import count
from pathlib import Path
from queue import Queue
from typing import TYPE_CHECKING, Literal

from ome_writers._backend import ArrayBackend

if TYPE_CHECKING:
    from collections.abc import Iterator

    import numpy as np

    from ome_writers._router import FrameRouter, PositionInfo
    from ome_writers.schema import AcquisitionSettings, Dimension


try:
    import ome_types.model as ome
    import tifffile
    from ome_types.model import PixelType
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
        self._finalized = False

    def is_incompatible(self, settings: AcquisitionSettings) -> Literal[False] | str:
        """Check if settings are compatible with TIFF backend."""
        path = settings.root_path.lower()
        valid_extensions = [".tif", ".tiff", ".ome.tif", ".ome.tiff"]
        if not any(path.endswith(ext) for ext in valid_extensions):
            return (
                "Root path must end with .tif, .tiff, .ome.tif, or .ome.tiff "
                "for TiffBackend."
            )
        return False

    def prepare(self, settings: AcquisitionSettings, router: FrameRouter) -> None:
        """Initialize OME-TIFF files and writer threads."""
        self._finalized = False
        root = Path(settings.root_path).expanduser().resolve()
        positions = router.positions
        storage_dims = router.array_storage_dimensions

        # Compute shape from storage dimensions
        shape = tuple(d.count if d.count is not None else 1 for d in storage_dims)
        dtype = settings.dtype

        # Prepare file paths
        fnames = self._prepare_files(root, len(positions), settings.overwrite)

        # Create writer thread for each position
        for p_idx, fname in enumerate(fnames):
            # Generate OME metadata
            ome_xml = self._generate_ome_xml(storage_dims, dtype, Path(fname).name)

            self._file_paths[p_idx] = fname
            self._queues[p_idx] = q = Queue()
            self._threads[p_idx] = thread = WriterThread(
                path=fname,
                shape=shape,
                dtype=dtype,
                image_queue=q,
                ome_xml=ome_xml,
            )
            thread.start()

    def write(
        self,
        position_info: PositionInfo,
        index: tuple[int, ...],
        frame: np.ndarray,
    ) -> None:
        """Write frame sequentially to the appropriate position's TIFF file.

        The index parameter is ignored since TIFF writes are sequential.
        """
        if self._finalized:
            raise RuntimeError("Cannot write after finalize().")
        if not self._threads:
            raise RuntimeError("Backend not prepared. Call prepare() first.")

        position_idx = position_info[0]
        self._queues[position_idx].put(frame)

    def finalize(self) -> None:
        """Flush and close all TIFF writers."""
        if not self._finalized:
            # Signal threads to stop
            for queue in self._queues.values():
                queue.put(None)

            # Wait for threads to finish
            for thread in self._threads.values():
                thread.join(timeout=5)

            self._threads.clear()
            self._queues.clear()
            self._finalized = True

    def update_metadata(self, metadata: ome.OME) -> None:
        """Update the OME metadata in the TIFF files.

        The metadata argument MUST be an instance of ome_types.OME.

        This method should be called after flush() to update the OME-XML
        description in the already-written TIFF files with complete metadata.

        Parameters
        ----------
        metadata : ome_types.model.OME
            Complete OME metadata object to write to the TIFF files.

        Raises
        ------
        TypeError
            If metadata is not an ome_types.model.OME instance.
        RuntimeError
            If metadata generation or file update fails.
        """
        if not isinstance(metadata, ome.OME):
            raise TypeError(
                f"Expected ome_types.model.OME metadata, got {type(metadata)}"
            )

        for position_idx in self._threads:
            self._update_position_metadata(position_idx, metadata)

    # -------------------

    def _update_position_metadata(self, position_idx: int, metadata: ome.OME) -> None:
        """Update OME metadata for a specific position's TIFF file."""
        file_path = self._file_paths[position_idx]

        if not Path(file_path).exists():
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
        except Exception as e:
            raise RuntimeError(
                f"Failed to create position-specific OME metadata for position "
                f"{position_idx}: {e}"
            ) from e

        try:
            tifffile.tiffcomment(file_path, comment=ascii_xml)
        except Exception as e:
            raise RuntimeError(
                f"Failed to update OME metadata in {file_path}: {e}"
            ) from e

    def _prepare_files(
        self, path: Path, num_positions: int, overwrite: bool
    ) -> list[str]:
        """Prepare file paths for each position."""
        path_str = str(path)

        # Strip known extensions
        ext = None
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

    def _generate_ome_xml(
        self, dims: tuple[Dimension, ...], dtype: str, filename: str
    ) -> str:
        """Generate OME-XML metadata for TIFF file.

        Creates basic OME metadata structure. The dimension order is determined
        by the storage dimensions order.
        """
        # Build shape dictionary from dimensions
        shape_dict: dict[str, int] = {}
        for d in dims:
            shape_dict[d.name.upper()] = d.count if d.count is not None else 1

        # OME-TIFF requires exactly 5 dimensions in a specific order
        # Default missing dimensions to size 1
        size_t = shape_dict.get("T", 1)
        size_c = shape_dict.get("C", 1)
        size_z = shape_dict.get("Z", 1)
        size_y = shape_dict.get("Y", 1)
        size_x = shape_dict.get("X", 1)

        # Compute dimension order from storage dimensions
        # OME dimension order describes how planes are ordered in the file
        # Valid orders: XYZCT, XYZTC, XYCTZ, XYCZT, XYTCZ, XYTZC
        # (all start with XY since those are the spatial in-plane dimensions)

        # Extract non-spatial dimension names from storage order
        non_spatial_names = [d.name.upper() for d in dims[:-2]]

        # Build OME dimension order: XY + remaining dimensions in storage order
        # Filter to only include Z, C, T dimensions
        remaining_dims = [d for d in non_spatial_names if d in "ZCT"]

        # Ensure we have Z, C, T (add missing ones at the end with size 1)
        for dim_char in "ZCT":
            if dim_char not in remaining_dims:
                remaining_dims.append(dim_char)

        # Build final dimension order string (XY + remaining)
        final_dim_order = "XY" + "".join(remaining_dims[:3])  # Only use first 3

        # Map dtype string to OME pixel type
        dtype_map = {
            "uint8": PixelType.UINT8,
            "uint16": PixelType.UINT16,
            "uint32": PixelType.UINT32,
            "int8": PixelType.INT8,
            "int16": PixelType.INT16,
            "int32": PixelType.INT32,
            "float": PixelType.FLOAT,
            "double": PixelType.DOUBLE,
        }
        pixel_type = dtype_map.get(dtype, PixelType.UINT16)

        # Create OME metadata
        pixels = ome.Pixels(
            id="Pixels:0",
            dimension_order=final_dim_order,  # String is auto-converted to enum
            size_x=size_x,
            size_y=size_y,
            size_z=size_z,
            size_c=size_c,
            size_t=size_t,
            type=pixel_type,
            big_endian=False,
            channels=[ome.Channel(id=f"Channel:0:{i}") for i in range(size_c)],
            tiff_data_blocks=[
                ome.TiffData(
                    plane_count=size_t * size_c * size_z,
                )
            ],
        )

        image = ome.Image(
            id="Image:0",
            pixels=pixels,
            name=Path(filename).stem,
        )

        ome_obj = ome.OME(
            images=[image],
        )

        return ome_obj.to_xml()


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
    ) -> None:
        super().__init__(daemon=True, name=f"TiffWriterThread-{next(_thread_counter)}")
        self._path = path
        self._shape = shape
        self._dtype = dtype
        self._image_queue = image_queue
        self._ome_xml = ome_xml
        self._res = 1 / pixelsize

    def run(self) -> None:
        """Write frames from queue to TIFF file sequentially."""

        def _queue_iterator() -> Iterator[np.ndarray]:
            """Yield frames from the queue until None is received."""
            while True:
                frame = self._image_queue.get()
                if frame is None:
                    break
                yield frame

        try:
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
            # Suppress over-eager tifffile exception for incomplete writes
            if "wrong number of bytes" in str(e):
                return
            raise


_thread_counter = count()

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
