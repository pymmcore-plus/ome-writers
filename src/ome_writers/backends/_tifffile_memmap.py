from __future__ import annotations

import importlib
import importlib.util
import warnings
from contextlib import suppress
from itertools import count
from pathlib import Path
from typing import TYPE_CHECKING, Any

from typing_extensions import Self

from ome_writers._dimensions import dims_to_ome
from ome_writers._stream_base import MultiPositionOMEStream

if TYPE_CHECKING:
    from collections.abc import Sequence

    import numpy as np
    import ome_types.model as ome

    from ome_writers._dimensions import Dimension

else:
    with suppress(ImportError):
        import ome_types.model as ome


class TifffileStream(MultiPositionOMEStream):
    """A concrete OMEStream implementation for writing to OME-TIFF files using memmap.

    This writer uses numpy.memmap for memory-efficient writing. Frames are written
    directly to memory-mapped temporary files during acquisition, then converted to
    final OME-TIFF files on flush(). This approach minimizes RAM usage while
    maintaining good write performance.

    It is designed for deterministic acquisitions where the full experiment shape is
    known ahead of time.

    If a 'p' (position) dimension is included in the dimensions, a separate
    OME-TIFF file will be created for each position.

    Parameters
    ----------
    flush_interval : int, optional
        Number of frames to acquire before flushing memmaps to disk. Default is 100.

    Attributes
    ----------
    _writers : Dict[int, MemmapWriter]
        A dictionary mapping position index to its memmap writer.
    """

    @classmethod
    def is_available(cls) -> bool:
        """Check if the tifffile package is available."""
        return bool(
            importlib.util.find_spec("tifffile") is not None
            and importlib.util.find_spec("ome_types") is not None
        )

    def __init__(self, flush_interval: int = 100) -> None:
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
        # Using dictionaries to handle multi-position ('p') acquisitions
        self._writers: dict[int, MemmapWriter] = {}
        self._is_active = False
        self._flush_interval = flush_interval

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

        # Create a memmap writer for each position
        for p_idx, fname in enumerate(fnames):
            # remove extension for name to use in OME metadata
            p_root, _ = self._get_path_root_and_extension(fname)
            ome = dims_to_ome(tczyx_dims, dtype=dtype, tiff_file_name=Path(p_root).name)
            self._writers[p_idx] = MemmapWriter(
                fname,
                shape=shape_5d,
                dtype=dtype,
                ome_xml=ome.to_xml(),
                flush_interval=self._flush_interval,
            )

        self._is_active = True
        return self

    def is_active(self) -> bool:
        """Return True if the stream is currently active."""
        return self._is_active

    def flush(self) -> None:
        """Flush all pending writes to the underlying TIFF files."""
        # Flush all memmap writers to disk and convert to OME-TIFF
        for writer in self._writers.values():
            writer.flush_to_tiff(self._tf)

        # Mark as inactive after flushing - this is consistent with other backends
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

        for position_idx in self._writers:
            self._update_position_metadata(position_idx, metadata)

    # -----------------------PRIVATE METHODS------------------------ #

    def _prepare_files(
        self, path: Path, num_positions: int, overwrite: bool
    ) -> list[str]:
        path_root, ext = self._get_path_root_and_extension(path)
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

    def _get_path_root_and_extension(self, path: Path | str) -> tuple[str, str]:
        """Split the path into root and extension for .(ome.)tif and .(ome.)tiff files.

        Returns
        -------
        tuple[str, str]
            A tuple containing the root path (without extension) and the extension.
        """
        path_root = str(path)
        for possible_ext in [".ome.tiff", ".ome.tif", ".tiff", ".tif"]:
            if path_root.endswith(possible_ext):
                return path_root[: -len(possible_ext)], possible_ext
        return path_root, Path(path).suffix

    def _write_to_backend(
        self, array_key: str, index: tuple[int, ...], frame: np.ndarray
    ) -> None:
        """TIFF-specific write implementation using memmap."""
        self._writers[int(array_key)].write_frame(index, frame)

    def _update_position_metadata(self, position_idx: int, metadata: ome.OME) -> None:
        """Add OME metadata to TIFF file efficiently without rewriting image data."""
        writer = self._writers[position_idx]
        if not Path(writer._path).exists():  # pragma: no cover
            warnings.warn(
                f"TIFF file for position {position_idx} does not exist at "
                f"{writer._path}. Not writing metadata.",
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
            self._tf.tiffcomment(writer._path, comment=ascii_xml)
        except Exception as e:
            raise RuntimeError(
                f"Failed to update OME metadata in {writer._path}"
            ) from e


class MemmapWriter:
    """Memmap-based writer for a single OME-TIFF position.

    Writes frames directly to a memory-mapped temporary file, then converts
    to final OME-TIFF format on flush. This approach minimizes RAM usage.
    """

    def __init__(
        self,
        path: str,
        shape: tuple[int, ...],
        dtype: np.dtype,
        ome_xml: str = "",
        flush_interval: int = 100,
        pixelsize: float = 1.0,
    ) -> None:
        import numpy as np

        self._path: str = path
        self._shape = shape
        self._dtype = dtype
        self._ome_xml = ome_xml
        self._flush_interval = flush_interval
        self._res = 1 / pixelsize
        self._frame_count = 0

        # Create temporary memmap file
        self._memmap_path = Path(f"{path}.memmap.{next(memmap_counter)}")
        self._memmap = np.memmap(
            str(self._memmap_path),
            dtype=self._dtype,
            mode="w+",
            shape=self._shape,
            order="C",
        )

    def write_frame(self, index: tuple[int, ...], frame: np.ndarray) -> None:
        """Write a single frame to the memmap at the specified index."""
        # place 2D frame into memmap at storage_idx
        idx = (*index, slice(None), slice(None))
        self._memmap[idx] = frame  # type: ignore

        # Periodic flush for durability without per-frame overhead
        self._frame_count += 1
        if self._frame_count % self._flush_interval == 0:
            try:
                self._memmap.flush()
            except Exception:
                pass

    def flush_to_tiff(self, tifffile_module: Any) -> None:
        """Convert memmap to final OME-TIFF file."""
        # Ensure memmap is flushed to disk
        self._memmap.flush()

        # Write memmap array directly to TIFF; tifffile will read from the memmap
        tifffile_module.imwrite(
            self._path,
            self._memmap,
            bigtiff=True,
            ome=False,
            resolutionunit=tifffile_module.RESUNIT.MICROMETER,
            photometric=tifffile_module.PHOTOMETRIC.MINISBLACK,
            description=self._ome_xml,
        )

        # Clean up memmap
        del self._memmap
        self._memmap_path.unlink(missing_ok=True)


memmap_counter = count()

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
