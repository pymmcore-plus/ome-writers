"""OME-TIFF backend using tifffile for sequential writes."""

from __future__ import annotations

import json
import math
import threading
import uuid
import warnings
from dataclasses import dataclass
from datetime import datetime, timezone
from itertools import count
from pathlib import Path
from queue import Queue
from typing import TYPE_CHECKING, Literal

import numpy as np

from ome_writers._backends._backend import ArrayBackend
from ome_writers._schema import StandardAxis
from ome_writers._units import ngff_to_ome_unit

if TYPE_CHECKING:
    from collections.abc import Iterator
    from typing import Any

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


@dataclass
class PositionManager:
    """Per-position writer/metadata state for TIFF backend."""

    file_path: str
    file_uuid: str | None
    thread: WriterThread | None
    queue: Queue[np.ndarray | None]
    metadata: ome.OME  # Cached metadata mirroring what's in the file
    name: str | None = None

    def __post_init__(self) -> None:
        self._lock = threading.Lock()
        self._metadata_dirty: bool = False

    def update_metadata(self, metadata: ome.OME, flush: bool = False) -> None:
        """Update cached metadata and mark as dirty.  Optionally flush to file."""
        with self._lock:
            self.metadata = metadata
            self._metadata_dirty = True
        # careful... our lock is not re-entrant, so avoid deadlock
        if flush:
            self.flush_metadata()

    def flush_metadata(self) -> None:
        """Flush current metadata to the TIFF file."""
        with self._lock:
            if not self._metadata_dirty:
                return

            try:
                tifffile.tiffcomment(
                    self.file_path,
                    comment=self.metadata.to_xml().encode("utf-8"),
                )
            except Exception as e:  # pragma: no cover
                warnings.warn(
                    f"Failed to update OME metadata in {self.file_path}: {e}",
                    stacklevel=2,
                )


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
        path = settings.root_path.lower()
        valid_extensions = [".tif", ".tiff", ".ome.tif", ".ome.tiff"]
        if not any(path.endswith(ext) for ext in valid_extensions):  # pragma: no cover
            return (
                "Root path must end with .tif, .tiff, .ome.tif, or .ome.tiff "
                "for TiffBackend."
            )

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
        valid_dims = {
            axis.value for axis in StandardAxis if axis != StandardAxis.POSITION
        }
        for dim in settings.array_storage_dimensions:
            if dim.name.lower() not in valid_dims:
                return (
                    f"Invalid dimension name '{dim.name}' for OME-TIFF. "
                    f"Valid names are: {', '.join(sorted(valid_dims))} "
                    f"(case-insensitive)."
                )

        return False

    def prepare(self, settings: AcquisitionSettings, router: FrameRouter) -> None:
        """Initialize OME-TIFF files and writer threads."""
        self._finalized = False
        self._storage_dims = storage_dims = settings.array_storage_dimensions
        self._dtype = settings.dtype
        positions = settings.positions
        root = Path(settings.root_path).expanduser().resolve()

        # Extract and validate compression
        if settings.compression in (None, "none"):
            compression = None
        else:
            compression = getattr(tifffile.COMPRESSION, settings.compression.upper())

        # Compute shape from storage dimensions
        shape = tuple(d.count if d.count is not None else 1 for d in storage_dims)

        # Check if any dimension is unbounded
        has_unbounded = any(d.count is None for d in storage_dims)

        # Prepare file paths
        fnames = self._prepare_files(root, len(positions), settings.overwrite)

        # Generate UUIDs and OME metadata for all positions
        num_pos = len(positions)
        uuids = [str(uuid.uuid4()) for _ in range(num_pos)] if num_pos > 1 else [None]
        all_images = [
            _create_ome_image(
                dims=storage_dims,
                dtype=self._dtype,
                filename=Path(fname).name,
                image_index=p_idx,
                file_uuid=uuids[p_idx],
                position_name=positions[p_idx].name if num_pos > 1 else None,
            )
            for p_idx, fname in enumerate(fnames)
        ]

        # Build complete OME metadata
        complete_ome = ome.OME(images=all_images)

        # Create writer thread for each position
        for p_idx, fname in enumerate(fnames):
            position_metadata = complete_ome.model_copy(deep=True)

            q = Queue[np.ndarray | None]()
            thread = WriterThread(
                path=fname,
                shape=shape,
                dtype=self._dtype,
                image_queue=q,
                ome_xml=position_metadata.to_xml(),
                has_unbounded=has_unbounded,
                compression=compression,
            )
            self._position_managers[p_idx] = PositionManager(
                file_path=fname,
                file_uuid=uuids[p_idx],
                thread=thread,
                queue=q,
                metadata=position_metadata,
                name=positions[p_idx].name if num_pos > 1 else None,
            )
            thread.start()

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

        self._position_managers[position_index].queue.put(frame)

        # Accumulate frame metadata with storage index
        if frame_metadata is not None:
            if position_index not in self._frame_metadata:
                self._frame_metadata[position_index] = []
            meta_with_idx = frame_metadata.copy()
            meta_with_idx["storage_index"] = index
            self._frame_metadata[position_index].append(meta_with_idx)

    def finalize(self) -> None:
        """Flush and close all TIFF writers."""
        if not self._finalized:
            # Signal threads to stop
            for writer in self._position_managers.values():
                writer.queue.put(None)

            # Wait for threads to finish
            for writer in self._position_managers.values():
                if writer.thread:
                    writer.thread.join(timeout=5)

            # Update OME metadata if unbounded dimensions were written
            has_unbounded = self._storage_dims and any(
                d.count is None for d in self._storage_dims
            )
            if has_unbounded:
                self._update_unbounded_metadata()
            elif self._frame_metadata:
                # For bounded dimensions with frame metadata, update metadata
                self._flush_frame_metadata()

            self._finalized = True

    def get_metadata(self) -> dict[int, ome.OME]:
        """Get the base OME metadata generated from acquisition settings.

        Returns a mapping of position indices to OME metadata objects that were
        auto-generated during prepare(). Each entry mirrors exactly what was written
        to the corresponding TIFF file.

        For multi-position acquisitions, each position's OME contains the complete
        metadata with all images (as per OME-TIFF companion file spec).
        For single-position acquisitions, the single entry contains just that image.

        Users can modify these objects to add meaningful names, timestamps, etc.,
        and pass the modified dict to update_metadata().

        Returns
        -------
        dict[int, ome_types.model.OME]
            Mapping of position indices to OME metadata objects, or empty dict if
            prepare() has not been called yet.
        """
        if not self._position_managers:  # pragma: no cover
            return {}
        return {
            p_idx: writer.metadata.model_copy(deep=True)
            for p_idx, writer in self._position_managers.items()
        }

    def update_metadata(self, metadata: dict[int, ome.OME]) -> None:
        """Update the OME metadata in the TIFF files.

        The metadata argument MUST be a dict mapping position indices to
        ome_types.OME instances.

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
            except KeyError as e:
                raise KeyError(f"Unknown position index: {pos_idx}") from e

    # -------------------

    def _flush_frame_metadata(self) -> None:
        """Flush accumulated frame metadata into OME-XML in TIFF files."""
        if not self._position_managers or not self._frame_metadata:  # pragma: no cover
            return

        # Get base OME (all position managers have the same complete structure)
        base_ome = next(iter(self._position_managers.values())).metadata.model_copy(
            deep=True
        )

        # Enhance the complete OME with frame metadata from all positions
        _enhance_ome_with_all_frame_metadata(base_ome, self._frame_metadata)

        # Update all position managers with the same complete enhanced OME
        for manager in self._position_managers.values():
            manager.update_metadata(base_ome.model_copy(deep=True), flush=True)

    def _update_unbounded_metadata(self) -> None:
        """Update OME metadata after writing unbounded dimensions.

        For unbounded dimensions, we write frames without knowing the final count.
        After writing completes, update the OME-XML with the actual frame counts.
        """
        # Get actual frames_written from first position's thread
        # (all positions should have written the same number of frames)
        if (
            not self._storage_dims
            or not (writer := self._position_managers.get(0))
            or not writer.thread
            or not (frames_written := writer.thread.frames_written)
        ):
            return  # no frames written

        # Infer unbounded dimension count from total frames written
        known_product = math.prod(d.count or 1 for d in self._storage_dims[:-2])
        unbounded_count = frames_written // known_product
        corrected_dims = tuple(
            d.model_copy(update={"count": unbounded_count}) if d.count is None else d
            for d in self._storage_dims
        )

        # Build complete OME with corrected dimensions
        complete_ome = ome.OME(
            images=[
                _create_ome_image(
                    dims=corrected_dims,
                    dtype=self._dtype,
                    filename=Path(writer.file_path).name,
                    image_index=p_idx,
                    file_uuid=writer.file_uuid,
                    position_name=writer.name,
                )
                for p_idx, writer in sorted(self._position_managers.items())
            ]
        )

        # Enhance with frame metadata (Planes and StructuredAnnotations)
        if self._frame_metadata:
            _enhance_ome_with_all_frame_metadata(complete_ome, self._frame_metadata)

        # Update all position managers with complete enhanced OME and flush
        for manager in self._position_managers.values():
            manager.update_metadata(complete_ome.model_copy(deep=True), flush=True)

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
                            description=self._ome_xml_bytes if i == 0 else None,
                            compression=self._compression,
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
                        description=self._ome_xml_bytes,
                        compression=self._compression,
                    )
        except Exception as e:  # pragma: no cover
            # Suppress over-eager tifffile exception for incomplete writes
            if "wrong number of bytes" in str(e):
                return
            raise


_thread_counter = count()

# ------------------------

# helpers for frame metadata


def _calculate_plane_indices(
    frame_idx: int, size_c: int, size_z: int, size_t: int, dim_order: str
) -> tuple[int, int, int]:
    """Calculate (C, Z, T) indices for a frame based on dimension order.

    Parameters
    ----------
    frame_idx : int
        Frame index in acquisition order
    size_c, size_z, size_t : int
        Size of each dimension
    dim_order : str
        OME dimension order string (e.g., "XYZCT")

    Returns
    -------
    tuple[int, int, int]
        (the_c, the_z, the_t) indices for this frame
    """
    # Extract order of non-spatial dims (C, Z, T only)
    non_spatial = "".join(d for d in dim_order if d in "CZT")

    # Calculate indices based on dimension order
    # OME-XML is fastest-varying on the RIGHT (opposite of numpy)
    indices = {"C": 0, "Z": 0, "T": 0}
    sizes = {"C": size_c, "Z": size_z, "T": size_t}

    remaining = frame_idx
    for dim in reversed(non_spatial):  # Right-most varies fastest
        indices[dim] = remaining % sizes[dim]
        remaining //= sizes[dim]

    return indices["C"], indices["Z"], indices["T"]


def _create_plane_with_metadata(
    the_c: int,
    the_z: int,
    the_t: int,
    frame_idx: int,
    frame_metadata: dict[str, Any],
) -> ome.Plane:
    """Create a Plane object with recognized metadata keys mapped to attributes.

    Parameters
    ----------
    the_c : int
        Channel index for this plane
    the_z : int
        Z index for this plane
    the_t : int
        Time index for this plane
    frame_idx : int
        Frame index in acquisition order (for annotation reference)
    frame_metadata : dict
        Frame metadata dict with recognized and custom keys

    Returns
    -------
    ome.Plane
        Plane object with recognized keys mapped to attributes
    """
    plane_kwargs: dict[str, Any] = {
        "the_c": the_c,
        "the_z": the_z,
        "the_t": the_t,
    }

    # Map recognized keys to Plane attributes
    if "delta_t" in frame_metadata:
        plane_kwargs["delta_t"] = float(frame_metadata["delta_t"])
        plane_kwargs["delta_t_unit"] = ome.UnitsTime.SECOND

    if "exposure_time" in frame_metadata:
        plane_kwargs["exposure_time"] = float(frame_metadata["exposure_time"])
        plane_kwargs["exposure_time_unit"] = ome.UnitsTime.SECOND

    if "position_x" in frame_metadata:
        plane_kwargs["position_x"] = float(frame_metadata["position_x"])
        plane_kwargs["position_x_unit"] = ome.UnitsLength.MICROMETER

    if "position_y" in frame_metadata:
        plane_kwargs["position_y"] = float(frame_metadata["position_y"])
        plane_kwargs["position_y_unit"] = ome.UnitsLength.MICROMETER

    if "position_z" in frame_metadata:
        plane_kwargs["position_z"] = float(frame_metadata["position_z"])
        plane_kwargs["position_z_unit"] = ome.UnitsLength.MICROMETER

    return ome.Plane(**plane_kwargs)


def _create_map_annotation_for_frame(
    frame_idx: int, frame_metadata: dict[str, Any], position_offset: int = 0
) -> ome.MapAnnotation:
    """Create a MapAnnotation containing all frame metadata.

    Parameters
    ----------
    frame_idx : int
        Frame index in acquisition order within this position
    frame_metadata : dict
        Complete frame metadata dict
    position_offset : int
        Position index offset for creating unique annotation IDs

    Returns
    -------
    ome.MapAnnotation
        MapAnnotation with all metadata as key-value pairs
    """
    # Convert all values to strings for MapAnnotation
    # TODO: if there are actually nested structures (lists/dicts),
    # we need to do better here. For now, just JSON-dump non-strings.
    map_pairs = [
        ome.Map.M(k=str(k), value=v if isinstance(v, str) else json.dumps(v))
        for k, v in frame_metadata.items()
    ]

    # Use position offset in ID to ensure uniqueness across positions
    return ome.MapAnnotation(
        id=f"Annotation:Pos{position_offset}:Frame:{frame_idx}",
        namespace="ome-writers.frame_metadata",
        value=ome.Map(ms=map_pairs),
    )


def _enhance_ome_with_all_frame_metadata(
    ome_obj: ome.OME,
    frame_metadata_by_position: dict[int, list[dict[str, Any]]],
) -> None:
    """Enhance OME metadata with frame metadata for all positions in-place.

    Parameters
    ----------
    ome_obj : ome.OME
        Complete OME metadata object with all images (modified in-place)
    frame_metadata_by_position : dict[int, list[dict]]
        Mapping of position indices to their frame metadata lists
    """
    if not frame_metadata_by_position or not ome_obj.images:
        return

    all_annotations = []
    for p_idx in sorted(frame_metadata_by_position.keys()):
        frame_metadata = frame_metadata_by_position[p_idx]
        if not frame_metadata or p_idx >= len(ome_obj.images):
            continue

        image = ome_obj.images[p_idx]
        if not image.pixels:
            continue

        # Create MapAnnotations for this position's frames
        map_annotations = [
            _create_map_annotation_for_frame(idx, meta, position_offset=p_idx)
            for idx, meta in enumerate(frame_metadata)
        ]
        all_annotations.extend(map_annotations)

        # Create Plane objects for this position's frames
        pixels = image.pixels
        size_c = pixels.size_c or 1
        size_z = pixels.size_z or 1
        size_t = pixels.size_t or 1
        dim_order = pixels.dimension_order.value

        planes = []
        for frame_idx, meta in enumerate(frame_metadata):
            the_c, the_z, the_t = _calculate_plane_indices(
                frame_idx, size_c, size_z, size_t, dim_order
            )
            plane = _create_plane_with_metadata(the_c, the_z, the_t, frame_idx, meta)

            # Link to MapAnnotation with position-specific ID
            annot_id = f"Annotation:Pos{p_idx}:Frame:{frame_idx}"
            plane.annotation_refs.append(ome.AnnotationRef(id=annot_id))
            planes.append(plane)

        # Add planes to this image's pixels
        pixels.planes = planes

    # Add all annotations to OME root
    if all_annotations:
        if isinstance(ome_obj.structured_annotations, list):
            ome_obj.structured_annotations.extend(all_annotations)
        else:
            ome_obj.structured_annotations = all_annotations  # type: ignore


# helpers for position-specific OME metadata updates


def _create_ome_image(
    dims: tuple[Dimension, ...],
    dtype: str,
    filename: str,
    image_index: int,
    file_uuid: str | None = None,
    position_name: str | None = None,
) -> ome.Image:
    """Generate OME Image object for TIFF file."""
    # Build shape dictionary from dimensions
    # OME-XML has fasted-varying dimension on the left (rightmost in storage order)
    shape_dict = {d.name.upper(): d.count or 1 for d in reversed(dims)}
    size_z = shape_dict.setdefault("Z", 1)
    size_c = shape_dict.setdefault("C", 1)
    size_t = shape_dict.setdefault("T", 1)
    our_order = "".join(x for x in shape_dict if x in "ZCT")  # only Z,C,T
    for order in ome.Pixels_DimensionOrder:
        if order.value.endswith(our_order):
            dim_order = order
            break
    else:
        raise ValueError(
            f"Cannot determine OME dimension order for storage dimensions: {dims}"
        ) from None

    # Build TiffData (with UUIDs for multi-position)
    tiff_data_blocks = ome.TiffData(
        plane_count=size_t * size_c * size_z,
        uuid=(
            ome.TiffData.UUID(value=f"urn:uuid:{file_uuid}", file_name=filename)
            if file_uuid
            else None
        ),
    )

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
        tiff_data_blocks=[tiff_data_blocks],
    )
    # Add physical sizes if available
    dims_by_name = {d.name.lower(): d for d in dims}
    for axis in ["x", "y", "z"]:
        if (dim := dims_by_name.get(axis)) and dim.scale is not None:
            setattr(pixels, f"physical_size_{axis}", dim.scale)
            # if dim.type is space, it's guaranteed to be a valid NGFF unit
            if dim.unit and dim.type == "space":
                if ome_unit_str := ngff_to_ome_unit(dim.unit):
                    try:
                        # ome-types will validate unit assignment
                        setattr(pixels, f"physical_size_{axis}_unit", ome_unit_str)
                    except Exception:  # pragma: no cover  (should be unreachable)
                        warnings.warn(
                            f"Could not convert unit '{dim.unit}' (→ '{ome_unit_str}') "
                            f"to ome.UnitsLength. Skipping unit assignment.",
                            stacklevel=2,
                        )

    name = position_name if position_name else Path(filename).stem.removesuffix(".ome")
    return ome.Image(
        id=f"Image:{image_index}",
        acquisition_date=datetime.now(timezone.utc),
        pixels=pixels,
        name=name,
    )
