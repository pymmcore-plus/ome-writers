from __future__ import annotations

import importlib
import importlib.util
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import tifffile
from typing_extensions import Self

from ome_writers._dimensions import Dimension
from ome_writers._stream_base import MultiPositionOMEStream

if TYPE_CHECKING:
    from collections.abc import Sequence


class TiffStream(MultiPositionOMEStream):
    """A concrete OMEStream implementation for writing to OME-TIFF files.

    This writer is designed for deterministic acquisitions where the full experiment
    shape is known ahead of time. It works by creating all necessary OME-TIFF
    files at the start of the acquisition and using memory-mapped arrays for
    efficient, sequential writing of incoming frames.

    If a 'p' (position) dimension is included in the dimensions, a separate
    OME-TIFF file will be created for each position.

    Attributes
    ----------
    _memmaps : Dict[int, np.memmap]
        A dictionary mapping position index to its numpy memmap array.
    """

    @classmethod
    def is_available(cls) -> bool:
        """Check if the tifffile package is available."""
        return importlib.util.find_spec("tifffile") is not None

    def __init__(self) -> None:
        super().__init__()
        # Using dictionaries to handle multi-position ('p') acquisitions
        self._memmaps: dict[int, np.memmap] = {}
        self._dim_order = "ptzcyx"
        self._is_active = False

    def create(
        self,
        path: str,
        dtype: np.dtype,
        dimensions: Sequence[Dimension],
        *,
        overwrite: bool = False,
    ) -> Self:
        # Use MultiPositionOMEStream to handle position logic
        num_positions, non_position_dims = self._init_positions(dimensions)
        self._delete_existing = overwrite
        self._path = Path(self._normalize_path(path))
        shape_5d = tuple(d.size for d in non_position_dims)

        path_root = str(self._path)
        for possible_ext in [".ome.tiff", ".ome.tif", ".tiff", ".tif"]:
            if path_root.endswith(possible_ext):
                ext = possible_ext
                path_root = path_root[: -len(possible_ext)]
                break
        else:
            ext = self._path.suffix

        # Create a memmap for each position
        for p_idx in range(num_positions):
            # only append position index if there are multiple positions
            if num_positions > 1:
                p_path = Path(f"{path_root}_p{p_idx:03d}{ext}")
            else:
                p_path = self._path

            # Check if file exists and handle overwrite parameter
            if p_path.exists():
                if not overwrite:
                    raise FileExistsError(
                        f"File {p_path} already exists. "
                        "Use overwrite=True to overwrite it."
                    )
                p_path.unlink()

            # Ensure the parent directory exists
            p_path.parent.mkdir(parents=True, exist_ok=True)

            # Create empty OME-TIFF file with the correct shape and metadata.
            tifffile.imwrite(
                p_path,
                shape=shape_5d,
                dtype=dtype,
                ome=True,
                metadata=self._generate_ome_metadata(p_path.name, non_position_dims),
            )

            # Create a memory map to the file on disk.
            self._memmaps[p_idx] = mm = tifffile.memmap(str(p_path), dtype=dtype)
            # This line is important, as tifffile.memmap can lose singleton dimensions
            mm.shape = shape_5d

        self._is_active = True
        return self

    def _write_to_backend(
        self, array_key: str, index: tuple[int, ...], frame: np.ndarray
    ) -> None:
        """TIFF-specific write implementation."""
        p_idx = int(array_key)
        memmap_array = self._memmaps[p_idx]
        memmap_array[index] = frame

    def flush(self) -> None:
        """Flushes all buffered data in the memory-mapped files to disk."""
        for memmap_array in self._memmaps.values():
            if hasattr(memmap_array, "flush"):
                memmap_array.flush()
        # Mark as inactive after flushing - this is consistent with other backends
        self._is_active = False

    def is_active(self) -> bool:
        """Return True if the stream is currently active."""
        return self._is_active

    def _generate_ome_metadata(
        self, name: str, dimensions: Sequence[Dimension]
    ) -> dict:
        """Create the metadata dictionary required by tifffile for OME-XML."""
        dim_map = {d.label: d for d in dimensions}
        axes = "".join(d.label for d in dimensions).upper()

        metadata = {
            "axes": axes,
            "Name": name,
            "PhysicalSizeX": dim_map.get("x", Dimension("x", 0, (1.0, ""))).ome_scale,
            "PhysicalSizeXUnit": dim_map.get(
                "x", Dimension("x", 0, (1.0, ""))
            ).ome_unit,
            "PhysicalSizeY": dim_map.get("y", Dimension("y", 0, (1.0, ""))).ome_scale,
            "PhysicalSizeYUnit": dim_map.get(
                "y", Dimension("y", 0, (1.0, ""))
            ).ome_unit,
        }
        if "z" in dim_map:
            metadata["PhysicalSizeZ"] = dim_map["z"].ome_scale
            metadata["PhysicalSizeZUnit"] = dim_map["z"].ome_unit
        if "t" in dim_map:
            metadata["TimeIncrement"] = dim_map["t"].ome_scale
            metadata["TimeIncrementUnit"] = dim_map["t"].ome_unit
        if "c" in dim_map:
            # Assuming channel names are just indices if not provided otherwise
            metadata["Channel"] = {
                "Name": [f"Channel_{i}" for i in range(dim_map["c"].size)]
            }
        return metadata
