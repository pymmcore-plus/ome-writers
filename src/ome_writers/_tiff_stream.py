from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import tifffile
from typing_extensions import Self

from ._dimensions import DimensionInfo
from ._stream_base import OMEStream

if TYPE_CHECKING:
    from collections.abc import Sequence


class TiffStreamWriter(OMEStream):
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
    _frame_counter : int
        A counter for the total number of frames appended to the stream.
    _total_frames : int
        The total number of frames expected in the entire stream.
    _loop_dims_info : List[DimensionInfo]
        An ordered list of non-spatial dimensions that are looped over.
    _array_dims_info : Dict[int, List[DimensionInfo]]
        A map of position index to the dimensions of its corresponding 5D array.
    """

    def __init__(self) -> None:
        super().__init__()
        # Using dictionaries to handle multi-position ('p') acquisitions
        self._memmaps: dict[int, np.memmap] = {}

        self._frame_counter: int = 0
        self._total_frames: int = 0
        self._loop_dims_info: list[DimensionInfo] = []
        self._array_dims_info: dict[int, list[DimensionInfo]] = {}
        self._dim_order = "ptzcyx"

    def create(
        self, path: str, dtype: np.dtype, dimensions: Sequence[DimensionInfo]
    ) -> Self:
        """
        Creates and pre-allocates OME-TIFF file(s) on disk.

        This method sets up the file structure based on the provided dimensions.
        It creates a memory-mapped array for each position, ready to receive frames.

        Parameters
        ----------
        path : str
            The base path for the output file(s).
        dtype : np.dtype
            The numpy data type of the image frames (e.g., np.uint16).
        dimensions : Sequence[DimensionInfo]
            A sequence describing the dimensions of the experiment.

        Returns
        -------
        Self
            The instance of the writer, allowing for chaining.
        """
        if self.is_active():
            raise RuntimeError("Stream is already active. Please close it first.")

        self._path = Path(self._normalize_path(path))
        self._dtype = np.dtype(dtype)
        self._dimensions = dimensions

        # 1. Parse and sort dimensions
        dim_map: dict[str, DimensionInfo] = {d.label: d for d in dimensions}
        sorted_dims = [dim_map[label] for label in self._dim_order if label in dim_map]

        y_dim = dim_map.get("y")
        x_dim = dim_map.get("x")
        if not (y_dim and x_dim):
            raise ValueError("Dimensions must include 'x' and 'y'.")

        self._loop_dims_info = [d for d in sorted_dims if d.label not in "yx"]
        p_dim = dim_map.get("p")
        n_positions = p_dim.size if p_dim else 1

        self._total_frames = np.prod([d.size for d in self._loop_dims_info]).item()

        # 2. Create a file for each position
        for p_idx in range(n_positions):
            # For each position, determine the file path
            if p_dim:
                p_path = self._path.with_stem(f"{self._path.stem}_p{p_idx:03d}")
            else:
                p_path = self._path
            p_path.parent.mkdir(parents=True, exist_ok=True)

            # Determine the shape and axes for this position's 5D stack
            pos_dims = [d for d in self._loop_dims_info if d.label != "p"]
            self._array_dims_info[p_idx] = pos_dims

            shape_5d = tuple(d.size for d in pos_dims)
            axes_5d = "".join(d.label for d in pos_dims).upper()

            # The full shape includes the y and x dimensions
            full_shape = (*shape_5d, y_dim.size, x_dim.size)
            full_axes = axes_5d + "YX"

            # 3. Generate OME metadata
            ome_metadata = self._generate_ome_metadata(
                p_path.name, full_axes, [*pos_dims, y_dim, x_dim]
            )

            # 4. Write the empty file with metadata
            tifffile.imwrite(
                p_path,
                shape=full_shape,
                dtype=self._dtype,
                ome=True,
                metadata=ome_metadata,
            )

            # 5. Create a memory map to the file on disk.
            # The memmap provides a writable numpy-like array interface.
            memmap_array = tifffile.memmap(str(p_path), dtype=self._dtype)

            # This line is important, as tifffile.memmap can lose singleton dimensions
            memmap_array.shape = full_shape
            self._memmaps[p_idx] = memmap_array

        self._is_active = True
        self._frame_counter = 0
        return self

    def append(self, frame: np.ndarray) -> None:
        """
        Appends a single 2D frame to the appropriate location in the stream.

        The writer maintains an internal counter and determines the frame's
        destination (position, time, z-slice, channel) automatically. Frames
        are assumed to arrive in a determined, C-style order (last index
        varying fastest).

        Parameters
        ----------
        frame : np.ndarray
            The 2D image frame to write.
        """
        if not self.is_active():
            raise RuntimeError("Stream is not active. Call create() first.")
        if self._frame_counter >= self._total_frames:
            raise RuntimeError(
                f"Attempted to write frame {self._frame_counter + 1}, but "
                f"stream is already full with {self._total_frames} frames."
            )

        # 1. Determine the multidimensional index for this frame
        loop_shape = tuple(d.size for d in self._loop_dims_info)
        multi_index = np.unravel_index(self._frame_counter, loop_shape)
        index_map = dict(zip((d.label for d in self._loop_dims_info), multi_index))

        # 2. Get the correct memmap array and the index within that array
        p_idx = index_map.get("p", 0)
        memmap_array = self._memmaps[p_idx]

        array_dims = [d.label for d in self._array_dims_info[p_idx]]
        write_index = tuple(index_map[d] for d in array_dims)

        # 3. Write the frame data
        memmap_array[write_index] = frame
        self._frame_counter += 1

    def flush(self) -> None:
        """Flushes all buffered data in the memory-mapped files to disk."""
        for memmap_array in self._memmaps.values():
            if hasattr(memmap_array, "flush"):
                memmap_array.flush()

    def close(self) -> None:
        """Flushes data and releases file resources."""
        if not self.is_active():
            return

        self.flush()
        # The underlying file handles for the memory maps are closed when the
        # memmap objects are garbage collected. We'll clear the dict here
        # to release the references and trigger that process.
        self._memmaps.clear()

        self._is_active = False
        print(f"Stream closed. {self._frame_counter} frames written.")

    def _generate_ome_metadata(
        self, name: str, axes: str, dimensions: list[DimensionInfo]
    ) -> dict:
        """Create the metadata dictionary required by tifffile for OME-XML."""
        dim_map = {d.label: d for d in dimensions}
        metadata = {
            "axes": axes,
            "Name": name,
            "PhysicalSizeX": dim_map.get(
                "x", DimensionInfo("x", 0, (1.0, ""))
            ).ome_scale,
            "PhysicalSizeXUnit": dim_map.get(
                "x", DimensionInfo("x", 0, (1.0, ""))
            ).ome_unit,
            "PhysicalSizeY": dim_map.get(
                "y", DimensionInfo("y", 0, (1.0, ""))
            ).ome_scale,
            "PhysicalSizeYUnit": dim_map.get(
                "y", DimensionInfo("y", 0, (1.0, ""))
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
