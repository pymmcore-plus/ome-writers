from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from ome_writers import OMEStream

if TYPE_CHECKING:
    from collections.abc import Sequence
    from typing import Self

    import numpy as np

    from ome_writers.dimensions import DimensionInfo


class TiffStream(OMEStream):
    """Base API for writing a stream of frames to an OME file (tiff or zarr)."""

    def __init__(self) -> None:
        try:
            import tifffile
        except ImportError as e:
            raise ImportError(
                "TiffStream requires tifffile: `pip install tifffile`."
            ) from e
        self._tf = tifffile
        super().__init__()
        # local cache of {position index -> zarr.Array}
        # (will have a dataset for each position)
        self._position_arrays: dict[int, np.memmap] = {}

    def create(
        self, path: str, dtype: np.dtype, dimensions: Sequence[DimensionInfo]
    ) -> Self:
        """Create aa new stream for path, dtype, and dimensions."""
        self._position_arrays.clear()
        for p_index, sizes in enumerate(position_sizes):
            sizes["y"], sizes["x"] = frame.shape[-2:]
            self._position_arrays[p_index] = self.new_array(p_index, frame.dtype, dimensions)
        return self

    def append(self, frame: np.ndarray) -> None:
        """Append a frame to the stream."""

    def is_active(self) -> bool:
        """Return True if stream is active."""

    def flush(self) -> None:
        """Flush to disk."""

    def _normalize_path(self, path: str) -> str:
        return str(Path(path).expanduser().resolve())

    def _create_position_arrays(self, position_sizes: list[dict[str, int]]) -> None:
        self.position_arrays.clear()
        for p_index, sizes in enumerate(position_sizes):
            key = self.get_position_key(p_index)
            sizes["y"], sizes["x"] = frame.shape[-2:]
            self.position_arrays[key] = self.new_array(key, frame.dtype, sizes)
