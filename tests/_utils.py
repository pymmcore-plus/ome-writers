"""Shared test utilities."""

from __future__ import annotations

import time
from contextlib import suppress
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from pathlib import Path

    import numpy as np


def read_array_data(path: Path | str) -> np.ndarray:
    """Read array data from either Zarr or TIFF file.

    Parameters
    ----------
    path : Path | str
        Path to either a Zarr array directory or TIFF file.

    Returns
    -------
    np.ndarray
        The array data.
    """
    # path = Path(path)

    # Detect format by checking if it's a directory (Zarr) or file (TIFF)
    if str(path).endswith((".tif", ".tiff")):
        # TIFF format
        import numpy as np
        import tifffile

        return np.asarray(tifffile.imread(path))

    # Zarr format - try tensorstore first, fall back to zarr
    try:
        import tensorstore as ts

        ts_array = ts.open(
            {"driver": "zarr3", "kvstore": {"driver": "file", "path": str(path)}},
            open=True,
        ).result()
        import numpy as np

        return np.asarray(ts_array.read().result())
    except ImportError:
        import numpy as np
        import zarr

        return np.asarray(zarr.open_array(path))


def wait_for_frames(
    backend: Any,
    position_idx: int = 0,
    expected_count: int | None = None,
    timeout: float = 5.0,
) -> None:
    """Wait for backend to finish writing frames."""
    with suppress(ImportError):
        from ome_writers._backends._tensorstore import TensorstoreBackend

        if isinstance(backend, TensorstoreBackend):
            # Resolve all pending async write futures
            for future in backend._futures:
                future.result()
            return

    with suppress(ImportError):
        from ome_writers._backends._tifffile import TiffBackend

        if isinstance(backend, TiffBackend):
            start = time.time()
            while time.time() - start < timeout:
                thread = backend._position_managers[position_idx].thread
                if thread is None:
                    break
                with thread.state_lock:
                    written = thread.frames_written
                if expected_count is None or written >= expected_count:
                    if written > 0:
                        break
                time.sleep(0.01)  # Small sleep to avoid busy-waiting
