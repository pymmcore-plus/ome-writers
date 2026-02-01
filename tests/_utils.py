"""Shared test utilities."""

from __future__ import annotations

from typing import TYPE_CHECKING

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
