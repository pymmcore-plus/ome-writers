"""Shared test utilities."""

from __future__ import annotations

import json
import time
from contextlib import suppress
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from ome_writers._stream import OMEStream


def read_array_data(root: Path | str, position_index: int = 0) -> np.ndarray:
    """Read array data from a Zarr, TIFF, or scratch output root.

    Handles format-specific position/resolution path logic internally.

    Parameters
    ----------
    root : Path | str
        Root output path (e.g. the OME-Zarr store, TIFF file, or scratch dir).
    position_index : int
        Position index to read (default 0).
    """

    root = Path(root)

    # TIFF format
    if str(root).endswith((".tif", ".tiff")):
        import tifffile

        return np.asarray(tifffile.imread(root))

    # Scratch format - has manifest.json with position shapes
    manifest = root / "manifest.json"
    if manifest.exists():
        meta = json.loads(manifest.read_text())
        shape = tuple(meta["position_shapes"][position_index])
        dtype = np.dtype(meta["dtype"])
        arr = np.memmap(
            root / f"pos_{position_index}.dat", dtype=dtype, mode="r", shape=shape
        )
        return np.array(arr)

    # Zarr format - resolve to resolution level 0 array
    array_path = root / "0"
    try:
        import tensorstore as ts

        ts_array = ts.open(
            {
                "driver": "zarr3",
                "kvstore": {"driver": "file", "path": str(array_path)},
            },
            open=True,
        ).result()
        return np.asarray(ts_array.read().result())
    except ImportError:
        import zarr

        return np.asarray(zarr.open_array(array_path))


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


def wait_for_pending_callbacks(
    stream: OMEStream, timeout: float = 1.0, barriers: int = 20
) -> None:
    """Wait for all pending async callbacks to complete (for testing).

    Submits barrier tasks serially to ensure all prior work completes tests.
    """
    if executor := stream._callback_executor:
        for _ in range(barriers):
            executor.submit(lambda: None).result(timeout=timeout)
