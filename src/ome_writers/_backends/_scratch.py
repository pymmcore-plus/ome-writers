"""Scratch array backend for ome-writers."""

from __future__ import annotations

import atexit
import json
import os
import shutil
import tempfile
import warnings
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

import numpy as np

from ome_writers._backends._backend import ArrayBackend
from ome_writers._schema import ScratchFormat

if TYPE_CHECKING:
    from collections.abc import Sequence
    from io import IOBase

    from ome_writers._backends._backend import ArrayLike
    from ome_writers._router import FrameRouter
    from ome_writers._schema import AcquisitionSettings


class ScratchBackend(ArrayBackend):
    """Backend that stores frames in numpy arrays (or memmap for crash recovery)."""

    __slots__ = (
        "_arrays",
        "_finalized",
        "_logical_shapes",
        "_metadata_fh",
        "_root_path",
        "_settings_dump",
        "_unbounded_axes",
    )

    def __init__(self) -> None:
        # One array per position; may be in-memory or memmap depending on settings.
        self._arrays: list[np.ndarray | np.memmap] = []
        # Logical shapes of each position
        # may be smaller than the backing array if it was resized.
        self._logical_shapes: list[list[int]] = []
        # indices of axes with unbounded (None) size
        self._unbounded_axes: list[int] = []
        # If set, memmap files and metadata will be written to root_path for recovery.
        self._root_path: Path | None = None
        # Dump of acquisition settings used for manifest.json;
        self._settings_dump: dict = {}
        # file handle for frame_metadata.jsonl, if root_path is set
        self._metadata_fh: IOBase | None = None
        # Flag to prevent multiple finalization steps
        self._finalized: bool = False

    def is_incompatible(self, settings: AcquisitionSettings) -> Literal[False] | str:
        return False

    def prepare(self, settings: AcquisitionSettings, router: FrameRouter) -> None:
        fmt = settings.format
        if not isinstance(fmt, ScratchFormat):  # pragma: no cover
            raise ValueError(f"Expected ScratchFormat, got {type(fmt)}")

        dims = settings.array_storage_dimensions
        shape = tuple(d.count or 1 for d in dims)
        self._unbounded_axes = [i for i, d in enumerate(dims) if d.count is None]

        if settings.root_path:
            self._root_path = Path(settings.root_path)
            if self._root_path.exists():
                if not settings.overwrite:
                    raise FileExistsError(
                        f"{self._root_path} already exists. "
                        f"Use overwrite=True to overwrite it."
                    )
                shutil.rmtree(self._root_path)
            self._root_path.mkdir(parents=True, exist_ok=True)

        self._settings_dump = settings.model_dump(mode="json", exclude_unset=True)

        # Check memory bound for pure in-memory mode
        if self._root_path is None:
            bytes_per_pos = int(np.prod(shape)) * np.dtype(settings.dtype).itemsize
            total_bytes = len(settings.positions) * bytes_per_pos

            if total_bytes > fmt.max_memory_bytes:
                if fmt.spill_to_disk:
                    self._root_path = Path(tempfile.mkdtemp(prefix="ome_scratch_"))
                    atexit.register(shutil.rmtree, self._root_path, True)
                    warnings.warn(
                        f"Scratch arrays would require "
                        f"~{total_bytes / 1e9:.1f} GB in memory. "
                        f"Using disk-backed storage at {self._root_path}",
                        stacklevel=4,
                    )
                else:
                    raise MemoryError(
                        f"Scratch arrays would require "
                        f"~{total_bytes / 1e9:.1f} GB, exceeding "
                        f"max_memory_bytes={fmt.max_memory_bytes / 1e9:.1f} GB. "
                        f"Set root_path to use disk-backed storage, "
                        f"or increase max_memory_bytes."
                    )

        # Initialize arrays (empty or memmap) and logical shapes for each position
        for i in range(len(settings.positions)):
            if self._root_path is not None:
                arr: np.ndarray = np.memmap(
                    self._root_path / f"pos_{i}.dat",
                    dtype=settings.dtype,
                    mode="w+",
                    shape=shape,
                )
            else:
                arr = np.zeros(shape, dtype=settings.dtype)
            self._arrays.append(arr)
            self._logical_shapes.append([*shape])

        # If using disk-backed storage
        # open metadata file for appending and write initial manifest
        if self._root_path is not None:
            self._write_manifest()
            self._metadata_fh = open(self._root_path / "frame_metadata.jsonl", "a")

    def write(
        self,
        position_index: int,
        index: tuple[int, ...],
        frame: np.ndarray,
        *,
        frame_metadata: dict[str, Any] | None = None,
    ) -> None:
        self._ensure_size(position_index, index)
        self._arrays[position_index][index] = frame
        self._update_logical_shape(position_index, index)
        if frame_metadata and self._metadata_fh is not None:
            record = {"_pos": position_index, "_idx": index, **frame_metadata}
            self._metadata_fh.write(json.dumps(record) + "\n")

    def advance(self, indices: Sequence[tuple[int, tuple[int, ...]]]) -> None:
        for pos_idx, index in indices:
            self._ensure_size(pos_idx, index)
            self._update_logical_shape(pos_idx, index)

    def finalize(self) -> None:
        if self._finalized:
            return
        self._finalized = True

        if self._metadata_fh is not None:
            self._metadata_fh.close()
            self._metadata_fh = None
        if self._root_path is not None:
            for arr in self._arrays:
                if isinstance(arr, np.memmap):
                    arr.flush()
            self._write_manifest()

    def get_arrays(self) -> Sequence[ArrayLike]:
        return [_ScratchArrayView(self, i) for i in range(len(self._arrays))]

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _update_logical_shape(self, pos_idx: int, index: tuple[int, ...]) -> None:
        logical = self._logical_shapes[pos_idx]
        for ax in self._unbounded_axes:
            idx_val = index[ax]
            if idx_val >= logical[ax]:
                logical[ax] = idx_val + 1

    def _ensure_size(self, pos_idx: int, index: tuple[int, ...]) -> None:
        """Grow backing array if index exceeds current allocation."""
        arr = self._arrays[pos_idx]
        old_shape = arr.shape
        new_shape = [*old_shape]
        needs_resize = False

        # determine if any unbounded axes need to grow to accommodate index;
        # if so, double size to amortize future resizes
        for ax in self._unbounded_axes:
            needed = index[ax] + 1 if isinstance(index[ax], int) else 0
            if needed > old_shape[ax]:
                new_shape[ax] = max(needed, old_shape[ax] * 2)
                needs_resize = True

        if not needs_resize:
            return

        new_shape_t = tuple(new_shape)
        copy_slices = tuple(slice(0, s) for s in old_shape)

        if self._root_path is None:
            # Fast path for in-memory arrays: allocate a new array and copy old data
            new_arr = np.zeros(new_shape_t, dtype=arr.dtype)
            new_arr[copy_slices] = arr
            self._arrays[pos_idx] = new_arr
            return

        dat_path = self._root_path / f"pos_{pos_idx}.dat"
        old_arr = self._arrays[pos_idx]
        dtype = old_arr.dtype
        if isinstance(old_arr, np.memmap):
            old_arr.flush()
        # Ensure backend no longer holds a reference to the old memmap before
        # truncate/remap (important on Windows).
        self._arrays[pos_idx] = np.empty((0,), dtype=dtype)
        del old_arr, arr

        try:
            # In-place extend: OS zero-fills new bytes, existing data untouched
            new_bytes = int(np.prod(new_shape_t)) * np.dtype(dtype).itemsize
            os.truncate(dat_path, new_bytes)
            self._arrays[pos_idx] = np.memmap(
                dat_path, dtype=dtype, mode="r+", shape=new_shape_t
            )
        except (PermissionError, OSError):
            # Fallback: copy via temp file (e.g. Windows with live views)
            tmp_path = dat_path.with_suffix(".dat.tmp")
            prev = np.memmap(dat_path, dtype=dtype, mode="r", shape=old_shape)
            new_arr = np.memmap(tmp_path, dtype=dtype, mode="w+", shape=new_shape_t)
            new_arr[copy_slices] = prev[:]
            new_arr.flush()
            del prev, new_arr  # release handles before move
            shutil.move(tmp_path, dat_path)
            self._arrays[pos_idx] = np.memmap(
                dat_path, dtype=dtype, mode="r+", shape=new_shape_t
            )

    def _write_manifest(self) -> None:
        """Write manifest.json alongside memmap files."""
        if self._root_path is None:
            return
        self._settings_dump["position_shapes"] = [
            self._logical_shapes[i] for i in range(len(self._arrays))
        ]
        (self._root_path / "manifest.json").write_text(json.dumps(self._settings_dump))


class _ScratchArrayView:
    """Live proxy that always reads from the backend's current array."""

    __slots__ = ("_backend", "_pos_idx")

    def __init__(self, backend: ScratchBackend, pos_idx: int) -> None:
        self._backend = backend
        self._pos_idx = pos_idx

    @property
    def shape(self) -> tuple[int, ...]:
        return tuple(self._backend._logical_shapes[self._pos_idx])

    @property
    def dtype(self) -> np.dtype:
        return self._backend._arrays[self._pos_idx].dtype

    def __setitem__(self, key: Any, value: Any) -> None:
        raise TypeError("_ScratchArrayView is read-only")

    def __getitem__(self, key: Any) -> Any:
        arr = self._backend._arrays[self._pos_idx]
        if self._backend._unbounded_axes:
            # Clip to logical shape so over-allocated backing storage is hidden
            bounds = (slice(0, s) for s in self._backend._logical_shapes[self._pos_idx])
            arr = arr[tuple(bounds)]
        result = arr[key]
        if isinstance(result, np.ndarray):
            # it's important to create a view, so that writeable=False doesn't
            # propagate back to the backend's array and break future writes.
            result = result.view()
            result.flags.writeable = False
        return result
