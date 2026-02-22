"""In-memory array backend for ome-writers."""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

import numpy as np

from ome_writers._backends._backend import ArrayBackend

if TYPE_CHECKING:
    from collections.abc import Sequence
    from io import IOBase

    from ome_writers._backends._backend import ArrayLike
    from ome_writers._router import FrameRouter
    from ome_writers._schema import AcquisitionSettings


class MemoryBackend(ArrayBackend):
    """Backend that stores frames in numpy arrays (or memmap for crash recovery)."""

    def __init__(self) -> None:
        self._arrays: list[np.ndarray | np.memmap] = []
        self._logical_shapes: list[list[int]] = []
        self._unbounded_axes: list[int] = []
        self._root_path: Path | None = None
        self._settings_dump: dict = {}
        self._metadata_fh: IOBase | None = None  # file handle for frame_metadata.jsonl
        self._finalized: bool = False

    def is_incompatible(self, settings: AcquisitionSettings) -> Literal[False] | str:
        return False

    def prepare(self, settings: AcquisitionSettings, router: FrameRouter) -> None:
        dims = settings.array_storage_dimensions
        self._settings_dump = settings.model_dump(mode="json", exclude_unset=True)

        shape = tuple(d.count or 1 for d in dims)
        self._unbounded_axes = [i for i, d in enumerate(dims) if d.count is None]

        if settings.root_path:
            self._root_path = Path(settings.root_path)
            self._root_path.mkdir(parents=True, exist_ok=True)

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
            record = {"pos": position_index, "idx": index, **frame_metadata}
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
        return [_MemoryArrayView(self, i) for i in range(len(self._arrays))]

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _update_logical_shape(self, pos_idx: int, index: tuple[int, ...]) -> None:
        logical = self._logical_shapes[pos_idx]
        for ax, idx_val in enumerate(index):
            if isinstance(idx_val, int) and idx_val >= logical[ax]:
                logical[ax] = idx_val + 1

    def _ensure_size(self, pos_idx: int, index: tuple[int, ...]) -> None:
        """Grow backing array if index exceeds current allocation."""
        arr = self._arrays[pos_idx]
        old_shape = arr.shape
        new_shape = [*old_shape]
        needs_resize = False

        for ax in self._unbounded_axes:
            needed = index[ax] + 1 if isinstance(index[ax], int) else 0
            if needed > old_shape[ax]:
                new_shape[ax] = max(needed, old_shape[ax] * 2)
                needs_resize = True

        if not needs_resize:
            return

        new_shape_t = tuple(new_shape)
        copy_slices = tuple(slice(0, s) for s in old_shape)

        if self._root_path is not None:
            dat_path = self._root_path / f"pos_{pos_idx}.dat"
            tmp_path = dat_path.with_suffix(".dat.tmp")
            new_arr = np.memmap(tmp_path, dtype=arr.dtype, mode="w+", shape=new_shape_t)
            new_arr[copy_slices] = arr[:]
            new_arr.flush()
            del arr
            self._arrays[pos_idx] = new_arr
            shutil.move(tmp_path, dat_path)
            self._arrays[pos_idx] = np.memmap(
                dat_path, dtype=new_arr.dtype, mode="r+", shape=new_shape_t
            )
        else:
            new_arr = np.zeros(new_shape_t, dtype=arr.dtype)
            new_arr[copy_slices] = arr
            self._arrays[pos_idx] = new_arr

    def _write_manifest(self) -> None:
        """Write manifest.json alongside memmap files."""
        if self._root_path is None:
            return
        self._settings_dump["position_shapes"] = [
            self._logical_shapes[i] for i in range(len(self._arrays))
        ]
        (self._root_path / "manifest.json").write_text(json.dumps(self._settings_dump))


class _MemoryArrayView:
    """Live proxy that always reads from the backend's current array."""

    __slots__ = ("_backend", "_pos_idx")

    def __init__(self, backend: MemoryBackend, pos_idx: int) -> None:
        self._backend = backend
        self._pos_idx = pos_idx

    @property
    def shape(self) -> tuple[int, ...]:
        return tuple(self._backend._logical_shapes[self._pos_idx])

    @property
    def dtype(self) -> np.dtype:
        return self._backend._arrays[self._pos_idx].dtype

    def __getitem__(self, key: Any) -> Any:
        return self._backend._arrays[self._pos_idx][key]
