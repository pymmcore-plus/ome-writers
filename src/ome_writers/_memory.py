from __future__ import annotations

import ctypes
import math
import os
import sys
import warnings
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from ome_writers import AcquisitionSettings

WIN = sys.platform == "win32"


def warn_if_high_memory_usage(settings: AcquisitionSettings) -> None:
    # Skip if not using chunk buffering
    if not any(d.chunk_size and d.chunk_size > 1 for d in settings.index_dimensions):
        return None

    # we only estimate memory on windows for now
    if sys.platform != "win32":
        return None  # pragma: no cover

    if (available_memory := _get_available_memory()) is None:
        return None  # pragma: no cover

    max_chunks = _max_chunks(settings)
    bytes_per_chunk = _bytes_per_chunk(settings)
    concurrent_positions = _concurrent_positions(settings)
    memory_threshold = float(os.getenv("OME_WRITERS_CHUNK_MEMORY_FRACTION", "0.8"))
    required_memory = max_chunks * bytes_per_chunk * concurrent_positions

    if required_memory > available_memory * memory_threshold:
        warnings.warn(
            f"Chunk buffering may use ~{required_memory / 1e9:.1f} GB "
            f"({concurrent_positions} position(s) x {max_chunks} chunks x "
            f"{bytes_per_chunk / 1e6:.1f} MB), exceeding {memory_threshold:.0%} "
            f"of available memory ({available_memory / 1e9:.1f} GB). "
            f"Consider reordering dimensions or reducing chunk sizes.",
            stacklevel=2,
        )


class MEMORYSTATUSEX(ctypes.Structure):
    _fields_ = (
        ("dwLength", ctypes.c_ulong),
        ("dwMemoryLoad", ctypes.c_ulong),
        ("ullTotalPhys", ctypes.c_ulonglong),
        ("ullAvailPhys", ctypes.c_ulonglong),
        ("ullTotalPageFile", ctypes.c_ulonglong),
        ("ullAvailPageFile", ctypes.c_ulonglong),
        ("ullTotalVirtual", ctypes.c_ulonglong),
        ("ullAvailVirtual", ctypes.c_ulonglong),
        ("sullAvailExtendedVirtual", ctypes.c_ulonglong),
    )


def _get_available_memory() -> int | None:
    if not WIN:
        return None

    stat = MEMORYSTATUSEX()
    stat.dwLength = ctypes.sizeof(stat)
    ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(stat))
    return stat.ullAvailPhys


def _max_chunks(settings: AcquisitionSettings) -> int:
    # Max chunks per position
    index_dims = list(settings.index_dimensions)
    storage_dims = list(settings.array_storage_dimensions)
    storage_pos = {d.name: i for i, d in enumerate(storage_dims[:-2])}
    max_chunks = 1
    for stor_dim in storage_dims[:-2]:
        if not stor_dim.chunk_size or stor_dim.chunk_size <= 1:
            continue
        acq_idx = next(
            (i for i, d in enumerate(index_dims) if d.name == stor_dim.name), None
        )
        if acq_idx is None:
            continue  # pragma: no cover
        chunks_from_faster = 1
        for faster_acq_dim in index_dims[acq_idx + 1 :]:
            if faster_acq_dim.count and faster_acq_dim.name in storage_pos:
                faster_storage_dim = storage_dims[storage_pos[faster_acq_dim.name]]
                chunk_size = faster_storage_dim.chunk_size or 1
                chunks_from_faster *= math.ceil(faster_acq_dim.count / chunk_size)
        max_chunks = max(max_chunks, chunks_from_faster)
    return max_chunks


def _bytes_per_chunk(settings: AcquisitionSettings) -> int:
    chunk_shape = tuple(d.chunk_size or 1 for d in settings.index_dimensions)
    frame_shape = tuple(d.count or 1 for d in settings.frame_dimensions)
    bytes_per_pixel = np.dtype(settings.dtype).itemsize
    return int(np.prod(chunk_shape + frame_shape)) * bytes_per_pixel


def _concurrent_positions(settings: AcquisitionSettings) -> int:
    from ome_writers._schema import Dimension

    pos_idx = settings.position_dimension_index
    if pos_idx is not None:
        first_chunked_idx = next(
            (
                i
                for i, d in enumerate(settings.dimensions)
                if isinstance(d, Dimension) and (d.chunk_size or 1) > 1
            ),
            None,
        )
        concurrent_positions = (
            1
            if first_chunked_idx is None or pos_idx < first_chunked_idx
            else len(settings.positions)
        )
    else:
        concurrent_positions = 1
    return concurrent_positions
