"""Common utilities shared between benchmark and profiling tools."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from ome_writers import (
    AcquisitionSettings,
    Dimension,
    PositionDimension,
    dims_from_standard_axes,
)


def parse_dimensions(dim_spec: str) -> list[Dimension | PositionDimension]:
    """Parse compact dimension specification.

    Format: name:count[:chunk[:shard]]
    Example: t:10:1,c:3,z:5,y:512:64,x:512:64
    """
    sizes: dict[str, int] = {}
    chunk_shapes: dict[str, int] = {}
    shard_shapes: dict[str, int | None] = {}
    for spec in dim_spec.split(","):
        if len(parts := spec.split(":")) < 2:
            raise ValueError(f"Invalid dimension spec: {spec} (need name:count)")

        name = parts[0]
        sizes[name] = int(parts[1])
        chunk_shapes[name] = int(parts[2]) if len(parts) > 2 else 1
        shard_shapes[name] = int(parts[3]) if len(parts) > 3 else None

    return dims_from_standard_axes(sizes, chunk_shapes, shard_shapes)


def parse_settings_file(
    file_path: Path, dtype: str, compression: str | None
) -> AcquisitionSettings:
    """Parse settings from a JSON file."""
    data = json.loads(Path(file_path).expanduser().resolve().read_text())
    if not isinstance(data, dict):
        raise ValueError("Settings file must contain a single JSON object")

    data["root_path"] = "tmp"
    data.setdefault("dtype", dtype)
    data.setdefault("compression", compression)
    settings = AcquisitionSettings.model_validate(data)
    for d in settings.dimensions:
        if isinstance(d, Dimension):
            d.chunk_size = d.chunk_size or 1
        if d.count is None:
            raise NotImplementedError(
                "Unbounded dimensions not supported in benchmark\n"
                f"Please modify dimension {d.name!r} in settings file {file_path}"
            )
    return settings


def generate_frames(settings: AcquisitionSettings) -> list[np.ndarray]:
    """Generate random frames based on acquisition settings."""
    dtype = np.dtype(settings.dtype)
    iinfo = np.iinfo(dtype)
    size = (settings.num_frames or 1, *settings.shape[-2:])
    return list(np.random.randint(iinfo.min, iinfo.max, size=size, dtype=dtype))
