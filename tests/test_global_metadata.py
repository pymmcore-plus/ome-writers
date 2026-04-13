from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

import numpy as np
import pytest

from ome_writers import AcquisitionSettings, Dimension, Plate, Position, create_stream

if TYPE_CHECKING:
    from pathlib import Path


XY = [
    Dimension(name="y", count=16, chunk_size=16, type="space"),
    Dimension(name="x", count=16, chunk_size=16, type="space"),
]
FRAME = np.zeros((16, 16), dtype=np.uint16)


def _settings(
    root: Path,
    fmt: dict[str, Any],
    dims: list[Dimension] | None = None,
    **extra: Any,
) -> AcquisitionSettings:
    return AcquisitionSettings(
        root_path=str(root),
        dimensions=dims if dims is not None else XY,
        dtype="uint16",
        overwrite=True,
        format=fmt,
        **extra,
    )


# ---------------------------------------------------------------------------
# Zarr backend
# ---------------------------------------------------------------------------


def _zarr_attrs(path: Path) -> dict:
    return json.loads((path / "zarr.json").read_text()).get("attributes", {})


def test_zarr_global_metadata_single_position(
    tmp_path: Path, zarr_backend: str
) -> None:
    """Single-pos zarr: nested values, replace, siblings, untouched ns."""
    root = tmp_path / "single.zarr"
    fmt = {"name": "ome-zarr", "backend": zarr_backend}
    dims = [Dimension(name="c", count=2, type="channel"), *XY]
    nested = {"a": 1, "nested": {"b": [2, 3], "c": "hello"}}

    with create_stream(_settings(root, fmt, dims)) as stream:
        stream.set_global_metadata("ns_a", {"first": True, "shared": 1})
        stream.set_global_metadata("ns_a", nested)  # replace
        stream.set_global_metadata("ns_b", {"y": 2})  # sibling
        stream.append(FRAME)

    attrs = _zarr_attrs(root)
    assert attrs["ns_a"] == nested  # replaced, not merged
    assert attrs["ns_b"] == {"y": 2}
    assert "ome" in attrs and "multiscales" in attrs["ome"]
    assert "ome_writers" in attrs


def test_zarr_global_metadata_multiposition(tmp_path: Path, zarr_backend: str) -> None:
    """Multi-pos zarr writes metadata at the root only."""
    root = tmp_path / "multi.zarr"
    fmt = {"name": "ome-zarr", "backend": zarr_backend}
    dims = [Dimension(name="p", type="position", coords=["Pos0", "Pos1"]), *XY]
    summary = {"mda_sequence": {"positions": ["Pos0", "Pos1"]}}

    with create_stream(_settings(root, fmt, dims)) as stream:
        stream.set_global_metadata("ns", summary)
        stream.append(FRAME)
        stream.append(FRAME)

    assert _zarr_attrs(root)["ns"] == summary
    for pos in ("Pos0", "Pos1"):
        assert "ns" not in _zarr_attrs(root / pos)


def test_zarr_global_metadata_plate(tmp_path: Path, zarr_backend: str) -> None:
    """Plate zarr writes metadata at the plate root only."""
    root = tmp_path / "plate.zarr"
    fmt = {"name": "ome-zarr", "backend": zarr_backend}
    dims = [
        Dimension(
            name="p",
            type="position",
            coords=[
                Position(name="field0", plate_row="A", plate_column="1"),
                Position(name="field0", plate_row="A", plate_column="2"),
            ],
        ),
        *XY,
    ]
    plate = Plate(name="p", row_names=["A"], column_names=["1", "2"])
    summary = {"experiment": "plate test"}

    with create_stream(_settings(root, fmt, dims, plate=plate)) as stream:
        stream.set_global_metadata("ns", summary)
        stream.append(FRAME)
        stream.append(FRAME)

    assert _zarr_attrs(root)["ns"] == summary
    for col in ("1", "2"):
        assert "ns" not in _zarr_attrs(root / "A" / col / "field0")


def test_zarr_single_position_root_mirror_aliased(
    tmp_path: Path, zarr_backend: str
) -> None:
    """In single-position layouts the parent root group *is* the image group,
    so `_root_meta_mirror` must be the same object as `_meta_mirrors["."]`.

    This invariant is load-bearing: `set_global_metadata` flushes through
    `_root_meta_mirror`, while `finalize()` flushes per-image mirrors via
    `_meta_mirrors`. If a future refactor opens a second, distinct mirror
    over the same file for single-position mode, `finalize()` would clobber
    whatever `set_global_metadata` wrote (last-writer-wins).
    """
    root = tmp_path / "aliased.zarr"
    fmt = {"name": "ome-zarr", "backend": zarr_backend}

    with create_stream(_settings(root, fmt)) as stream:
        backend = stream._backend  # type: ignore[attr-defined]
        assert backend._image_group_paths == ["."]
        assert backend._root_meta_mirror is backend._meta_mirrors["."]
        stream.append(FRAME)


def test_global_metadata_invalid_namespace(tmp_path: Path, zarr_backend: str) -> None:
    """Reserved and empty namespaces raise ValueError."""
    fmt = {"name": "ome-zarr", "backend": zarr_backend}
    with create_stream(_settings(tmp_path / "reserved.zarr", fmt)) as stream:
        for ns in ("ome", "ome_writers"):
            with pytest.raises(ValueError, match="reserved"):
                stream.set_global_metadata(ns, {})
        with pytest.raises(ValueError, match="non-empty"):
            stream.set_global_metadata("", {})
        stream.append(FRAME)
