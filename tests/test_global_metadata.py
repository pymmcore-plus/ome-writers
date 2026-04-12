"""Tests for OMEStream.set_global_metadata across backends."""

from __future__ import annotations

import json
import threading
from typing import TYPE_CHECKING, Any

import numpy as np
import pytest

from ome_writers import (
    AcquisitionSettings,
    Dimension,
    Plate,
    Position,
    create_stream,
)

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

pytest.importorskip("zarr", reason="zarr not available")


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


# ---------------------------------------------------------------------------
# TIFF backend
# ---------------------------------------------------------------------------

try:
    from ome_types import from_tiff, from_xml
except ImportError:
    pytest.skip("ome_types not installed", allow_module_level=True)


def _anns_by_ns(ome_obj: Any, namespace: str) -> list[Any]:
    sa = ome_obj.structured_annotations
    if sa is None:
        return []
    return [a for a in sa.map_annotations if a.namespace == namespace]


def _decode(annotation: Any) -> dict:
    return {entry.k: json.loads(entry.value) for entry in annotation.value.ms}


def test_tiff_global_metadata_single_file(tmp_path: Path, tiff_backend: str) -> None:
    """Single-file TIFF: replace + siblings + not referenced by any plane."""
    path = tmp_path / "single.ome.tiff"
    fmt = {"name": "ome-tiff", "backend": tiff_backend}
    dims = [
        Dimension(name="t", count=2, type="time"),
        Dimension(name="y", count=16, type="space"),
        Dimension(name="x", count=16, type="space"),
    ]
    summary = {"mda": {"positions": 1}, "note": "single"}

    with create_stream(_settings(path, fmt, dims)) as stream:
        stream.set_global_metadata("ns_a", {"first": True})
        stream.set_global_metadata("ns_a", summary)  # replace
        stream.set_global_metadata("ns_b", {"y": 2})  # sibling
        stream.append(FRAME)
        stream.append(FRAME)

    ome_obj = from_tiff(str(path))
    ns_a = _anns_by_ns(ome_obj, "ns_a")
    ns_b = _anns_by_ns(ome_obj, "ns_b")
    assert len(ns_a) == 1 and _decode(ns_a[0]) == summary
    assert len(ns_b) == 1 and _decode(ns_b[0]) == {"y": 2}

    # Not referenced by any plane.
    ann_ids = {ns_a[0].id, ns_b[0].id}
    for image in ome_obj.images:
        for plane in image.pixels.planes:
            assert all(ref.id not in ann_ids for ref in plane.annotation_refs)


@pytest.mark.parametrize("mode", ["companion-file", "master-tiff", "redundant"])
def test_tiff_global_metadata_multi_file_modes(
    tmp_path: Path, tiff_backend: str, mode: str
) -> None:
    """Each multi-file mode routes the annotation to the correct file(s)."""
    d = tmp_path / mode
    fmt = {"name": "ome-tiff", "backend": tiff_backend, "multi_file_metadata": mode}
    dims = [
        Dimension(name="p", type="position", coords=["Pos0", "Pos1"]),
        Dimension(name="t", count=2, type="time"),
        Dimension(name="y", count=16, type="space"),
        Dimension(name="x", count=16, type="space"),
    ]
    summary = {"mda": mode}

    with create_stream(_settings(d, fmt, dims)) as stream:
        stream.set_global_metadata("ns", summary)
        for _ in range(4):
            stream.append(FRAME)

    per_pos = [
        _anns_by_ns(from_tiff(str(d / f"{mode}_p{p:03d}.ome.tiff")), "ns")
        for p in range(2)
    ]
    if mode == "redundant":
        for anns in per_pos:
            assert len(anns) == 1 and _decode(anns[0]) == summary
    elif mode == "master-tiff":
        assert len(per_pos[0]) == 1 and _decode(per_pos[0][0]) == summary
        assert per_pos[1] == []
    else:  # companion-file
        assert per_pos[0] == [] and per_pos[1] == []
        companion = from_xml((d / "companion.ome").read_text())
        c_anns = _anns_by_ns(companion, "ns")
        assert len(c_anns) == 1 and _decode(c_anns[0]) == summary


def test_tiff_global_metadata_concurrent_with_close(
    tmp_path: Path, tiff_backend: str
) -> None:
    """Concurrent summary updates and close should not crash."""
    path = tmp_path / "concurrent_close.ome.tiff"
    fmt = {"name": "ome-tiff", "backend": tiff_backend}
    dims = [
        Dimension(name="t", count=16, type="time"),
        Dimension(name="y", count=16, type="space"),
        Dimension(name="x", count=16, type="space"),
    ]
    errors: list[BaseException] = []
    stop = threading.Event()

    stream = create_stream(_settings(path, fmt, dims))
    stream.set_global_metadata("ns", {"i": 0})

    def _loop() -> None:
        i = 1
        while not stop.is_set():
            try:
                stream.set_global_metadata("ns", {"i": i})
            except RuntimeError:  # expected once close finalizes
                break
            except BaseException as e:
                errors.append(e)
                break
            i += 1

    worker = threading.Thread(target=_loop)
    worker.start()
    try:
        for _ in range(8):
            stream.append(FRAME)
    finally:
        stream.close()
        stop.set()
        worker.join(timeout=2)

    assert not errors
    matches = _anns_by_ns(from_tiff(str(path)), "ns")
    assert len(matches) == 1
