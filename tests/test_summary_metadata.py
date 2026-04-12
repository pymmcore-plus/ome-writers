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
    OMEStream,
    Plate,
    Position,
    create_stream,
)

if TYPE_CHECKING:
    from pathlib import Path


# ---------------------------------------------------------------------------
# Zarr backend tests
# ---------------------------------------------------------------------------

pytest_zarr = pytest.importorskip("zarr", reason="zarr not available")


def _write_a_frame(stream: OMEStream, shape: tuple[int, int] = (16, 16)) -> None:
    stream.append(np.random.randint(0, 100, shape, dtype=np.uint16))


def test_zarr_summary_metadata_single_position(
    tmp_path: Path, zarr_backend: str
) -> None:
    """Single-position Zarr writes global metadata to root zarr.json."""
    root = tmp_path / "single.zarr"
    settings = AcquisitionSettings(
        root_path=str(root),
        dimensions=[
            Dimension(name="c", count=2, type="channel"),
            Dimension(name="y", count=16, chunk_size=16, type="space"),
            Dimension(name="x", count=16, chunk_size=16, type="space"),
        ],
        dtype="uint16",
        overwrite=True,
        format={"name": "ome-zarr", "backend": zarr_backend},
    )

    summary = {"a": 1, "nested": {"b": [2, 3], "c": "hello"}}
    with create_stream(settings) as stream:
        stream.set_global_metadata("pymmcore_plus", summary)
        for _ in range(2):
            _write_a_frame(stream)

    data = json.loads((root / "zarr.json").read_text())
    attrs = data["attributes"]

    assert attrs["pymmcore_plus"] == summary
    # ome-writers namespaces are untouched
    assert "ome" in attrs
    assert "multiscales" in attrs["ome"]


def test_zarr_summary_metadata_multiposition(tmp_path: Path, zarr_backend: str) -> None:
    """Multi-position Zarr writes global metadata in one place only."""
    root = tmp_path / "multi.zarr"
    settings = AcquisitionSettings(
        root_path=str(root),
        dimensions=[
            Dimension(name="p", type="position", coords=["Pos0", "Pos1"]),
            Dimension(name="c", count=1, type="channel"),
            Dimension(name="y", count=16, chunk_size=16, type="space"),
            Dimension(name="x", count=16, chunk_size=16, type="space"),
        ],
        dtype="uint16",
        overwrite=True,
        format={"name": "ome-zarr", "backend": zarr_backend},
    )

    summary = {"mda_sequence": {"positions": ["Pos0", "Pos1"]}}
    with create_stream(settings) as stream:
        stream.set_global_metadata("pymmcore_plus", summary)
        for _ in range(2):
            _write_a_frame(stream)

    # Summary lives under the parent root group's zarr.json
    root_data = json.loads((root / "zarr.json").read_text())
    assert root_data["attributes"]["pymmcore_plus"] == summary

    # Per-position zarr.json files must not carry the namespace
    for pos_name in ("Pos0", "Pos1"):
        pos_data = json.loads((root / pos_name / "zarr.json").read_text())
        assert "pymmcore_plus" not in pos_data["attributes"]


def test_zarr_summary_metadata_plate(tmp_path: Path, zarr_backend: str) -> None:
    """Plate Zarr writes global metadata to the plate root zarr.json once."""
    root = tmp_path / "plate.zarr"
    settings = AcquisitionSettings(
        root_path=str(root),
        dimensions=[
            Dimension(
                name="p",
                type="position",
                coords=[
                    Position(name="field0", plate_row="A", plate_column="1"),
                    Position(name="field0", plate_row="A", plate_column="2"),
                ],
            ),
            Dimension(name="y", count=16, chunk_size=16, type="space"),
            Dimension(name="x", count=16, chunk_size=16, type="space"),
        ],
        dtype="uint16",
        overwrite=True,
        format={"name": "ome-zarr", "backend": zarr_backend},
        plate=Plate(name="p", row_names=["A"], column_names=["1", "2"]),
    )

    summary = {"experiment": "plate test"}
    with create_stream(settings) as stream:
        stream.set_global_metadata("pymmcore_plus", summary)
        for _ in range(2):
            _write_a_frame(stream)

    root_data = json.loads((root / "zarr.json").read_text())
    assert root_data["attributes"]["pymmcore_plus"] == summary

    # Nothing leaks into well/field groups.
    for col in ("1", "2"):
        well_json = root / "A" / col / "field0" / "zarr.json"
        well_data = json.loads(well_json.read_text())
        assert "pymmcore_plus" not in well_data.get("attributes", {})


def test_zarr_summary_metadata_replace(tmp_path: Path, zarr_backend: str) -> None:
    """Same namespace replaces rather than merges."""
    root = tmp_path / "replace.zarr"
    settings = AcquisitionSettings(
        root_path=str(root),
        dimensions=[
            Dimension(name="y", count=16, chunk_size=16, type="space"),
            Dimension(name="x", count=16, chunk_size=16, type="space"),
        ],
        dtype="uint16",
        overwrite=True,
        format={"name": "ome-zarr", "backend": zarr_backend},
    )

    with create_stream(settings) as stream:
        stream.set_global_metadata("ns", {"first": True, "shared": 1})
        stream.set_global_metadata("ns", {"second": True})
        _write_a_frame(stream)

    attrs = json.loads((root / "zarr.json").read_text())["attributes"]
    # second call replaced the first — no merge
    assert attrs["ns"] == {"second": True}


def test_zarr_summary_metadata_siblings(tmp_path: Path, zarr_backend: str) -> None:
    """Different namespaces coexist as siblings."""
    root = tmp_path / "siblings.zarr"
    settings = AcquisitionSettings(
        root_path=str(root),
        dimensions=[
            Dimension(name="y", count=16, chunk_size=16, type="space"),
            Dimension(name="x", count=16, chunk_size=16, type="space"),
        ],
        dtype="uint16",
        overwrite=True,
        format={"name": "ome-zarr", "backend": zarr_backend},
    )

    with create_stream(settings) as stream:
        stream.set_global_metadata("ns_a", {"x": 1})
        stream.set_global_metadata("ns_b", {"y": 2})
        _write_a_frame(stream)

    attrs = json.loads((root / "zarr.json").read_text())["attributes"]
    assert attrs["ns_a"] == {"x": 1}
    assert attrs["ns_b"] == {"y": 2}
    # ome / ome_writers namespaces still intact
    assert "ome" in attrs
    assert "ome_writers" in attrs


def test_summary_metadata_reserved_namespace(tmp_path: Path, zarr_backend: str) -> None:
    settings = AcquisitionSettings(
        root_path=str(tmp_path / "reserved.zarr"),
        dimensions=[
            Dimension(name="y", count=16, chunk_size=16, type="space"),
            Dimension(name="x", count=16, chunk_size=16, type="space"),
        ],
        dtype="uint16",
        overwrite=True,
        format={"name": "ome-zarr", "backend": zarr_backend},
    )

    with create_stream(settings) as stream:
        with pytest.raises(ValueError, match="reserved"):
            stream.set_global_metadata("ome", {})
        with pytest.raises(ValueError, match="reserved"):
            stream.set_global_metadata("ome_writers", {})
        with pytest.raises(ValueError, match="non-empty"):
            stream.set_global_metadata("", {})
        _write_a_frame(stream)


# ---------------------------------------------------------------------------
# TIFF backend tests
# ---------------------------------------------------------------------------

try:
    from ome_types import from_tiff, from_xml
except ImportError:
    pytest.skip("ome_types not installed", allow_module_level=True)


def _get_map_annotations(ome_obj: Any) -> list[Any]:
    sa = ome_obj.structured_annotations
    if sa is None:
        return []
    return list(sa.map_annotations)


def _decode_summary(annotation: Any) -> dict:
    """Return decoded summary payload from a MapAnnotation."""
    for entry in annotation.value.ms:
        if entry.k == "data_json":
            return json.loads(entry.value)
    raise AssertionError("no data_json key in MapAnnotation.value")


def test_tiff_summary_metadata_single_file(tmp_path: Path, tiff_backend: str) -> None:
    """Single-file OME-TIFF stores a single MapAnnotation at OME root."""
    path = tmp_path / "single.ome.tiff"
    settings = AcquisitionSettings(
        root_path=str(path),
        dimensions=[
            Dimension(name="t", count=2, type="time"),
            Dimension(name="c", count=1, type="channel"),
            Dimension(name="y", count=16, type="space"),
            Dimension(name="x", count=16, type="space"),
        ],
        dtype="uint16",
        format={"name": "ome-tiff", "backend": tiff_backend},
    )

    summary = {"mda": {"positions": 1}, "note": "single"}
    with create_stream(settings) as stream:
        stream.set_global_metadata("pymmcore_plus", summary)
        for _ in range(2):
            stream.append(np.zeros((16, 16), dtype=np.uint16))

    ome_obj = from_tiff(str(path))
    annotations = _get_map_annotations(ome_obj)
    summary_anns = [a for a in annotations if a.namespace == "pymmcore_plus"]
    assert len(summary_anns) == 1
    assert _decode_summary(summary_anns[0]) == summary

    # Not referenced by any plane.
    ann_id = summary_anns[0].id
    for image in ome_obj.images:
        for plane in image.pixels.planes:
            assert all(ref.id != ann_id for ref in plane.annotation_refs)


def test_tiff_summary_metadata_companion_file(
    tmp_path: Path, tiff_backend: str
) -> None:
    """Companion-file mode writes MapAnnotation to companion only."""
    multipos_dir = tmp_path / "companion"
    settings = AcquisitionSettings(
        root_path=str(multipos_dir),
        dimensions=[
            Dimension(name="p", type="position", coords=["Pos0", "Pos1"]),
            Dimension(name="t", count=2, type="time"),
            Dimension(name="y", count=16, type="space"),
            Dimension(name="x", count=16, type="space"),
        ],
        dtype="uint16",
        format={
            "name": "ome-tiff",
            "backend": tiff_backend,
            "multi_file_metadata": "companion-file",
        },
    )

    summary = {"mda": "companion"}
    with create_stream(settings) as stream:
        stream.set_global_metadata("pymmcore_plus", summary)
        for _ in range(4):
            stream.append(np.zeros((16, 16), dtype=np.uint16))

    companion_path = multipos_dir / "companion.ome"
    assert companion_path.exists()
    ome_obj = from_xml(companion_path.read_text())
    annotations = _get_map_annotations(ome_obj)
    matches = [a for a in annotations if a.namespace == "pymmcore_plus"]
    assert len(matches) == 1
    assert _decode_summary(matches[0]) == summary

    # No TIFF file should carry the summary namespace annotation.
    for pos_idx in range(2):
        tiff_path = multipos_dir / f"companion_p{pos_idx:03d}.ome.tiff"
        ome_in_tiff = from_tiff(str(tiff_path))
        for a in _get_map_annotations(ome_in_tiff):
            assert a.namespace != "pymmcore_plus"


def test_tiff_summary_metadata_master_tiff(tmp_path: Path, tiff_backend: str) -> None:
    """Master-tiff mode writes MapAnnotation to master file only."""
    multipos_dir = tmp_path / "master"
    settings = AcquisitionSettings(
        root_path=str(multipos_dir),
        dimensions=[
            Dimension(name="p", type="position", coords=["Pos0", "Pos1"]),
            Dimension(name="t", count=2, type="time"),
            Dimension(name="y", count=16, type="space"),
            Dimension(name="x", count=16, type="space"),
        ],
        dtype="uint16",
        format={
            "name": "ome-tiff",
            "backend": tiff_backend,
            "multi_file_metadata": "master-tiff",
        },
    )

    summary = {"mda": "master"}
    with create_stream(settings) as stream:
        stream.set_global_metadata("pymmcore_plus", summary)
        for _ in range(4):
            stream.append(np.zeros((16, 16), dtype=np.uint16))

    master_path = multipos_dir / "master_p000.ome.tiff"
    other_path = multipos_dir / "master_p001.ome.tiff"
    assert master_path.exists() and other_path.exists()

    master_ome = from_tiff(str(master_path))
    matches = [
        a for a in _get_map_annotations(master_ome) if a.namespace == "pymmcore_plus"
    ]
    assert len(matches) == 1
    assert _decode_summary(matches[0]) == summary

    other_ome = from_tiff(str(other_path))
    for a in _get_map_annotations(other_ome):
        assert a.namespace != "pymmcore_plus"


def test_tiff_summary_metadata_redundant_fans_out(
    tmp_path: Path, tiff_backend: str
) -> None:
    """Redundant mode writes a copy of the annotation into every file."""
    multipos_dir = tmp_path / "redundant"
    settings = AcquisitionSettings(
        root_path=str(multipos_dir),
        dimensions=[
            Dimension(name="p", type="position", coords=["Pos0", "Pos1"]),
            Dimension(name="t", count=2, type="time"),
            Dimension(name="y", count=16, type="space"),
            Dimension(name="x", count=16, type="space"),
        ],
        dtype="uint16",
        format={
            "name": "ome-tiff",
            "backend": tiff_backend,
            "multi_file_metadata": "redundant",
        },
    )

    summary = {"mda": "redundant", "note": "every file"}
    with create_stream(settings) as stream:
        stream.set_global_metadata("pymmcore_plus", summary)
        for _ in range(4):
            stream.append(np.zeros((16, 16), dtype=np.uint16))

    for pos_idx in range(2):
        tiff_path = multipos_dir / f"redundant_p{pos_idx:03d}.ome.tiff"
        ome_obj = from_tiff(str(tiff_path))
        matches = [
            a for a in _get_map_annotations(ome_obj) if a.namespace == "pymmcore_plus"
        ]
        assert len(matches) == 1, f"pos {pos_idx} missing summary annotation"
        assert _decode_summary(matches[0]) == summary

        # Not referenced by any plane.
        ann_id = matches[0].id
        for image in ome_obj.images:
            for plane in image.pixels.planes:
                assert all(ref.id != ann_id for ref in plane.annotation_refs)


def test_tiff_summary_metadata_replace(tmp_path: Path, tiff_backend: str) -> None:
    """Same namespace: second call replaces the first."""
    path = tmp_path / "replace.ome.tiff"
    settings = AcquisitionSettings(
        root_path=str(path),
        dimensions=[
            Dimension(name="y", count=16, type="space"),
            Dimension(name="x", count=16, type="space"),
        ],
        dtype="uint16",
        format={"name": "ome-tiff", "backend": tiff_backend},
    )

    with create_stream(settings) as stream:
        stream.set_global_metadata("ns", {"first": True})
        stream.set_global_metadata("ns", {"second": True})
        stream.append(np.zeros((16, 16), dtype=np.uint16))

    ome_obj = from_tiff(str(path))
    matches = [a for a in _get_map_annotations(ome_obj) if a.namespace == "ns"]
    assert len(matches) == 1
    assert _decode_summary(matches[0]) == {"second": True}


def test_tiff_summary_metadata_siblings(tmp_path: Path, tiff_backend: str) -> None:
    """Different namespaces: two distinct MapAnnotations."""
    path = tmp_path / "siblings.ome.tiff"
    settings = AcquisitionSettings(
        root_path=str(path),
        dimensions=[
            Dimension(name="y", count=16, type="space"),
            Dimension(name="x", count=16, type="space"),
        ],
        dtype="uint16",
        format={"name": "ome-tiff", "backend": tiff_backend},
    )

    with create_stream(settings) as stream:
        stream.set_global_metadata("ns_a", {"x": 1})
        stream.set_global_metadata("ns_b", {"y": 2})
        stream.append(np.zeros((16, 16), dtype=np.uint16))

    ome_obj = from_tiff(str(path))
    namespaces = {
        a.namespace
        for a in _get_map_annotations(ome_obj)
        if a.namespace in ("ns_a", "ns_b")
    }
    assert namespaces == {"ns_a", "ns_b"}


def test_tiff_summary_metadata_concurrent_with_close(
    tmp_path: Path, tiff_backend: str
) -> None:
    """Concurrent summary updates and close should not crash."""
    path = tmp_path / "concurrent_close.ome.tiff"
    settings = AcquisitionSettings(
        root_path=str(path),
        dimensions=[
            Dimension(name="t", count=16, type="time"),
            Dimension(name="y", count=16, type="space"),
            Dimension(name="x", count=16, type="space"),
        ],
        dtype="uint16",
        format={"name": "ome-tiff", "backend": tiff_backend},
    )

    errors: list[BaseException] = []
    stop = threading.Event()

    stream = create_stream(settings)
    stream.set_global_metadata("ns", {"i": 0})

    def _set_summary_loop() -> None:
        i = 1
        while not stop.is_set():
            try:
                stream.set_global_metadata("ns", {"i": i})
            except RuntimeError:
                # Expected once close/finalize has completed.
                break
            except BaseException as e:
                errors.append(e)
                break
            i += 1

    worker = threading.Thread(target=_set_summary_loop)
    worker.start()
    try:
        for _ in range(8):
            stream.append(np.zeros((16, 16), dtype=np.uint16))
    finally:
        stream.close()
        stop.set()
        worker.join(timeout=2)

    assert not errors
    ome_obj = from_tiff(str(path))
    matches = [a for a in _get_map_annotations(ome_obj) if a.namespace == "ns"]
    assert len(matches) == 1
