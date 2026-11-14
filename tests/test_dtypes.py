"""Writing every numpy dtype a format supports, and rejecting the rest early."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import numpy as np
import pytest

from ome_writers import AcquisitionSettings, Dimension, _stream, create_stream
from tests._utils import read_array_data, wait_for_frames

if TYPE_CHECKING:
    from pathlib import Path

DTYPES = [
    "bool",
    "int8",
    "uint8",
    "int16",
    "uint16",
    "int32",
    "uint32",
    "int64",
    "uint64",
    "float16",
    "float32",
    "float64",
    "complex64",
    "complex128",
]
# OME-XML has no pixel type for 64-bit ints or float16, Bio-Formats can't read
# complex pixels, and tifffile's contiguous mode drops bool frames after the first.
OME_TIFF_UNSUPPORTED = [
    "bool",
    "int64",
    "uint64",
    "float16",
    "complex64",
    "complex128",
]
ACQUIRE_ZARR_UNSUPPORTED = ["bool", "float16", "complex64", "complex128"]
EXT = {b.name: f".{b.format.replace('-', '.')}" for b in _stream.BACKENDS}


def _settings(path: Path, dtype: str, backend: str = "auto") -> AcquisitionSettings:
    return AcquisitionSettings(
        root_path=path,
        dimensions=[
            # named channels, so OME-Zarr writes omero display windows
            Dimension(name="c", type="channel", coords=["a", "b", "c"]),
            Dimension(name="y", count=4, type="space"),
            Dimension(name="x", count=6, type="space"),
        ],
        dtype=dtype,
        format=backend,
    )


def _frames(dtype: str) -> np.ndarray:
    data = np.arange(3 * 4 * 6).reshape(3, 4, 6)
    if dtype == "bool":
        return (data % 3 == 0).astype(dtype)
    out = (data - 30).astype(dtype) if dtype.startswith("int") else data.astype(dtype)
    if out.dtype.kind == "c":
        out = out + 1j * out[::-1]
    return out


def _is_unsupported(backend: str, dtype: str) -> bool:
    if backend == "tifffile":
        return dtype in OME_TIFF_UNSUPPORTED
    return backend == "acquire-zarr" and dtype in ACQUIRE_ZARR_UNSUPPORTED


@pytest.mark.parametrize("dtype", DTYPES)
def test_dtype_roundtrip(tmp_path: Path, any_backend: str, dtype: str) -> None:
    if _is_unsupported(any_backend, dtype):
        pytest.skip(f"{any_backend} can't write {dtype}")

    path = tmp_path / f"out{EXT[any_backend]}"
    data = _frames(dtype)
    with create_stream(_settings(path, dtype, any_backend)) as stream:
        for frame in data:
            stream.append(frame)
        if any_backend != "acquire-zarr":  # no read-only views
            wait_for_frames(stream._backend, expected_count=len(data))
            live = np.asarray(stream.view(dynamic_shape=False))
            assert live.dtype == data.dtype
            np.testing.assert_array_equal(live, data)

    out = read_array_data(path)
    assert out.dtype == data.dtype
    np.testing.assert_array_equal(out, data)


@pytest.mark.parametrize("dtype", OME_TIFF_UNSUPPORTED)
def test_ome_tiff_rejects_dtype(tmp_path: Path, dtype: str) -> None:
    with pytest.raises(ValueError, match="not supported for OME-TIFF"):
        _settings(tmp_path / "out.ome.tiff", dtype)


@pytest.mark.parametrize("dtype", ACQUIRE_ZARR_UNSUPPORTED)
def test_acquire_zarr_rejects_dtype(tmp_path: Path, dtype: str) -> None:
    if "acquire-zarr" not in _stream.AVAILABLE_BACKENDS:
        pytest.skip("acquire-zarr not installed")
    settings = _settings(tmp_path / "out.ome.zarr", dtype, "acquire-zarr")
    with pytest.raises(
        NotImplementedError, match=f"data type '{dtype}' is not supported"
    ):
        create_stream(settings)
    assert not (tmp_path / "out.ome.zarr").exists()


@pytest.mark.parametrize(
    ("dtype", "window"),
    [
        ("uint8", (0, 255)),
        ("int16", (-32768, 32767)),
        ("bool", (0, 1)),
        ("float32", (0, 1)),
        ("complex64", (0, 1)),
    ],
)
def test_omero_window_by_dtype(
    tmp_path: Path, zarr_backend: str, dtype: str, window: tuple[int, int]
) -> None:
    if _is_unsupported(zarr_backend, dtype):
        pytest.skip(f"{zarr_backend} can't write {dtype}")
    path = tmp_path / "out.ome.zarr"
    with create_stream(_settings(path, dtype, zarr_backend)) as stream:
        for frame in _frames(dtype):
            stream.append(frame)

    attrs = json.loads((path / "zarr.json").read_text())["attributes"]
    for channel in attrs["ome"]["omero"]["channels"]:
        w = channel["window"]
        assert (w["min"], w["max"], w["start"], w["end"]) == (*window, *window)


@pytest.mark.parametrize(
    "dtype", [d for d in DTYPES if d not in OME_TIFF_UNSUPPORTED], ids=str
)
def test_bioformats_reads_dtype(tmp_path: Path, dtype: str) -> None:
    bf_reader = pytest.importorskip("tests._bf_reader")
    path = tmp_path / "out.ome.tiff"
    data = _frames(dtype)
    with create_stream(_settings(path, dtype, "tifffile")) as stream:
        for frame in data:
            stream.append(frame)

    _, (series,) = bf_reader.read_series_with_bioformats(str(path))
    assert series.dtype == data.dtype
    np.testing.assert_array_equal(series, data)


@pytest.mark.parametrize("dtype", ["datetime64[ns]", "timedelta64[s]", "object"])
def test_ome_zarr_rejects_dtype(tmp_path: Path, dtype: str) -> None:
    with pytest.raises(ValueError, match="not supported for OME-Zarr"):
        _settings(tmp_path / "out.ome.zarr", dtype)


def test_ome_tiff_rejects_dtype_left_by_failed_assignment(tmp_path: Path) -> None:
    settings = _settings(tmp_path / "out.ome.tiff", "uint16")
    # pydantic keeps the rejected value on the model
    with pytest.raises(ValueError, match="not supported for OME-TIFF"):
        settings.dtype = "float16"
    with pytest.raises(NotImplementedError, match="not supported for OME-TIFF"):
        create_stream(settings)
    assert not list(tmp_path.iterdir())
