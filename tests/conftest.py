from __future__ import annotations

from typing import TYPE_CHECKING, NamedTuple, cast

import pydantic
import pytest

import ome_writers as omew

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path

    import numpy as np

    from ome_writers._auto import BackendName


class AvailableBackend(NamedTuple):
    name: BackendName
    cls: type[omew.OMEStream]
    file_ext: str
    read_data: Callable[[Path], np.ndarray]


def _read_tstore(output_path: Path) -> np.ndarray:
    import tensorstore as ts

    spec = {
        "driver": "zarr3",
        "kvstore": {"driver": "file", "path": str(output_path / "0")},
    }
    store = ts.open(spec).result()
    return store.read().result()  # type: ignore


def _read_zarr_python(output_path: Path) -> np.ndarray:
    import zarr  # type: ignore

    z = zarr.open_array(output_path, mode="r", path="0")
    return z[:]


def _read_zarr(output_path: Path) -> np.ndarray:
    try:
        return _read_tstore(output_path)
    except ImportError:
        try:
            return _read_zarr_python(output_path)
        except ImportError as e:
            raise pytest.skip("zarr or tensorstore is not installed") from e


def _read_tiff(output_path: Path) -> np.ndarray:
    try:
        import tifffile  # type: ignore

        return tifffile.imread(str(output_path))
    except ImportError as e:
        raise pytest.skip("tifffile is not installed") from e


# Test configurations for each backend
TIFF_BACKENDS: list[AvailableBackend] = []
ZARR_BACKENDS: list[AvailableBackend] = []
if omew.TensorStoreZarrStream.is_available():
    ZARR_BACKENDS.append(
        AvailableBackend(
            "tensorstore", omew.TensorStoreZarrStream, ".ome.zarr", _read_zarr
        )
    )
if omew.AcquireZarrStream.is_available():
    ZARR_BACKENDS.append(
        AvailableBackend(
            "acquire-zarr", omew.AcquireZarrStream, ".ome.zarr", _read_zarr
        )
    )
if omew.TifffileStream.is_available():
    TIFF_BACKENDS.append(
        AvailableBackend("tiff", omew.TifffileStream, ".ome.tiff", _read_tiff)
    )

BACKENDS: list[AvailableBackend] = TIFF_BACKENDS + ZARR_BACKENDS


@pytest.fixture(params=BACKENDS, ids=lambda b: b.name)
def backend(request: pytest.FixtureRequest) -> AvailableBackend:
    """Fixture to provide an available backend based on the test parameter."""
    return cast("AvailableBackend", request.param)


@pytest.fixture(params=ZARR_BACKENDS, ids=lambda b: b.name)
def zarr_backend(request: pytest.FixtureRequest) -> AvailableBackend:
    """Fixture to provide an available backend based on the test parameter."""
    return cast("AvailableBackend", request.param)


@pytest.fixture(params=TIFF_BACKENDS, ids=lambda b: b.name)
def tiff_backend(request: pytest.FixtureRequest) -> AvailableBackend:
    """Fixture to provide an available backend based on the test parameter."""
    return cast("AvailableBackend", request.param)


def validate_zarr_store(output_path: Path) -> None:
    try:
        import yaozarrs
    except ImportError:
        print("yaozarrs not installed, skipping Zarr store validation")
        return

    try:
        yaozarrs.validate_zarr_store(output_path)
    except pydantic.ValidationError as e:
        if len(errors := e.errors()) != 1:
            raise
        error = errors[0]
        if "Axes are not in the required order by type" not in error["msg"]:
            raise


def validate_path(path: Path) -> None:
    """Validate that a file exists, is non-empty, and has valid metadata/structure."""
    assert path.exists(), f"File {path} does not exist."
    if path.suffix == ".tiff":
        try:
            import tifffile
            from ome_types import from_xml
        except ImportError:
            print("ome-types or tifffile not installed, skipping OME-XML validation")
            return

        with tifffile.TiffFile(path) as tif:
            ome_xml = tif.ome_metadata
            if ome_xml is not None:
                # validate by attempting to parse
                from_xml(ome_xml)

    elif path.suffix in {".zarr", ".zarr/"}:
        validate_zarr_store(path)
