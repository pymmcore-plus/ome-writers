import importlib.util
import sys

import pytest

ZARR_BACKENDS = []
TIFF_BACKENDS = []
if importlib.util.find_spec("tensorstore") is not None:
    ZARR_BACKENDS.append("tensorstore")
if importlib.util.find_spec("zarr") is not None and sys.version_info >= (3, 11):
    ZARR_BACKENDS.append("zarr-python")
if importlib.util.find_spec("acquire_zarr") is not None:
    ZARR_BACKENDS.append("acquire-zarr")
if importlib.util.find_spec("tifffile") is not None:
    TIFF_BACKENDS = ["tiff"]

AVAILABLE_BACKENDS = ZARR_BACKENDS + TIFF_BACKENDS


@pytest.fixture(params=AVAILABLE_BACKENDS)
def any_backend(request: pytest.FixtureRequest) -> str:
    """Fixture to parametrize tests over available backends."""
    return request.param


@pytest.fixture(params=ZARR_BACKENDS)
def zarr_backend(request: pytest.FixtureRequest) -> str:
    """Fixture to parametrize tests over available Zarr backends."""
    return request.param


@pytest.fixture(params=TIFF_BACKENDS)
def tiff_backend(request: pytest.FixtureRequest) -> str:
    """Fixture to parametrize tests over available TIFF backends."""
    return request.param
