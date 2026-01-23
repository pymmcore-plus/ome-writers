import pytest

from ome_writers import _stream

ZARR_BACKENDS = []
TIFF_BACKENDS = []
if _stream._is_tensorstore_available():
    ZARR_BACKENDS.append("tensorstore")
if _stream._is_zarr_available():
    ZARR_BACKENDS.append("zarr-python")
if _stream._is_zarrs_available():
    ZARR_BACKENDS.append("zarrs-python")
if _stream._is_acquire_zarr_available():
    ZARR_BACKENDS.append("acquire-zarr")
if _stream._is_tifffile_available():
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
