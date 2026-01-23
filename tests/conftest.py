import pytest

from ome_writers import _stream

ZARR_BACKENDS = []
TIFF_BACKENDS = []
for name, meta in _stream.AVAILABLE_BACKENDS.items():
    if meta.format == "zarr":
        ZARR_BACKENDS.append(name)
    elif meta.format == "tiff":
        TIFF_BACKENDS.append(name)
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
