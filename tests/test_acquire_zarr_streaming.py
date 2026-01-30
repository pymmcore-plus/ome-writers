from itertools import pairwise
from pathlib import Path

import numpy as np
import pytest

from ome_writers._schema import AcquisitionSettings, Dimension
from ome_writers._stream import create_stream

pytest.importorskip("acquire_zarr", reason="acquire-zarr not installed")


def test_acquire_zarr_full_streaming_support(tmp_path: Path) -> None:
    """Test that our backend abstraction doesn't break cool acquire-zarr features.

    One very nice thing about acquire-zarr's stream is that it makes no assumptions
    about shape of each buffer being passed to `stream.append()`,  C-contiguous
    buffers are simply concatenated according to the dimensionality declared in
    the settings.

    This test ensures that our backend abstraction preserves this behavior.
    """

    settings = AcquisitionSettings(
        root_path=str(tmp_path / "output.zarr"),
        dimensions=[
            Dimension(name="z", count=18, chunk_size=6, unit="um", scale=0.5),
            Dimension(name="y", count=128, chunk_size=64, unit="um", scale=0.1),
            Dimension(name="x", count=128, chunk_size=64, unit="um", scale=0.1),
        ],
        dtype="uint16",
        format="acquire-zarr",
    )

    shape = tuple(d.count or 1 for d in settings.dimensions)
    flat_data = np.arange(np.prod(shape), dtype=settings.dtype)
    # break the data into 10 arbitrary, non-frame/chunk-aligned, somewhat random pieces
    boundaries = [0, 1500, 3000, 5000, 7000, 9000, 12000, 15000, 18000, 22000, None]
    append_bits = [flat_data[start:stop] for start, stop in pairwise(boundaries)]

    with create_stream(settings) as stream:
        for bit in append_bits:
            stream.append(bit)

    output_data = _zarr_array_to_numpy(f"{settings.output_path}/0")
    assert output_data.shape == (18, 128, 128)
    assert output_data.dtype == np.dtype(settings.dtype)
    assert np.array_equal(output_data.flatten(), flat_data)


def _zarr_array_to_numpy(path: str) -> np.ndarray:
    try:
        import tensorstore as ts

        ts_array = ts.open(
            {"driver": "zarr3", "kvstore": {"driver": "file", "path": path}},
            open=True,
        ).result()
        return np.asarray(ts_array.read().result())
    except ImportError:
        import zarr

        return np.asarray(zarr.open_array(path))
