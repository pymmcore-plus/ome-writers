"""Tests to improve code coverage."""

from __future__ import annotations

import gc
from typing import TYPE_CHECKING

import numpy as np
import pytest

from ome_writers import AcquisitionSettings, Dimension, create_stream
from ome_writers._stream import get_format_for_backend
from ome_writers._util import fake_data_for_sizes

if TYPE_CHECKING:
    from pathlib import Path


def test_fake_data() -> None:
    """Test fake_data_for_sizes with 2D-only image."""
    # 2d only
    data_gen, _dims, dtype = fake_data_for_sizes(sizes={"y": 32, "x": 32})
    frames = list(data_gen)
    assert len(frames) == 1
    assert frames[0].shape == (32, 32)
    assert dtype == np.uint16

    # Complex 6d
    sizes = {"p": 2, "t": 3, "c": 4, "z": 5, "y": 16, "x": 16}
    data_gen, _dims, dtype = fake_data_for_sizes(sizes=sizes, dtype=np.uint8)
    frames = list(data_gen)
    assert len(frames) == 2 * 3 * 4 * 5
    assert frames[0].shape == (16, 16)
    assert dtype == np.uint8


def test_stream_safety(tmp_path: Path, first_backend: str) -> None:
    """Deleting a stream that appended but wasn't closed should warn."""
    suffix = get_format_for_backend(first_backend)
    settings = AcquisitionSettings(
        root_path=tmp_path / f"test.ome.{suffix}",
        dimensions=[
            Dimension(name="t", count=2, type="time"),
            Dimension(name="y", count=32, type="space"),
            Dimension(name="x", count=32, type="space"),
        ],
        dtype="uint16",
        format=first_backend,
        overwrite=True,
    )

    frame = np.zeros((32, 32), dtype=np.uint16)

    # stream appended and deleted without cleanup, should warn
    stream = create_stream(settings)
    stream.append(frame)
    with pytest.warns(UserWarning, match="OMEStream was not closed"):
        del stream
        gc.collect()

    stream2 = create_stream(settings)
    del stream2  # nothing appended, no warning
    gc.collect()

    stream3 = create_stream(settings)
    stream3.append(frame)
    stream3.close()
    del stream3  # close called no warning
    gc.collect()

    with create_stream(settings) as stream4:
        stream4.append(frame)
    del stream4  # context manager used, no warning
    gc.collect()
