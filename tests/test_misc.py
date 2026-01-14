"""Tests to improve code coverage."""

from __future__ import annotations

import numpy as np

from ome_writers._util import fake_data_for_sizes


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
