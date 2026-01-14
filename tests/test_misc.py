"""Tests to improve code coverage."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest

from ome_writers._dimensions import Dimension
from ome_writers._util import dims_from_useq, fake_data_for_sizes

if TYPE_CHECKING:
    from pathlib import Path


def test_fake_data_2d_only() -> None:
    """Test fake_data_for_sizes with 2D-only image."""
    data_gen, _dims, dtype = fake_data_for_sizes(sizes={"y": 32, "x": 32})

    frames = list(data_gen)
    assert len(frames) == 1
    assert frames[0].shape == (32, 32)
    assert dtype == np.uint16


def test_dims_from_useq_unsupported_axis() -> None:
    """Test dims_from_useq with unsupported axis type."""
    pytest.importorskip("useq")
    from useq import MDASequence

    # This should work fine with standard axes
    seq = MDASequence(
        time_plan={"interval": 0.1, "loops": 2},  # type: ignore[arg-type]
    )
    dims = dims_from_useq(seq, image_width=32, image_height=32)
    assert len(dims) == 3  # t, y, x


def test_dims_from_useq_invalid_input() -> None:
    """Test dims_from_useq with invalid input."""
    with pytest.raises(ValueError, match=r"seq must be a useq\.MDASequence"):
        dims_from_useq("not a sequence", image_width=32, image_height=32)  # type: ignore[arg-type]
