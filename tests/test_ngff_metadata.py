"""Tests for the _ngff_metadata module."""

from __future__ import annotations

import pytest

from ome_writers._dimensions import Dimension
from ome_writers._ngff_metadata import (
    _ome_axes_scales,
    dim_to_yaozarrs_axis_v05,
    ome_meta_v5,
)


def test_ome_axes_scales() -> None:
    """Test _ome_axes_scales function."""
    dims = [
        Dimension(label="t", size=5, unit=(1.0, "s")),
        Dimension(label="c", size=3),
        Dimension(label="z", size=10, unit=(0.5, "um")),
        Dimension(label="y", size=512, unit=(0.325, "um")),
        Dimension(label="x", size=512, unit=(0.325, "um")),
    ]

    axes, scales = _ome_axes_scales(dims)

    assert len(axes) == 5
    assert len(scales) == 5

    # Check time axis
    assert axes[0] == {"name": "t", "type": "time", "unit": "second"}
    assert scales[0] == 1.0

    # Check channel axis
    assert axes[1] == {"name": "c", "type": "channel", "unit": "unknown"}
    assert scales[1] == 1.0

    # Check space axes
    assert axes[2] == {"name": "z", "type": "space", "unit": "micrometer"}
    assert scales[2] == 0.5

    assert axes[3] == {"name": "y", "type": "space", "unit": "micrometer"}
    assert scales[3] == 0.325

    assert axes[4] == {"name": "x", "type": "space", "unit": "micrometer"}
    assert scales[4] == 0.325


def test_ome_axes_scales_minimal() -> None:
    """Test _ome_axes_scales with minimal dimensions."""
    dims = [
        Dimension(label="y", size=512),
        Dimension(label="x", size=512),
    ]

    axes, scales = _ome_axes_scales(dims)

    assert len(axes) == 2
    assert len(scales) == 2

    assert axes[0] == {"name": "y", "type": "space", "unit": "unknown"}
    assert axes[1] == {"name": "x", "type": "space", "unit": "unknown"}
    assert scales == [1.0, 1.0]


def test_ome_meta_v5_single_array() -> None:
    """Test ome_meta_v5 with a single array."""
    dims = [
        Dimension(label="t", size=5, unit=(1.0, "s")),
        Dimension(label="c", size=3),
        Dimension(label="z", size=10, unit=(0.5, "um")),
        Dimension(label="y", size=512, unit=(0.325, "um")),
        Dimension(label="x", size=512, unit=(0.325, "um")),
    ]

    array_dims = {"0": dims}

    attrs = ome_meta_v5(array_dims)

    assert "ome" in attrs
    assert attrs["ome"]["version"] == "0.5"
    assert "multiscales" in attrs["ome"]
    assert len(attrs["ome"]["multiscales"]) == 1

    multiscale = attrs["ome"]["multiscales"][0]
    assert "axes" in multiscale
    assert "datasets" in multiscale
    assert len(multiscale["datasets"]) == 1

    dataset = multiscale["datasets"][0]
    assert dataset["path"] == "0"
    assert "coordinateTransformations" in dataset


def test_ome_meta_v5_multiple_arrays_same_axes() -> None:
    """Test ome_meta_v5 with multiple arrays sharing the same axes."""
    dims = [
        Dimension(label="t", size=5, unit=(1.0, "s")),
        Dimension(label="c", size=3),
        Dimension(label="z", size=10, unit=(0.5, "um")),
        Dimension(label="y", size=512, unit=(0.325, "um")),
        Dimension(label="x", size=512, unit=(0.325, "um")),
    ]

    # Multiple arrays with the same dimensions (e.g., different resolution levels)
    array_dims = {"0": dims, "1": dims, "2": dims}

    attrs = ome_meta_v5(array_dims)

    # Should create a single multiscale with multiple datasets
    assert len(attrs["ome"]["multiscales"]) == 1
    multiscale = attrs["ome"]["multiscales"][0]
    assert len(multiscale["datasets"]) == 3


def test_ome_meta_v5_multiple_arrays_different_axes() -> None:
    """Test ome_meta_v5 with multiple arrays with different axes."""
    dims1 = [
        Dimension(label="t", size=5, unit=(1.0, "s")),
        Dimension(label="c", size=3),
        Dimension(label="y", size=512, unit=(0.325, "um")),
        Dimension(label="x", size=512, unit=(0.325, "um")),
    ]

    dims2 = [
        Dimension(label="z", size=10, unit=(0.5, "um")),
        Dimension(label="y", size=512, unit=(0.325, "um")),
        Dimension(label="x", size=512, unit=(0.325, "um")),
    ]

    array_dims = {"0": dims1, "1": dims2}

    attrs = ome_meta_v5(array_dims)

    # Should create multiple multiscales for different axes configurations
    assert len(attrs["ome"]["multiscales"]) == 2


def test_dim_to_yaozarrs_axis_v05_time() -> None:
    """Test dim_to_yaozarrs_axis_v05 for time axis."""
    yaozarrs = pytest.importorskip("yaozarrs")

    dim = Dimension(label="t", size=10, unit=(1.0, "s"))
    axis = dim_to_yaozarrs_axis_v05(dim)

    assert isinstance(axis, yaozarrs.v05.TimeAxis)
    assert axis.name == "t"
    assert axis.unit == "second"


def test_dim_to_yaozarrs_axis_v05_channel() -> None:
    """Test dim_to_yaozarrs_axis_v05 for channel axis."""
    yaozarrs = pytest.importorskip("yaozarrs")

    dim = Dimension(label="c", size=3)
    axis = dim_to_yaozarrs_axis_v05(dim)

    assert isinstance(axis, yaozarrs.v05.ChannelAxis)
    assert axis.name == "c"


def test_dim_to_yaozarrs_axis_v05_space() -> None:
    """Test dim_to_yaozarrs_axis_v05 for space axes."""
    yaozarrs = pytest.importorskip("yaozarrs")

    for label in ["x", "y", "z"]:
        dim = Dimension(label=label, size=512, unit=(0.325, "um"))
        axis = dim_to_yaozarrs_axis_v05(dim)

        assert isinstance(axis, yaozarrs.v05.SpaceAxis)
        assert axis.name == label
        assert axis.unit == "micrometer"


def test_dim_to_yaozarrs_axis_v05_custom() -> None:
    """Test dim_to_yaozarrs_axis_v05 for custom axis."""
    yaozarrs = pytest.importorskip("yaozarrs")

    dim = Dimension(label="other", size=5)
    axis = dim_to_yaozarrs_axis_v05(dim)

    assert isinstance(axis, yaozarrs.v05.CustomAxis)
    assert axis.name == "other"


def test_dim_to_yaozarrs_axis_v05_unknown_unit() -> None:
    """Test dim_to_yaozarrs_axis_v05 with unknown unit."""
    yaozarrs = pytest.importorskip("yaozarrs")

    # Dimension without unit
    dim = Dimension(label="x", size=512)
    axis = dim_to_yaozarrs_axis_v05(dim)

    assert isinstance(axis, yaozarrs.v05.SpaceAxis)
    assert axis.name == "x"
    assert axis.unit is None
