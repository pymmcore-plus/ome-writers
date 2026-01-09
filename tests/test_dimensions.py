"""Tests for the _dimensions module."""

from __future__ import annotations

import pytest

from ome_writers._dimensions import Dimension
from ome_writers._ngff_metadata import build_yaozarrs_image_metadata_v05


def test_dimension_properties() -> None:
    """Test Dimension NamedTuple properties."""
    dim = Dimension(label="t", size=10, unit=(1.0, "s"), chunk_size=5)

    assert dim.label == "t"
    assert dim.size == 10
    assert dim.unit == (1.0, "s")
    assert dim.chunk_size == 5
    assert dim.ome_dim_type == "time"
    assert dim.ome_unit == "second"
    assert dim.ome_scale == 1.0


def test_dimension_ome_dim_type() -> None:
    """Test ome_dim_type property for different labels."""
    assert Dimension(label="x", size=512).ome_dim_type == "space"
    assert Dimension(label="y", size=512).ome_dim_type == "space"
    assert Dimension(label="z", size=10).ome_dim_type == "space"
    assert Dimension(label="t", size=10).ome_dim_type == "time"
    assert Dimension(label="c", size=3).ome_dim_type == "channel"
    assert Dimension(label="p", size=2).ome_dim_type == "other"
    assert Dimension(label="other", size=5).ome_dim_type == "other"


def test_dimension_ome_unit() -> None:
    """Test ome_unit property for different unit types."""
    dim_um = Dimension(label="x", size=512, unit=(0.5, "um"))
    assert dim_um.ome_unit == "micrometer"

    dim_s = Dimension(label="t", size=10, unit=(1.0, "s"))
    assert dim_s.ome_unit == "second"

    dim_ml = Dimension(label="c", size=3, unit=(1.0, "ml"))
    assert dim_ml.ome_unit == "milliliter"

    dim_unknown = Dimension(label="x", size=512, unit=(1.0, "pixels"))
    assert dim_unknown.ome_unit == "unknown"

    dim_no_unit = Dimension(label="x", size=512)
    assert dim_no_unit.ome_unit == "unknown"


def test_dimension_ome_scale() -> None:
    """Test ome_scale property."""
    dim_with_scale = Dimension(label="x", size=512, unit=(0.325, "um"))
    assert dim_with_scale.ome_scale == 0.325

    dim_without_unit = Dimension(label="x", size=512)
    assert dim_without_unit.ome_scale == 1.0


def test_build_yaozarrs_image_metadata_v05() -> None:
    """Test building yaozarrs v05 Image metadata."""
    yaozarrs = pytest.importorskip("yaozarrs")

    dims = [
        Dimension(label="t", size=5, unit=(1.0, "s")),
        Dimension(label="c", size=3, unit=(1.0, "ml")),
        Dimension(label="z", size=10, unit=(0.5, "um")),
        Dimension(label="y", size=512, unit=(0.325, "um")),
        Dimension(label="x", size=512, unit=(0.325, "um")),
    ]

    image = build_yaozarrs_image_metadata_v05(dims)

    # Check that we got an Image object
    assert isinstance(image, yaozarrs.v05.Image)

    # Check that we have one multiscale
    assert len(image.multiscales) == 1
    multiscale = image.multiscales[0]

    # Check axes
    assert len(multiscale.axes) == 5
    assert multiscale.axes[0].name == "t"
    assert multiscale.axes[1].name == "c"
    assert multiscale.axes[2].name == "z"
    assert multiscale.axes[3].name == "y"
    assert multiscale.axes[4].name == "x"

    # Check datasets
    assert len(multiscale.datasets) == 1
    dataset = multiscale.datasets[0]
    assert dataset.path == "0"

    # Check scales in coordinate transformations
    assert len(dataset.coordinateTransformations) == 1
    scale_transform = dataset.coordinateTransformations[0]
    assert scale_transform.scale == [1.0, 1.0, 0.5, 0.325, 0.325]


def test_dimension_with_chunk_size() -> None:
    """Test Dimension with chunk_size specified."""
    dim = Dimension(label="z", size=100, chunk_size=10)
    assert dim.chunk_size == 10

    dim_none = Dimension(label="z", size=100, chunk_size=None)
    assert dim_none.chunk_size is None

    dim_default = Dimension(label="z", size=100)
    assert dim_default.chunk_size is None


def test_dimension_minimal() -> None:
    """Test Dimension with minimal arguments."""
    dim = Dimension(label="x", size=512)
    assert dim.label == "x"
    assert dim.size == 512
    assert dim.unit is None
    assert dim.chunk_size is None
