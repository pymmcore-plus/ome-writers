"""Tests for Dimension conversion methods."""

from __future__ import annotations

import json

import numpy as np
import pytest

import ome_writers as omew


def test_dimension_to_ome_types() -> None:
    """Test dims_to_ome() conversion."""
    pytest.importorskip("ome_types")
    from ome_types import model as ome

    # Create test dimensions
    dims = [
        omew.Dimension(label="t", size=3, unit=(1.0, "s")),
        omew.Dimension(label="c", size=2),
        omew.Dimension(label="z", size=5, unit=(0.5, "um")),
        omew.Dimension(label="y", size=64, unit=(0.1, "um")),
        omew.Dimension(label="x", size=64, unit=(0.1, "um")),
    ]

    # Convert to OME types
    ome_obj = omew.dims_to_ome(dims, dtype=np.uint16)

    # Validate the result
    assert isinstance(ome_obj, ome.OME)
    assert len(ome_obj.images) == 1
    assert ome_obj.images[0].pixels is not None

    pixels = ome_obj.images[0].pixels
    assert pixels.size_t == 3
    assert pixels.size_c == 2
    assert pixels.size_z == 5
    assert pixels.size_y == 64
    assert pixels.size_x == 64
    assert pixels.type.value == "uint16"

    # The OME object structure is valid if we can access all fields
    assert ome_obj.creator is not None


def test_dimension_to_ome_types_with_positions() -> None:
    """Test dims_to_ome() with position dimension."""
    pytest.importorskip("ome_types")
    from ome_types import model as ome

    dims = [
        omew.Dimension(label="t", size=2, unit=(1.0, "s")),
        omew.Dimension(label="c", size=1),
        omew.Dimension(label="y", size=32, unit=(0.2, "um")),
        omew.Dimension(label="x", size=32, unit=(0.2, "um")),
        omew.Dimension(label="p", size=3),
    ]

    ome_obj = omew.dims_to_ome(dims, dtype=np.uint8)

    assert isinstance(ome_obj, ome.OME)
    # Should create 3 images for 3 positions
    assert len(ome_obj.images) == 3

    for i, img in enumerate(ome_obj.images):
        assert img.id == f"Image:{i}"
        assert img.pixels is not None
        assert img.pixels.size_t == 2
        assert img.pixels.size_c == 1
        assert img.pixels.size_y == 32
        assert img.pixels.size_x == 32

    # The OME object structure is valid if we can access all fields
    assert ome_obj.creator is not None


def test_dimension_to_yaozarrs_v5() -> None:
    """Test dims_to_yaozarrs_v5() conversion."""
    pytest.importorskip("yaozarrs")
    import json

    from yaozarrs import validate_ome_json

    dims = [
        omew.Dimension(label="t", size=3, unit=(1.0, "s")),
        omew.Dimension(label="c", size=2),
        omew.Dimension(label="y", size=64, unit=(0.1, "um")),
        omew.Dimension(label="x", size=64, unit=(0.1, "um")),
    ]

    # Convert to yaozarrs v0.5 format
    array_dims = {"0": dims}
    image = omew.dims_to_yaozarrs_v5(array_dims)

    # Validate structure - image is a yaozarrs.v05.Image object
    assert image.version == "0.5"
    assert len(image.multiscales) == 1

    multiscale = image.multiscales[0]
    assert len(multiscale.axes) == 4
    assert len(multiscale.datasets) == 1
    assert multiscale.datasets[0].path == "0"

    # Check axes
    axes = multiscale.axes
    assert len(axes) == 4
    assert axes[0].name == "t"
    assert axes[0].type == "time"
    assert axes[1].name == "c"
    assert axes[1].type == "channel"
    assert axes[2].name == "y"
    assert axes[2].type == "space"
    assert axes[3].name == "x"
    assert axes[3].type == "space"

    # Validate using yaozarrs (convert Image object to dict first)
    zarr_meta = {"ome": image.model_dump(exclude_unset=True, by_alias=True)}
    validate_ome_json(json.dumps(zarr_meta))


def test_dimension_to_yaozarrs_v5_multiple_arrays() -> None:
    """Test dims_to_yaozarrs_v5() with multiple arrays."""
    pytest.importorskip("yaozarrs")
    import json

    from yaozarrs import validate_ome_json

    dims = [
        omew.Dimension(label="t", size=2, unit=(1.0, "s")),
        omew.Dimension(label="c", size=1),
        omew.Dimension(label="y", size=32, unit=(0.2, "um")),
        omew.Dimension(label="x", size=32, unit=(0.2, "um")),
    ]

    # Create multiple arrays with same dimensions (e.g., multi-position)
    array_dims = {"0": dims, "1": dims, "2": dims}
    image = omew.dims_to_yaozarrs_v5(array_dims)

    # Validate structure - image is a yaozarrs.v05.Image object
    assert image.version == "0.5"

    multiscale = image.multiscales[0]
    assert len(multiscale.datasets) == 3
    assert multiscale.datasets[0].path == "0"
    assert multiscale.datasets[1].path == "1"
    assert multiscale.datasets[2].path == "2"

    # Validate using yaozarrs (convert Image object to dict first)
    zarr_meta = {"ome": image.model_dump(exclude_unset=True, by_alias=True)}
    validate_ome_json(json.dumps(zarr_meta))


def test_backward_compatibility_dims_to_ome() -> None:
    """Test that the old dims_to_ome function still works."""
    pytest.importorskip("ome_types")

    dims = [
        omew.Dimension(label="y", size=32, unit=(0.2, "um")),
        omew.Dimension(label="x", size=32, unit=(0.2, "um")),
    ]

    # Old function should still work
    ome_obj = omew.dims_to_ome(dims, dtype=np.uint16)
    assert ome_obj is not None
    assert len(ome_obj.images) == 1


def test_backward_compatibility_dims_to_yaozarrs_v5() -> None:
    """Test that the old ome_meta_v5 function still works."""
    pytest.importorskip("yaozarrs")

    dims = [
        omew.Dimension(label="y", size=32, unit=(0.2, "um")),
        omew.Dimension(label="x", size=32, unit=(0.2, "um")),
    ]

    # Old function should still work and return a dict
    zarr_meta = omew.ome_meta_v5({"0": dims})

    assert zarr_meta is not None

    # Validate structure - image is a yaozarrs.v05.Image object
    assert zarr_meta.version == "0.5"

    multiscale = zarr_meta.multiscales[0]
    assert len(multiscale.datasets) == 1
    assert multiscale.datasets[0].path == "0"

    from yaozarrs import validate_ome_json

    # Validate using yaozarrs (convert Image object to dict first)
    zarr_meta = {"ome": zarr_meta.model_dump(exclude_unset=True, by_alias=True)}
    validate_ome_json(json.dumps(zarr_meta))
