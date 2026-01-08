"""Tests to improve code coverage."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest

from ome_writers._auto import init_stream
from ome_writers._dimensions import Dimension
from ome_writers._util import (
    DimensionIndexIterator,
    dims_from_useq,
    fake_data_for_sizes,
)
from ome_writers.backends._acquire_zarr import AcquireZarrStream
from ome_writers.backends._tifffile import TifffileStream
from ome_writers.backends._yaozarrs import TensorStoreZarrStream, ZarrPythonStream

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


def test_dims_from_useq_with_grid() -> None:
    """Test dims_from_useq properly combines grid and position dimensions."""
    useq = pytest.importorskip("useq")

    # Grid only
    seq1 = useq.MDASequence(
        grid_plan=useq.GridRowsColumns(rows=2, columns=3),
    )
    dims1 = dims_from_useq(seq1, image_width=32, image_height=32)
    # Should have p (6 positions from grid), y, x
    assert len(dims1) == 3
    assert dims1[0].label == "p"
    assert dims1[0].size == 6  # 2*3
    assert dims1[1].label == "y"
    assert dims1[2].label == "x"

    # Stage positions with grid
    seq2 = useq.MDASequence(
        stage_positions=[(0.0, 0.0), (10.0, 10.0)],
        grid_plan=useq.GridRowsColumns(rows=2, columns=2),
    )
    dims2 = dims_from_useq(seq2, image_width=64, image_height=64)
    # Should have p (2*4=8 total positions), y, x
    assert len(dims2) == 3
    assert dims2[0].label == "p"
    assert dims2[0].size == 8  # 2 positions * 4 grid points
    assert dims2[1].label == "y"
    assert dims2[2].label == "x"

    # Full sequence with time, channels, z, stage positions, and grid
    seq3 = useq.MDASequence(
        axis_order="ptgcz",
        stage_positions=[(0.0, 0.0), (10.0, 10.0)],
        time_plan={"interval": 0.1, "loops": 3},
        channels=["DAPI", "FITC"],
        z_plan={"range": 2, "step": 1.0},
        grid_plan=useq.GridRowsColumns(rows=2, columns=2),
    )
    dims3 = dims_from_useq(seq3, image_width=128, image_height=128)
    # Should have p, t, c, z, y, x (g combined into p)
    dim_labels = [d.label for d in dims3]
    assert "g" not in dim_labels, "Grid 'g' should be combined into 'p'"
    assert "p" in dim_labels
    p_dim = next(d for d in dims3 if d.label == "p")
    assert p_dim.size == 8  # 2 stage positions * 4 grid points
    t_dim = next(d for d in dims3 if d.label == "t")
    assert t_dim.size == 3
    c_dim = next(d for d in dims3 if d.label == "c")
    assert c_dim.size == 2
    z_dim = next(d for d in dims3 if d.label == "z")
    assert z_dim.size == 3


def test_dimension_index_iterator_empty() -> None:
    """Test DimensionIndexIterator with empty dimensions."""
    dims: list[Dimension] = []
    it = DimensionIndexIterator(dims, storage_order_dimensions=[])

    assert len(it) == 0
    assert list(it) == []


def test_dimension_index_iterator_2d_only() -> None:
    """Test DimensionIndexIterator with only spatial dimensions (2D)."""
    # Only spatial dimensions, no time/channel/etc
    dims = [
        Dimension(label="y", size=32, unit=None, chunk_size=1),
        Dimension(label="x", size=32, unit=None, chunk_size=1),
    ]
    it = DimensionIndexIterator(dims, storage_order_dimensions=[])

    # Should yield one frame with empty index tuple
    assert len(it) == 0  # No non-spatial dimensions
    result = list(it)
    assert len(result) == 1
    assert result[0] == (0, ())


def test_dimension_index_iterator_validation() -> None:
    """Test DimensionIndexIterator parameter validation."""
    dims = [
        Dimension(label="t", size=2, unit=(1.0, "s"), chunk_size=1),
        Dimension(label="y", size=32, unit=None, chunk_size=1),
        Dimension(label="x", size=32, unit=None, chunk_size=1),
    ]

    # Should raise error if storage_order_dimensions includes 'y'
    with pytest.raises(ValueError, match="should not include 'y', 'x'"):
        DimensionIndexIterator(dims, storage_order_dimensions=["t", "y"])

    # Should raise error if storage_order_dimensions includes 'x'
    with pytest.raises(ValueError, match="should not include 'y', 'x'"):
        DimensionIndexIterator(dims, storage_order_dimensions=["x"])

    # Should raise error if storage_order_dimensions includes position key
    with pytest.raises(ValueError, match="should not include 'y', 'x'"):
        DimensionIndexIterator(dims, storage_order_dimensions=["p"])


def test_init_stream_backends() -> None:
    """Test init_stream with different backend names."""
    if AcquireZarrStream.is_available():
        stream = init_stream("test.zarr", backend="acquire-zarr")
        assert isinstance(stream, AcquireZarrStream)

    if TifffileStream.is_available():
        stream = init_stream("test.tiff", backend="tiff")
        assert isinstance(stream, TifffileStream)


@pytest.mark.skipif(
    not (
        AcquireZarrStream.is_available()
        or TensorStoreZarrStream.is_available()
        or ZarrPythonStream.is_available()
    ),
    reason="no zarr backend available",
)
def test_autobackend_zarr(tmp_path: Path) -> None:
    """Test automatic backend selection for .zarr files."""
    zarr_path = tmp_path / "test.zarr"
    stream = init_stream(str(zarr_path), backend="auto")
    # Should select acquire-zarr if available, otherwise tensorstore or zarr-python
    assert isinstance(
        stream, (AcquireZarrStream, TensorStoreZarrStream, ZarrPythonStream)
    )


@pytest.mark.skipif(not TifffileStream.is_available(), reason="tifffile not available")
def test_autobackend_tiff(tmp_path: Path) -> None:
    """Test automatic backend selection for .tiff files."""
    tiff_path = tmp_path / "test.tiff"
    stream = init_stream(str(tiff_path), backend="auto")
    assert isinstance(stream, TifffileStream)

    # Also test .ome.tiff extension
    ome_tiff_path = tmp_path / "test.ome.tiff"
    stream = init_stream(str(ome_tiff_path), backend="auto")
    assert isinstance(stream, TifffileStream)
