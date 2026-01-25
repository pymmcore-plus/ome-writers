import pytest

from ome_writers._util import dims_from_useq

try:
    import useq
except ImportError:
    pytest.skip("useq not installed", allow_module_level=True)


# Test cases for dims_from_useq with different position configurations
SEQ = [
    useq.MDASequence(
        axis_order="ptcz",
        stage_positions=[(0.0, 0.0), (10.0, 10.0)],
        time_plan={"interval": 0.1, "loops": 3},
        channels=["DAPI", "Cy5"],
        z_plan={"range": 2, "step": 1.0},
    ),
    useq.MDASequence(
        axis_order="ptcz",
        stage_positions=[(0.0, 0.0), (10.0, 10.0)],
        time_plan={"interval": 0.1, "loops": 3},
        channels=["DAPI", "Cy5"],
        z_plan={"range": 2, "step": 1.0},
        grid_plan=useq.GridRowsColumns(rows=1, columns=2),
    ),
    useq.MDASequence(
        axis_order="ptcz",
        stage_positions=[
            useq.Position(x=0.0, y=0.0, name="single_pos"),
            useq.Position(
                x=10.0,
                y=10.0,
                name="grid",
                sequence=useq.MDASequence(
                    grid_plan=useq.GridRowsColumns(rows=1, columns=2)
                ),
            ),
        ],
        time_plan={"interval": 0.1, "loops": 3},
        channels=["DAPI", "Cy5"],
        z_plan={"range": 2, "step": 1.0},
    ),
]

# Expected results for each test case
EXPECTED_RESULTS = [
    (["p", "t", "c", "z", "y", "x"], ["0", "1"]),  # simple positions
    (
        ["p", "t", "c", "z", "y", "x"],
        ["p000g000", "p000g001", "p001g000", "p001g001"],
    ),  # grid+positions
    (
        ["p", "t", "c", "z", "y", "x"],
        ["single_pos", "grid_g000", "grid_g001"],
    ),  # position subsequences
]


def _seq_id(seq: "useq.MDASequence") -> str:
    """Generate test ID from sequence configuration."""
    if any(pos.sequence for pos in seq.stage_positions):
        return "position_subsequences"
    elif seq.grid_plan is not None:
        return "grid_and_positions"
    else:
        return "simple_positions"


@pytest.mark.parametrize("seq", SEQ, ids=_seq_id)
def test_useq_to_dims(seq: "useq.MDASequence") -> None:
    """Test dims_from_useq with different position configurations."""
    from ome_writers._schema import PositionDimension

    # Get expected results for this sequence
    seq_idx = SEQ.index(seq)
    expected_names, expected_pos_names = EXPECTED_RESULTS[seq_idx]

    # Test with units and pixel_size_um
    dims = dims_from_useq(
        seq,
        image_width=64,
        image_height=64,
        units={"t": (0.1, "second"), "z": (3.0, "micrometer")},
        pixel_size_um=0.103,
    )

    # Check dimension names
    assert [dim.name for dim in dims] == expected_names

    # Check position names
    pos_dim = dims[0]
    assert isinstance(pos_dim, PositionDimension)
    assert pos_dim.names == expected_pos_names

    # Check units and scales for non-position dimensions
    for dim in dims[1:]:
        if dim.name == "t":
            assert dim.unit == "second"
            assert dim.scale == 0.1
        elif dim.name == "z":
            assert dim.unit == "micrometer"
            assert dim.scale == 3.0
        elif dim.name in ("y", "x"):
            assert dim.unit == "micrometer"
            assert dim.scale == 0.103
        elif dim.name == "c":
            assert dim.unit is None
            assert dim.scale == 1.0


def test_useq_grid_and_positions_explicit_g() -> None:
    """Test grid + positions with explicit 'g' in axis_order."""
    from ome_writers._schema import PositionDimension

    seq = useq.MDASequence(
        axis_order="pgtcz",
        stage_positions=[(0.0, 0.0), (10.0, 10.0)],
        time_plan={"interval": 0.1, "loops": 3},
        channels=["DAPI", "Cy5"],
        z_plan={"range": 2, "step": 1.0},
        grid_plan=useq.GridRowsColumns(rows=1, columns=2),
    )

    dims = dims_from_useq(seq, image_width=32, image_height=32)
    assert [dim.name for dim in dims] == ["p", "t", "c", "z", "y", "x"]

    pos_dim = dims[0]
    assert isinstance(pos_dim, PositionDimension)
    assert pos_dim.names == ["p000g000", "p000g001", "p001g000", "p001g001"]


def test_useq_units_without_pixel_size() -> None:
    """Test that units can be provided for non-spatial dimensions without pixel size."""
    seq = useq.MDASequence(
        axis_order="ptcz",
        stage_positions=[(0.0, 0.0)],
        time_plan={"interval": 0.1, "loops": 3},
        channels=["DAPI", "Cy5"],
        z_plan={"range": 2, "step": 1.0},
    )

    # Provide units for z dimension but no pixel_size_um
    dims = dims_from_useq(
        seq,
        image_width=64,
        image_height=64,
        units={"z": (3.0, "micrometer")},
    )

    # Check that z has the custom scale and unit
    z_dim = next(d for d in dims if d.name == "z")
    assert z_dim.unit == "micrometer"
    assert z_dim.scale == 3.0

    # Check that x and y have None scale (no pixel_size_um provided)
    x_dim = next(d for d in dims if d.name == "x")
    y_dim = next(d for d in dims if d.name == "y")
    assert x_dim.scale is None
    assert y_dim.scale is None
    # But they should still have micrometer unit from StandardAxis defaults
    assert x_dim.unit == "micrometer"
    assert y_dim.unit == "micrometer"
