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
        axis_order="ptgcz",
        stage_positions=[(0.0, 0.0), (10.0, 10.0)],
        time_plan={"interval": 0.1, "loops": 3},
        channels=["DAPI", "Cy5"],
        z_plan={"range": 2, "step": 1.0},
        grid_plan=useq.GridRowsColumns(rows=1, columns=2),
    ),
    useq.MDASequence(
        axis_order="ptgzc",
        stage_positions=[(0.0, 0.0), (10.0, 10.0)],
        time_plan={"interval": 0.1, "loops": 3},
        channels=["DAPI", "Cy5"],
        z_plan={"range": 2, "step": 1.0},
        grid_plan=useq.GridRowsColumns(rows=1, columns=2),
    ),
    useq.MDASequence(
        axis_order="pgtcz",
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
# Format: (dim_names, pos_names, grid_info)
# grid_info is a list of (grid_row, grid_column) tuples or None for non-grid positions
EXPECTED_RESULTS = [
    # simple positions
    (["p", "t", "c", "z", "y", "x"], ["0", "1"], None),
    (
        ["p", "t", "c", "z", "y", "x"],
        ["0000", "0000", "0001", "0001"],  # Same name, different grid coords
        [(0, 0), (0, 1), (0, 0), (0, 1)],
    ),  # grid+positions (ptgcz)
    (
        ["p", "t", "z", "c", "y", "x"],
        ["0000", "0000", "0001", "0001"],  # Same name, different grid coords
        [(0, 0), (0, 1), (0, 0), (0, 1)],
    ),  # grid+positions (ptgzc)
    (
        ["p", "t", "c", "z", "y", "x"],
        ["single_pos", "grid", "grid"],  # Same name, different grid coords
        [None, (0, 0), (0, 1)],
    ),  # position subsequences
]


def _seq_id(seq: "useq.MDASequence") -> str:
    """Generate test ID from sequence configuration."""
    if any(pos.sequence for pos in seq.stage_positions):
        return "position_subsequences"
    elif seq.grid_plan is not None:
        return f"grid_and_positions_{seq.axis_order}"
    else:
        return "simple_positions"


@pytest.mark.parametrize("seq", SEQ, ids=_seq_id)
def test_useq_to_dims(seq: "useq.MDASequence") -> None:
    """Test dims_from_useq with different position configurations."""
    from ome_writers._schema import PositionDimension

    # Get expected results for this sequence
    seq_idx = SEQ.index(seq)
    expected_names, expected_pos_names, expected_grid_info = EXPECTED_RESULTS[seq_idx]

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

    # Check grid_row and grid_column for positions with grid info
    if expected_grid_info is not None:
        for pos, grid_info in zip(pos_dim.positions, expected_grid_info, strict=True):
            if grid_info is None:
                assert pos.grid_row is None
                assert pos.grid_column is None
            else:
                assert pos.grid_row == grid_info[0]
                assert pos.grid_column == grid_info[1]

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


# WellPlatePlan test cases: (sequence, expected_positions)
# expected_positions format: [(name, plate_row, plate_col, grid_row, grid_col), ...]
WELL_PLATE_CASES = [
    pytest.param(
        useq.MDASequence(
            axis_order="ptcz",
            stage_positions=useq.WellPlatePlan(
                plate=useq.WellPlate.from_str("96-well"),
                a1_center_xy=(0.0, 0.0),
                selected_wells=((0, 1), (0, 1)),  # Wells A1 and B2
                well_points_plan=useq.GridRowsColumns(rows=1, columns=2),
            ),
            time_plan={"interval": 0.1, "loops": 2},
            channels=["DAPI"],
        ),
        [
            ("A1_0000", "A", "1", 0, 0),
            ("A1_0001", "A", "1", 0, 1),
            ("B2_0000", "B", "2", 0, 0),
            ("B2_0001", "B", "2", 0, 1),
        ],
        id="well_points_plan_only",
    ),
    pytest.param(
        useq.MDASequence(
            axis_order="ptcz",
            stage_positions=useq.WellPlatePlan(
                plate=useq.WellPlate.from_str("96-well"),
                a1_center_xy=(0.0, 0.0),
                selected_wells=((0, 1), (0, 1)),  # Wells A1 and B2
            ),
            time_plan={"interval": 0.1, "loops": 2},
            channels=["DAPI"],
            grid_plan=useq.GridRowsColumns(rows=1, columns=2),
        ),
        [
            ("A1", "A", "1", 0, 0),
            ("A1", "A", "1", 0, 1),
            ("B2", "B", "2", 0, 0),
            ("B2", "B", "2", 0, 1),
        ],
        id="sequence_grid_only",
    ),
    pytest.param(
        useq.MDASequence(
            axis_order="ptcz",
            stage_positions=useq.WellPlatePlan(
                plate=useq.WellPlate.from_str("96-well"),
                a1_center_xy=(0.0, 0.0),
                selected_wells=((0,), (0,)),  # Just well A1
                well_points_plan=useq.GridRowsColumns(rows=1, columns=2),
            ),
            time_plan={"interval": 0.1, "loops": 2},
            channels=["DAPI"],
            grid_plan=useq.GridRowsColumns(rows=2, columns=1),
        ),
        [
            ("A1_0000", "A", "1", 0, 0),
            ("A1_0000", "A", "1", 1, 0),
            ("A1_0001", "A", "1", 0, 0),
            ("A1_0001", "A", "1", 1, 0),
        ],
        id="both_grids",
    ),
]


@pytest.mark.parametrize(("seq", "expected"), WELL_PLATE_CASES)
def test_well_plate_plan(seq: "useq.MDASequence", expected: list) -> None:
    """Test dims_from_useq with WellPlatePlan extracts plate and grid coordinates."""
    from ome_writers._schema import PositionDimension

    dims = dims_from_useq(seq, image_width=64, image_height=64)

    pos_dim = dims[0]
    assert isinstance(pos_dim, PositionDimension)
    assert len(pos_dim.positions) == len(expected)

    for pos, (name, plate_row, plate_col, grid_row, grid_col) in zip(
        pos_dim.positions, expected, strict=True
    ):
        assert pos.name == name
        assert pos.plate_row == plate_row
        assert pos.plate_column == plate_col
        assert pos.grid_row == grid_row
        assert pos.grid_column == grid_col
