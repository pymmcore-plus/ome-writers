from __future__ import annotations

from dataclasses import dataclass

import pytest

from ome_writers._schema import AcquisitionSettings
from ome_writers._util import dims_from_useq

try:
    import useq
except ImportError:
    pytest.skip("useq not installed", allow_module_level=True)


@dataclass
class ExpectedPosition:
    """Expected position metadata."""

    name: str
    plate_row: str | None = None
    plate_col: str | None = None
    grid_row: int | None = None
    grid_col: int | None = None


@dataclass
class Case:
    """Test case for dims_from_useq."""

    seq: useq.MDASequence
    expected_dim_names: list[str]
    expected_positions: list[ExpectedPosition] | None = None
    id: str = ""


# Test cases for dims_from_useq with different configurations
SEQ_CASES = [
    Case(
        seq=useq.MDASequence(
            axis_order="ptcz",
            stage_positions=[(0.0, 0.0), (10.0, 10.0)],
            time_plan={"interval": 0.1, "loops": 3},
            channels=["DAPI", "Cy5"],
            z_plan={"range": 2, "step": 1.0},
        ),
        expected_dim_names=["p", "t", "c", "z", "y", "x"],
        expected_positions=[ExpectedPosition(name="0"), ExpectedPosition(name="1")],
        id="simple_positions",
    ),
    Case(
        seq=useq.MDASequence(
            axis_order="ptgcz",
            stage_positions=[(0.0, 0.0), (10.0, 10.0)],
            time_plan={"interval": 0.1, "loops": 3},
            channels=["DAPI", "Cy5"],
            z_plan={"range": 2, "step": 1.0},
            grid_plan=useq.GridRowsColumns(rows=1, columns=2),
        ),
        expected_dim_names=["p", "t", "c", "z", "y", "x"],
        expected_positions=[
            ExpectedPosition(name="0000", grid_row=0, grid_col=0),
            ExpectedPosition(name="0000", grid_row=0, grid_col=1),
            ExpectedPosition(name="0001", grid_row=0, grid_col=0),
            ExpectedPosition(name="0001", grid_row=0, grid_col=1),
        ],
        id="grid_and_positions_ptgcz",
    ),
    Case(
        seq=useq.MDASequence(
            axis_order="ptgzc",
            stage_positions=[(0.0, 0.0), (10.0, 10.0)],
            time_plan={"interval": 0.1, "loops": 3},
            channels=["DAPI", "Cy5"],
            z_plan={"range": 2, "step": 1.0},
            grid_plan=useq.GridRowsColumns(rows=1, columns=2),
        ),
        expected_dim_names=["p", "t", "z", "c", "y", "x"],
        expected_positions=[
            ExpectedPosition(name="0000", grid_row=0, grid_col=0),
            ExpectedPosition(name="0000", grid_row=0, grid_col=1),
            ExpectedPosition(name="0001", grid_row=0, grid_col=0),
            ExpectedPosition(name="0001", grid_row=0, grid_col=1),
        ],
        id="grid_and_positions_ptgzc",
    ),
    Case(
        seq=useq.MDASequence(
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
        expected_dim_names=["p", "t", "c", "z", "y", "x"],
        expected_positions=[
            ExpectedPosition(name="single_pos"),
            ExpectedPosition(name="grid", grid_row=0, grid_col=0),
            ExpectedPosition(name="grid", grid_row=0, grid_col=1),
        ],
        id="position_subsequences",
    ),
    Case(
        seq=useq.MDASequence(
            axis_order="tcpz",
            stage_positions=[(0.0, 0.0), (10.0, 10.0)],
            time_plan={"interval": 0.1, "loops": 2},
            channels=["DAPI", "Cy5"],
            z_plan={"range": 2, "step": 1.0},
        ),
        expected_dim_names=["t", "c", "p", "z", "y", "x"],
        expected_positions=[ExpectedPosition(name="0"), ExpectedPosition(name="1")],
        id="position_not_first",
    ),
    Case(
        seq=useq.MDASequence(
            axis_order="tcz",
            time_plan={"interval": 0.1, "loops": 2},
            channels=["DAPI", "Cy5"],
            z_plan={"range": 2, "step": 1.0},
        ),
        expected_dim_names=["t", "c", "z", "y", "x"],
        expected_positions=None,
        id="no_position_dimension",
    ),
    Case(
        seq=useq.MDASequence(
            axis_order="ptcz",
            stage_positions=useq.WellPlatePlan(
                plate=useq.WellPlate.from_str("96-well"),
                a1_center_xy=(0.0, 0.0),
                selected_wells=((0, 1), (0, 1)),
                well_points_plan=useq.GridRowsColumns(rows=1, columns=2),
            ),
            time_plan={"interval": 0.1, "loops": 2},
            channels=["DAPI"],
        ),
        expected_dim_names=["p", "t", "c", "y", "x"],
        expected_positions=[
            ExpectedPosition("A1_0000", "A", "1", grid_row=0, grid_col=0),
            ExpectedPosition("A1_0001", "A", "1", grid_row=0, grid_col=1),
            ExpectedPosition("B2_0000", "B", "2", grid_row=0, grid_col=0),
            ExpectedPosition("B2_0001", "B", "2", grid_row=0, grid_col=1),
        ],
        id="well_plate_with_points",
    ),
    Case(
        seq=useq.MDASequence(
            axis_order="ptcz",
            stage_positions=useq.WellPlatePlan(
                plate=useq.WellPlate.from_str("96-well"),
                a1_center_xy=(0.0, 0.0),
                selected_wells=((0, 1), (0, 1)),
            ),
            time_plan={"interval": 0.1, "loops": 2},
            channels=["DAPI"],
            grid_plan=useq.GridRowsColumns(rows=1, columns=2),
        ),
        expected_dim_names=["p", "t", "c", "y", "x"],
        expected_positions=[
            ExpectedPosition(name="A1", plate_row="A", plate_col="1"),
            ExpectedPosition(name="B2", plate_row="B", plate_col="2"),
        ],
        id="well_plate_with_seq_grid",
    ),
    Case(
        seq=useq.MDASequence(
            axis_order="ptcz",
            stage_positions=useq.WellPlatePlan(
                plate=useq.WellPlate.from_str("96-well"),
                a1_center_xy=(0.0, 0.0),
                selected_wells=((0,), (0,)),
                well_points_plan=useq.GridRowsColumns(rows=1, columns=2),
            ),
            time_plan={"interval": 0.1, "loops": 2},
            channels=["DAPI"],
            grid_plan=useq.GridRowsColumns(rows=2, columns=1),
        ),
        expected_dim_names=["p", "t", "c", "y", "x"],
        expected_positions=[
            ExpectedPosition("A1_0000", "A", "1", grid_row=0, grid_col=0),
            ExpectedPosition("A1_0001", "A", "1", grid_row=0, grid_col=1),
        ],
        id="well_plate_both_grids",
    ),
]


@pytest.mark.parametrize("case", SEQ_CASES, ids=lambda c: c.id)
def test_useq_to_dims(case: Case) -> None:
    """Test dims_from_useq with different position configurations."""
    from ome_writers._schema import PositionDimension

    seq = case.seq
    pix_size = 0.103
    img_count = 64
    dims = dims_from_useq(
        seq, image_width=img_count, image_height=img_count, pixel_size_um=pix_size
    )

    # Check dimension names
    assert [dim.name for dim in dims] == case.expected_dim_names

    # Check units and scales for non-position dimensions
    for dim in dims:
        if dim.name == "t" and seq.time_plan:
            assert dim.unit == "second"
            assert dim.scale == seq.time_plan.interval.total_seconds()
            assert dim.count == seq.time_plan.num_timepoints()
        elif dim.name == "z" and seq.z_plan:
            assert dim.unit == "micrometer"
            assert dim.scale == seq.z_plan.step
            assert dim.count == seq.z_plan.num_positions()
        elif dim.name in ("y", "x"):
            assert dim.unit == "micrometer"
            assert dim.scale == pix_size
            assert dim.count == img_count
        elif dim.name == "c":
            assert dim.unit is None
            assert dim.scale == 1.0
            assert dim.count == len(seq.channels)

    events = list(case.seq)
    settings = AcquisitionSettings(dimensions=dims, root_path="", dtype="u2")
    assert settings.num_frames == len(events)

    pos_dim = next((d for d in dims if isinstance(d, PositionDimension)), None)
    if case.expected_positions is None:
        assert pos_dim is None
        return

    assert pos_dim is not None

    # Verify that number of positions matches unique (p,g) combinations
    unique_pg = {(e.index.get("p", 0), e.index.get("g")) for e in events}
    assert len(pos_dim.positions) == len(unique_pg), (
        f"Position count mismatch: dims_from_useq created {len(pos_dim.positions)} "
        f"positions but useq iteration has {len(unique_pg)} unique (p,g) combos"
    )

    # Check all position attributes
    for pos, exp_pos in zip(pos_dim.positions, case.expected_positions, strict=True):
        assert pos.name == exp_pos.name
        assert pos.plate_row == exp_pos.plate_row
        assert pos.plate_column == exp_pos.plate_col
        assert pos.grid_row == exp_pos.grid_row
        assert pos.grid_column == exp_pos.grid_col


# Ragged dimension test cases
# Format: (sequence, error_match_pattern)
RAGGED_CASES = [
    pytest.param(
        useq.MDASequence(
            channels=[
                useq.Channel(config="DAPI", do_stack=True, exposure=1),
                useq.Channel(config="Cy5", do_stack=False, exposure=1),
            ],
            z_plan={"range": 4, "step": 1.0},
        ),
        r"mixed Channel\.do_stack values are not supported",
        id="mixed_do_stack",
    ),
    pytest.param(
        useq.MDASequence(
            axis_order="pgtcz",
            stage_positions=[
                useq.Position(
                    x=0.0,
                    y=0.0,
                    name="pos1",
                    sequence=useq.MDASequence(
                        grid_plan=useq.GridRowsColumns(rows=2, columns=2)
                    ),
                ),
                useq.Position(
                    x=10.0,
                    y=10.0,
                    name="pos2",
                    sequence=useq.MDASequence(z_plan={"range": 4, "step": 1.0}),
                ),
            ],
            channels=["DAPI"],
        ),
        "Ragged dimensions detected",
        id="position_subsequences",
    ),
]


@pytest.mark.parametrize(("seq", "error_pattern"), RAGGED_CASES)
def test_ragged_dimensions(seq: useq.MDASequence, error_pattern: str) -> None:
    """Test that ragged dimension cases raise NotImplementedError."""
    with pytest.raises(NotImplementedError, match=error_pattern):
        dims_from_useq(seq, image_width=64, image_height=64)
