from __future__ import annotations

from dataclasses import dataclass

import pytest

from ome_writers._schema import AcquisitionSettings
from ome_writers._useq import dims_from_useq

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
            axis_order="pgtcz",
            stage_positions=[
                useq.Position(x=0.0, y=0.0, name="single_pos"),
                useq.Position(
                    name="grid",
                    sequence=useq.MDASequence(grid_plan={"rows": 1, "columns": 2}),
                ),
            ],
            grid_plan=useq.GridRowsColumns(rows=2, columns=1),
            z_plan={"range": 2, "step": 1.0},
        ),
        expected_dim_names=["p", "z", "y", "x"],
        expected_positions=[
            ExpectedPosition(name="single_pos", grid_row=0, grid_col=0),
            ExpectedPosition(name="single_pos", grid_row=1, grid_col=0),
            ExpectedPosition(name="grid", grid_row=0, grid_col=0),
            ExpectedPosition(name="grid", grid_row=0, grid_col=1),
        ],
        id="subsequence_mixed_with_global_grid",
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
            axis_order="gtc",  # Grid with no explicit position
            grid_plan=useq.GridRowsColumns(rows=1, columns=2),
            time_plan={"interval": 0.1, "loops": 2},
            channels=["DAPI"],
        ),
        expected_dim_names=["p", "t", "c", "y", "x"],  # Position dim should be created
        expected_positions=[
            ExpectedPosition(name="0000", grid_row=0, grid_col=0),
            ExpectedPosition(name="0001", grid_row=0, grid_col=1),
        ],
        id="grid_only_no_position",
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
    # MultiPhaseTimePlan - no single .interval attribute
    Case(
        seq=useq.MDASequence(
            axis_order="tc",
            time_plan=[
                {"interval": 0.1, "loops": 3},
                {"interval": 0.5, "loops": 2},
            ],
            channels=["DAPI"],
        ),
        expected_dim_names=["t", "c", "y", "x"],
        expected_positions=None,
        id="multi_phase_time_plan",
    ),
    # ZAbsolutePositions - no .step attribute
    Case(
        seq=useq.MDASequence(
            axis_order="cz",
            channels=["DAPI"],
            z_plan=useq.ZAbsolutePositions(absolute=[0, 1, 2, 3]),
        ),
        expected_dim_names=["c", "z", "y", "x"],
        expected_positions=None,
        id="z_absolute_positions",
    ),
    Case(
        seq=useq.MDASequence(
            channels=["DAPI"],
            z_plan=useq.ZAbsolutePositions(absolute=[0, 1, 2, 3]),
        ),
        expected_dim_names=["c", "z", "y", "x"],
        expected_positions=None,
        id="axis_order_not_specified",
    ),
    # ZRelativePositions - no .step attribute
    Case(
        seq=useq.MDASequence(
            axis_order="cz",
            channels=["DAPI"],
            z_plan=useq.ZRelativePositions(relative=[-1, 0, 1]),
        ),
        expected_dim_names=["c", "z", "y", "x"],
        expected_positions=None,
        id="z_relative_positions",
    ),
    # RandomPoints grid - row/col are None
    Case(
        seq=useq.MDASequence(
            axis_order="gc",
            grid_plan=useq.RandomPoints(
                num_points=3,
                max_width=100,
                max_height=100,
                shape="ellipse",
                random_seed=42,
            ),
            channels=["DAPI"],
        ),
        expected_dim_names=["p", "c", "y", "x"],
        expected_positions=[
            ExpectedPosition(name="0000", grid_row=None, grid_col=None),
            ExpectedPosition(name="0001", grid_row=None, grid_col=None),
            ExpectedPosition(name="0002", grid_row=None, grid_col=None),
        ],
        id="random_points_grid",
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
            assert dim.count == seq.time_plan.num_timepoints()
            # MultiPhaseTimePlan doesn't have .interval attribute
            if hasattr(seq.time_plan, "interval"):
                assert dim.unit == "second"
                assert dim.scale == seq.time_plan.interval.total_seconds()  # ty: ignore
        elif dim.name == "z" and seq.z_plan:
            # ZAbsolutePositions/ZRelativePositions don't have .step attribute
            assert dim.unit == "micrometer"
            if hasattr(seq.z_plan, "step"):
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
        "Sequences with Channel.do_stack=False values are not supported",
        id="mixed_do_stack",
    ),
    # acquire_every > 1 - creates ragged time dimension per channel
    pytest.param(
        useq.MDASequence(
            axis_order="tc",
            time_plan={"interval": 0.1, "loops": 4},
            channels=[
                useq.Channel(config="DAPI", acquire_every=1),
                useq.Channel(config="Cy5", acquire_every=2),
            ],
        ),
        r"acquire_every",
        id="acquire_every_mixed",
    ),
    # All acquire_every > 1 (uniform) - still skips frames
    pytest.param(
        useq.MDASequence(
            axis_order="tc",
            time_plan={"interval": 0.1, "loops": 4},
            channels=[
                useq.Channel(config="DAPI", acquire_every=2),
                useq.Channel(config="Cy5", acquire_every=2),
            ],
        ),
        r"acquire_every",
        id="acquire_every_uniform",
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
    pytest.param(
        useq.MDASequence(
            axis_order="gtpc",  # Grid and position NOT adjacent!
            stage_positions=[(0.0, 0.0), (10.0, 10.0)],
            grid_plan=useq.GridRowsColumns(rows=1, columns=2),
            time_plan={"interval": 0.1, "loops": 2},
            channels=["DAPI"],
        ),
        "non-adjacent position and grid axes",
        id="grid_position_not_adjacent",
    ),
    pytest.param(
        useq.MDASequence(
            axis_order="ptgcz",
            stage_positions=[(0.0, 0.0), (10.0, 10.0)],
            time_plan={"interval": 0.1, "loops": 3},
            channels=["DAPI", "Cy5"],
            z_plan={"range": 2, "step": 1.0},
            grid_plan=useq.GridRowsColumns(rows=1, columns=2),
        ),
        "non-adjacent position and grid axes",
        id="position_grid_not_adjacent_ptgcz",
    ),
    pytest.param(
        useq.MDASequence(
            axis_order="ptgzc",
            stage_positions=[(0.0, 0.0), (10.0, 10.0)],
            time_plan={"interval": 0.1, "loops": 3},
            channels=["DAPI", "Cy5"],
            z_plan={"range": 2, "step": 1.0},
            grid_plan=useq.GridRowsColumns(rows=1, columns=2),
        ),
        "non-adjacent position and grid axes",
        id="position_grid_not_adjacent_ptgzc",
    ),
    pytest.param(
        useq.MDASequence(
            z_plan={"range": 2, "step": 1.0},
            time_plan={"interval": 0, "duration": 3},
        ),
        "Unbounded useq sequences are not yet supported",
        id="t_duration_no_interval",
    ),
    pytest.param(
        useq.MDASequence(
            stage_positions=useq.WellPlatePlan(
                plate=useq.WellPlate.from_str("96-well"),
                a1_center_xy=(0.0, 0.0),
                selected_wells=((0,), (0,)),
            ),
            grid_plan=useq.GridRowsColumns(rows=1, columns=2),
            channels=["DAPI"],
        ),
        "WellPlatePlan with grid_plan is not supported",
        id="well_plate_with_grid_plan",
    ),
    # Grid-first ordering with both positions and grid_plan
    pytest.param(
        useq.MDASequence(
            axis_order="gptcz",
            stage_positions=[(0.0, 0.0), (10.0, 10.0)],
            grid_plan=useq.GridRowsColumns(rows=1, columns=2),
            channels=["DAPI"],
        ),
        "Grid-first ordering.*is not supported",
        id="grid_first_with_positions",
    ),
]


@pytest.mark.parametrize(("seq", "error_pattern"), RAGGED_CASES)
def test_unsupported_sequences_raise(seq: useq.MDASequence, error_pattern: str) -> None:
    """Test that ragged dimension cases raise NotImplementedError."""
    with pytest.raises(NotImplementedError, match=error_pattern):
        dims_from_useq(seq, image_width=64, image_height=64)


def test_useq_manual_units() -> None:
    """Test that manual units in useq sequence are respected."""
    seq = useq.MDASequence(
        axis_order="tzc",
        time_plan={"interval": 1, "loops": 3},
        channels=["DAPI"],
    )

    dims = dims_from_useq(
        seq,
        image_width=64,
        image_height=64,
        units={
            "t": (
                1.0,
                "minute",
            )
        },
    )

    time_dim = next(dim for dim in dims if dim.type == "time")
    assert time_dim.unit == "minute"
    assert time_dim.scale == 1.0
    assert time_dim.count == 3
