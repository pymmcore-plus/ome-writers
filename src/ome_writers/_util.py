from __future__ import annotations

from itertools import product
from typing import TYPE_CHECKING, cast

import numpy as np
import numpy.typing as npt

from ome_writers._schema import (
    Dimension,
    Position,
    PositionDimension,
    StandardAxis,
    dims_from_standard_axes,
)

if TYPE_CHECKING:
    from collections.abc import Iterator, Mapping
    from typing import TypeAlias

    import useq
    from useq import WellPlatePlan


def fake_data_for_sizes(
    sizes: Mapping[str, int],
    *,
    dtype: npt.DTypeLike = np.uint16,
    chunk_sizes: Mapping[str, int] | None = None,
) -> tuple[Iterator[np.ndarray], list[Dimension | PositionDimension], np.dtype]:
    """Simple helper function to create a data generator and dimensions.

    Provide the sizes of the dimensions you would like to "acquire", along with the
    datatype and chunk sizes. The function will return a generator that yields
    2-D (YX) planes of data, along with the dimension information and the dtype.

    This can be passed to create_stream to create a stream for writing data.

    Parameters
    ----------
    sizes : Mapping[str, int]
        A mapping of dimension labels to their sizes. Must include 'y' and 'x'.
    dtype : np.typing.DTypeLike, optional
        The data type of the generated data. Defaults to np.uint16.
    chunk_sizes : Mapping[str, int] | None, optional
        A mapping of dimension labels to their chunk sizes. If None, defaults to 1 for
        all dimensions, besizes 'y' and 'x', which default to their full sizes.
    """
    if not {"y", "x"} <= sizes.keys():  # pragma: no cover
        raise ValueError("sizes must include both 'y' and 'x'")

    dims = dims_from_standard_axes(sizes=sizes, chunk_shapes=chunk_sizes)

    shape = [d.count for d in dims]
    if any(x is None for x in shape):  # pragma: no cover
        raise ValueError("This function does not yet support unbounded dimensions.")

    dtype = np.dtype(dtype)
    if not np.issubdtype(dtype, np.integer):  # pragma: no cover
        raise ValueError(f"Unsupported dtype: {dtype}.  Must be an integer type.")

    # rng = np.random.default_rng()
    # data = rng.integers(0, np.iinfo(dtype).max, size=shape, dtype=dtype)
    data = np.ones(shape, dtype=dtype)  # type: ignore

    def _build_plane_generator() -> Iterator[np.ndarray]:
        """Yield 2-D planes in y-x order."""
        i = 0
        if not (non_spatial_sizes := shape[:-2]):  # it's just a 2-D image
            yield data
        else:
            for idx in product(*(range(cast("int", n)) for n in non_spatial_sizes)):
                yield data[idx] * i
                i += 1

    return _build_plane_generator(), dims, dtype


# UnitTuple is a tuple of (scale, unit); e.g. (1, "s")
UnitTuple: TypeAlias = tuple[float, str]


def dims_from_useq(
    seq: useq.MDASequence,
    image_width: int,
    image_height: int,
    units: Mapping[str, UnitTuple | None] | None = None,
    pixel_size_um: float | None = None,
) -> list[Dimension | PositionDimension]:
    """Convert a useq.MDASequence to a list of Dimensions for ome-writers.

    Not all useq sequences are supported. The following restrictions apply:

    - Position and grid axes must be adjacent in axis_order (e.g., "pgcz" or "gptc",
      not "ptgc"). They are flattened into a single position dimension.
    - When both stage_positions and grid_plan are specified, position must come before
      grid in axis_order (e.g., "pgtcz" not "gptcz"). Grid-first is only supported
      when using grid_plan alone without stage_positions.
    - Position subsequences may only contain a grid_plan, not time/channel/z plans.
      Different positions may have different grid sizes.
    - All channels must have the same `do_stack` value when a z_plan is present.
      Mixed do_stack or all do_stack=False with z_plan is not supported.
    - All channels must have `acquire_every=1`. Skipping timepoints creates ragged
      dimensions.
    - Unbounded time plans (duration with interval=0) are not supported.
    - WellPlatePlan cannot be combined with an outer grid_plan. Use
      `well_points_plan` on the WellPlatePlan instead.

    Parameters
    ----------
    seq : useq.MDASequence
        The `useq.MDASequence` to convert.
    image_width : int
        The expected width of the images in the stream.
    image_height : int
        The expected height of the images in the stream.
    units : Mapping[str, UnitTuple | None] | None, optional
        An optional mapping of dimension labels to their units.
    pixel_size_um : float | None, optional
        The size of a pixel in micrometers. If provided, it will be used to set the
        scale for the spatial dimensions.

    Raises
    ------
    NotImplementedError
        If the sequence contains any of the unsupported patterns listed above.
    """
    try:
        from useq import Axis, MDASequence
    except ImportError:
        raise ValueError("seq must be a useq.MDASequence") from None

    if not isinstance(seq, MDASequence):  # pragma: no cover
        raise ValueError("seq must be a useq.MDASequence")

    # validate all of the rules mentioned in the docstring: squareness, etc...
    _validate_sequence(seq)

    units = units or {}
    dims: list[Dimension | PositionDimension] = []
    position_dim_added = False
    used_axes = seq.used_axes

    # Check if we have position-like content even if 'p' is not in used_axes
    # (e.g., grid_plan creates positions but may only show 'g' in used_axes)
    has_positions = (
        Axis.POSITION in used_axes or Axis.GRID in used_axes
        # or bool(seq.stage_positions)
    )

    # Build dimensions in axis_order (slowest to fastest)
    # skipping unused axes, (size=0)
    for ax_name in seq.axis_order:
        if ax_name not in used_axes:
            continue

        if ax_name in (Axis.POSITION, Axis.GRID):
            if not position_dim_added and has_positions:
                if positions := _build_positions(seq):
                    dims.append(StandardAxis.POSITION.to_dimension(positions=positions))
                    position_dim_added = True
            continue

        # Build dimension for t, c, z
        std_axis = StandardAxis(str(ax_name))
        dim = std_axis.to_dimension(count=seq.sizes[ax_name], scale=1)
        if isinstance(dim, Dimension):
            if unit := units.get(str(ax_name)):
                dim.scale, dim.unit = unit
            else:
                # Default units for known axes
                if std_axis == StandardAxis.TIME and seq.time_plan:
                    # MultiPhaseTimePlan doesn't have interval attribute
                    if hasattr(seq.time_plan, "interval"):
                        dim.scale = seq.time_plan.interval.total_seconds()  # ty: ignore
                        dim.unit = "second"
                elif std_axis == StandardAxis.Z and seq.z_plan:
                    # ZAbsolutePositions/ZRelativePositions don't have step
                    dim.unit = "micrometer"
                    if hasattr(seq.z_plan, "step"):
                        dim.scale = seq.z_plan.step  # ty: ignore
        dims.append(dim)

    dims.extend(
        [
            StandardAxis.Y.to_dimension(count=image_height, scale=pixel_size_um),
            StandardAxis.X.to_dimension(count=image_width, scale=pixel_size_um),
        ]
    )

    return dims


def _validate_sequence(seq: useq.MDASequence) -> None:
    """Validate sequence for supported patterns (without iterating events).

    Raises
    ------
    NotImplementedError
        If the sequence contains unsupported patterns.
    """
    from useq import Axis, WellPlatePlan

    # Check for unbounded dimensions (e.g., DurationInterval with interval=0)
    try:
        _ = seq.sizes
    except ZeroDivisionError:
        raise NotImplementedError(
            "Failed to determine dimension sizes from sequence. "
            "This usually happens when the sequence has unbounded dimensions "
            "(e.g. time_plan with duration and interval=0). "
            "Unbounded useq sequences are not yet supported."
        ) from None

    # Check P/G adjacency in axis_order
    if Axis.POSITION in seq.axis_order and Axis.GRID in seq.axis_order:
        p_idx = seq.axis_order.index(Axis.POSITION)
        g_idx = seq.axis_order.index(Axis.GRID)
        if abs(p_idx - g_idx) != 1:
            raise NotImplementedError(
                f"Cannot handle axis_order={seq.axis_order} with non-adjacent position "
                "and grid axes. We flatten (p,g) into a single position dimension, "
                "which requires them to be adjacent in iteration order."
            )

    # Check do_stack uniformity when z_plan exists
    if seq.z_plan and seq.channels:
        do_stack_values = {c.do_stack for c in seq.channels}
        if len(do_stack_values) > 1:
            raise NotImplementedError(
                "Sequences with mixed Channel.do_stack values are not supported. "
                "This creates ragged dimensions where different channels have "
                "different z-stack sizes."
            )
        if do_stack_values == {False}:
            raise NotImplementedError(
                "Sequences where all channels have do_stack=False are not supported "
                "when z_plan is specified. This acquires only the middle z-plane, "
                "which doesn't match the z_plan dimensions."
            )

    # Check acquire_every == 1 for all channels
    if seq.channels:
        acquire_every_values = {c.acquire_every for c in seq.channels}
        if acquire_every_values != {1}:
            raise NotImplementedError(
                "Sequences with Channel.acquire_every > 1 are not supported. "
                "This creates ragged dimensions where different channels have "
                "different numbers of timepoints."
            )

    # Check WellPlatePlan + grid_plan (use well_points_plan instead)
    is_well_plate = isinstance(seq.stage_positions, WellPlatePlan)
    if is_well_plate and seq.grid_plan:
        raise NotImplementedError(
            "WellPlatePlan with grid_plan is not supported. "
            "Use well_points_plan on the WellPlatePlan instead."
        )

    # Check position subsequences only contain grid_plan (no t/c/z)
    if not is_well_plate:
        for pos in seq.stage_positions:
            if sub := pos.sequence:
                if sub.time_plan or sub.channels or sub.z_plan:
                    raise NotImplementedError(
                        "Ragged dimensions detected: different positions have "
                        "different dimensionality. Found dimension sizes (t,c,z): "
                        "Position subsequences may only contain grid_plan."
                    )
                if sub.grid_plan:
                    pass

    # Check grid-first ordering when both positions and grid exist
    has_both = bool(seq.stage_positions) and seq.grid_plan is not None
    if has_both and Axis.GRID in seq.axis_order and Axis.POSITION in seq.axis_order:
        g_idx = seq.axis_order.index(Axis.GRID)
        p_idx = seq.axis_order.index(Axis.POSITION)
        if g_idx < p_idx:
            raise NotImplementedError(
                "Grid-first ordering (grid before position in axis_order) is not "
                "supported when both stage_positions and grid_plan are specified. "
                "Change axis_order so position comes before grid "
                "(e.g., 'pgtcz' instead of 'gptcz')."
            )


def _build_positions(seq: useq.MDASequence) -> list[Position]:
    """Build Position list from sequence without iterating events.

    If we've reached this function, we can assume that the sequence has stage_positions
    or grid_plan defined and that they are adjacent in the axis_order.
    """
    from useq import WellPlatePlan

    # Case 1: WellPlatePlan
    # we've previously asserted that seq.grid_plan is None.
    if isinstance(seq.stage_positions, WellPlatePlan):
        return _build_well_plate_positions(seq.stage_positions)

    # Case 3: Stage positions (with optional global grid_plan)
    if seq.stage_positions:
        return _build_stage_positions_plan(seq)

    # Case 2: Grid plan only (no stage_positions)
    elif seq.grid_plan is not None:
        return [
            Position(
                name=f"{i:04d}",
                grid_row=getattr(gp, "row", None),
                grid_column=getattr(gp, "col", None),
            )
            for i, gp in enumerate(seq.grid_plan)
        ]

    return []


def _build_well_plate_positions(plate_plan: WellPlatePlan) -> list[Position]:
    """Return Positions in WellPlatePlan in order of visit."""
    from useq import RelativePosition

    # iterating over plate_plan yields AbsolutePosition objects for every
    # field of view, for every well, with absolute offsets applied.
    # it's 1-to-1 with the Positions we want to create...
    # however, their row/column provenance is not preserved,
    # So we do our own iteration of selected_well_indices to get that info.
    plate_iter = iter(plate_plan)

    # the well_points_plan is an iterator of RelativePosition objects, explaining
    # how to iterate within each well.
    # it's *included* in the iteration of plate_plan above, but this is the only
    # way to get the grid_row/grid_column info for each position in a well.
    # It could be one of:
    # GridRowsColumns | GridWidthHeight | RandomPoints | RelativePosition
    wpp = plate_plan.well_points_plan
    well_positions: list[RelativePosition] = (
        [wpp] if isinstance(wpp, RelativePosition) else list(wpp)
    )

    positions: list[Position] = []
    for row_idx, col_idx in plate_plan.selected_well_indices:
        plate_row = _row_idx_to_letter(row_idx)
        plate_column = str(col_idx + 1)
        for well_pos in well_positions:
            pos = next(plate_iter)  # grab the next AbsolutePosition in the outer loop
            positions.append(
                Position(
                    name=pos.name,
                    plate_row=plate_row,
                    plate_column=plate_column,
                    grid_row=getattr(well_pos, "row", None),
                    grid_column=getattr(well_pos, "col", None),
                    x_coord=pos.x,
                    y_coord=pos.y,
                )
            )

    return positions


def _row_idx_to_letter(index: int) -> str:
    """Convert 0-based row index to letter (0->A, 1->B, ..., 25->Z, 26->AA)."""
    name = ""
    while index >= 0:
        name = chr(index % 26 + 65) + name
        index = index // 26 - 1
    return name


def _build_stage_positions_plan(seq: useq.MDASequence) -> list[Position]:
    """Build positions from stage_positions with optional grid plans.

    Handles three cases:
    - Positions with subsequence grids (use subsequence grid)
    - Positions without subsequence but with global grid (use global grid)
    - Positions without any grid (single position)

    Always iterates position-first (grid-first with positions is not supported).
    """
    global_grid = list(seq.grid_plan) if seq.grid_plan else None

    positions: list[Position] = []
    for p_idx, pos in enumerate(seq.stage_positions):
        name = pos.name if pos.name else str(p_idx)

        # Determine which grid to use for this position
        # Priority: subsequence grid > global grid > no grid
        if pos.sequence and pos.sequence.grid_plan:
            grid = pos.sequence.grid_plan
        elif global_grid:
            grid = global_grid
        else:
            grid = None

        if grid:
            for gp in grid:
                # if this line ever raises an exception,
                # break it into two parts:
                # 1. create position, 2. try to add coords, suppressing errors.
                pos_sum = pos + gp  # type: ignore [operator]
                positions.append(
                    Position(
                        name=name,
                        grid_row=getattr(gp, "row", None),
                        grid_column=getattr(gp, "col", None),
                        x_coord=pos_sum.x,
                        y_coord=pos_sum.y,
                        z_coord=pos_sum.z,
                    )
                )
        else:
            positions.append(
                Position(name=name, x_coord=pos.x, y_coord=pos.y, z_coord=pos.z)
            )

    return positions
