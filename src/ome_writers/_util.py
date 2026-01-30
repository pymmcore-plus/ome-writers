from __future__ import annotations

import re
from collections import defaultdict
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
        If the sequence would produce ragged (non-rectangular) dimensions.
    """
    try:
        from useq import Axis, MDASequence
    except ImportError:
        raise ValueError("seq must be a useq.MDASequence") from None

    if not isinstance(seq, MDASequence):  # pragma: no cover
        raise ValueError("seq must be a useq.MDASequence")

    # FIXME:
    # We are doing a little magic interpretation here ... mostly due to useq v1
    # limitations handling unbounded dimensions:
    # if you try to call seq.sizes on an unbounded sequence (e.g. time_plan with
    # duration=3 and interval=0), it raises ZeroDivisionError.
    # We might be able to add more magic to cast that to an unbounded acquisition as
    # long as time is the first dimension... but that's too fragile.
    # we need a better way and it probably means using useq.v2. only
    try:
        _ = seq.sizes
        used_axes = seq.used_axes
        events = _validate_events_not_ragged(seq)
    except ZeroDivisionError:
        raise NotImplementedError(
            "Failed to determine dimension sizes from sequence. "
            "This usually happens when the sequence has unbounded dimensions "
            "(e.g. time_plan with duration and interval=0). "
            "Unbounded useq sequences are not yet supported."
        ) from None

    units = units or {}
    dims: list[Dimension | PositionDimension] = []
    for ax_name in seq.axis_order:
        if ax_name not in used_axes:
            continue
        if ax_name == Axis.POSITION:
            positions = _build_positions_from_events(seq, events)
            dims.append(StandardAxis.POSITION.to_dimension(positions=positions))
        elif ax_name == Axis.GRID:
            # Grid-only: create position dimension from grid points
            if Axis.POSITION not in used_axes:
                positions = _build_positions_from_events(seq, events)
                dims.append(StandardAxis.POSITION.to_dimension(positions=positions))
            # If position is used, it's already handled above
        else:
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
                            dim.scale = seq.time_plan.interval.total_seconds()
                            dim.unit = "second"
                    elif std_axis == StandardAxis.Z and seq.z_plan:
                        # ZAbsolutePositions/ZRelativePositions don't have step
                        dim.unit = "micrometer"
                        if hasattr(seq.z_plan, "step"):
                            dim.scale = seq.z_plan.step
            dims.append(dim)

    dims.extend(
        [
            StandardAxis.Y.to_dimension(count=image_height, scale=pixel_size_um),
            StandardAxis.X.to_dimension(count=image_width, scale=pixel_size_um),
        ]
    )

    return dims


def _validate_events_not_ragged(seq: useq.MDASequence) -> list[useq.MDAEvent]:
    """Validate that the sequence produces rectangular (non-ragged) dimensions.

    Parameters
    ----------
    seq : useq.MDASequence
        The sequence to validate.

    Raises
    ------
    NotImplementedError
        If the sequence would produce ragged dimensions or if dimension order
        cannot be guaranteed to match frame arrival order.
    """
    from useq import Axis

    # If both position and grid are in axis_order, they must be adjacent
    # We flatten (p,g) into a single position dimension, which requires them to be
    # next to each other in iteration order to maintain frame order correctness
    if Axis.POSITION in seq.axis_order and Axis.GRID in seq.axis_order:
        p_idx = seq.axis_order.index(Axis.POSITION)
        g_idx = seq.axis_order.index(Axis.GRID)
        if abs(p_idx - g_idx) != 1:
            raise NotImplementedError(
                f"Cannot handle axis_order={seq.axis_order} with non-adjacent position "
                "and grid axes. We flatten (p,g) into a single position dimension, "
                "which requires them to be adjacent in iteration order."
            )

    # Channel.do_stack=False with z_plan creates ragged z dimension
    if seq.z_plan and seq.channels:
        do_stack_values = {c.do_stack for c in seq.channels}
        if len(do_stack_values) > 1:
            raise NotImplementedError(
                "Sequences with mixed Channel.do_stack values are not supported. "
                "This creates ragged dimensions where different channels have "
                "different z-stack sizes."
            )
        # All do_stack=False means only middle z is acquired - mismatch with z_plan
        if do_stack_values == {False}:
            raise NotImplementedError(
                "Sequences where all channels have do_stack=False are not supported "
                "when z_plan is specified. This acquires only the middle z-plane, "
                "which doesn't match the z_plan dimensions."
            )

    # Channel.acquire_every > 1 creates ragged time dimension per channel
    if seq.time_plan and seq.channels:
        acquire_every_values = {c.acquire_every for c in seq.channels}
        if acquire_every_values != {1}:
            raise NotImplementedError(
                "Sequences with Channel.acquire_every > 1 are not supported. "
                "This creates ragged dimensions where different channels have "
                "different numbers of timepoints."
            )

    # Validate by actually iterating through events
    # This catches all forms of raggedness including position subsequences
    dims_per_position = defaultdict(lambda: {"t": set(), "c": set(), "z": set()})
    events = list(seq)
    for event in events:
        # Track which t/c/z indices this (p, g) sees
        pos_key = (event.index.get("p", 0), event.index.get("g"))
        dims_per_position[pos_key]["t"].add(event.index.get("t", 0))
        dims_per_position[pos_key]["c"].add(event.index.get("c", 0))
        dims_per_position[pos_key]["z"].add(event.index.get("z", 0))

    # All positions should have the same number of t/c/z values
    dim_sizes = [
        (len(dims["t"]), len(dims["c"]), len(dims["z"]))
        for dims in dims_per_position.values()
    ]
    unique_sizes = set(dim_sizes)
    if len(unique_sizes) > 1:
        raise NotImplementedError(
            "Ragged dimensions detected: different positions have different "
            f"dimensionality. Found dimension sizes (t,c,z): {unique_sizes}. "
            "This is not supported."
        )
    return events


def _build_positions_from_events(
    seq: useq.MDASequence, events: list[useq.MDAEvent]
) -> list[Position]:
    """Build Position list by observing useq iteration."""
    from useq import Axis, WellPlatePlan

    is_well_plate = isinstance(seq.stage_positions, WellPlatePlan)
    seq_grid_list = list(seq.grid_plan) if seq.grid_plan else None
    pos_grids = {}
    well_points_list = None
    if is_well_plate:
        well_points_list = list(seq.stage_positions.well_points_plan)
    else:
        for i, pos in enumerate(seq.stage_positions):
            if pos.sequence and pos.sequence.grid_plan:
                pos_grids[i] = list(pos.sequence.grid_plan)

    # Determine sort order: if grid comes before position, sort by (g,p), else (p,g)
    grid_first = False
    if Axis.GRID in seq.axis_order:
        g_idx = seq.axis_order.index(Axis.GRID)
        p_idx = (
            seq.axis_order.index(Axis.POSITION)
            if Axis.POSITION in seq.axis_order
            else float("inf")
        )
        grid_first = g_idx < p_idx

    # Determine if this is a grid-only sequence (no position dimension)
    has_position_dim = Axis.POSITION in seq.axis_order and seq.sizes.get(
        Axis.POSITION, 0
    )

    seen = {}
    for event in events:
        p = event.index.get("p", 0)
        g = event.index.get("g")
        key = (g, p) if grid_first else (p, g)
        if key not in seen:
            p_idx, g_idx = p, g
            plate_row, plate_col = None, None
            if event.pos_name not in (None, "None"):
                name = event.pos_name
                if is_well_plate:
                    well_name = event.pos_name.split("_")[0]
                    if match := re.match(r"([A-Za-z]+)(\d+)", well_name):
                        plate_row, plate_col = match.groups()
            elif g_idx is not None:
                # For grid points, format name to include both p and g indices
                # For grid-only (no position dim), use just the grid index
                if has_position_dim:
                    name = f"{p_idx:04d}"
                else:
                    name = f"{g_idx:04d}"
            else:
                name = str(p_idx)

            grid_row, grid_col = _extract_grid_coords(
                p_idx, g_idx, is_well_plate, seq_grid_list, well_points_list, pos_grids
            )
            seen[key] = Position(
                name=name,
                plate_row=plate_row,
                plate_column=plate_col,
                grid_row=grid_row,
                grid_column=grid_col,
            )

    return [seen[k] for k in sorted(seen.keys())]


def _extract_grid_coords(
    p_idx: int,
    g_idx: int | None,
    is_well_plate: bool,
    seq_grid_list: list | None,
    well_points_list: list | None,
    pos_grids: dict[int, list],
) -> tuple[int | None, int | None]:
    """Extract grid row/col from grid lists."""
    if g_idx is None:
        if is_well_plate and well_points_list:
            point_idx = p_idx % len(well_points_list)
            if point_idx < len(well_points_list):
                if pos := well_points_list[point_idx]:
                    return pos.row, pos.col
        return None, None

    if seq_grid_list and g_idx < len(seq_grid_list):
        return seq_grid_list[g_idx].row, seq_grid_list[g_idx].col

    if (grid_list := pos_grids.get(p_idx)) and g_idx < len(grid_list):
        return grid_list[g_idx].row, grid_list[g_idx].col

    return None, None
