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

    events = _validate_events_not_ragged(seq)

    units = units or {}
    dims: list[Dimension | PositionDimension] = []

    for ax_name in seq.axis_order:
        if ax_name == Axis.POSITION:
            positions = _build_positions_from_events(seq, events)
            position_dim = StandardAxis.POSITION.to_dimension(positions=positions)
            dims.append(position_dim)
        elif ax_name != Axis.GRID and seq.sizes.get(ax_name, 0):
            size = seq.sizes.get(ax_name, 0)
            std_axis = StandardAxis(str(ax_name))
            dim = std_axis.to_dimension(count=size, scale=1)
            if isinstance(dim, Dimension):
                if unit := units.get(str(ax_name)):
                    dim.scale, dim.unit = unit
                else:
                    # Default units for known axes
                    if std_axis == StandardAxis.TIME and seq.time_plan:
                        dim.scale = seq.time_plan.interval.total_seconds()
                        dim.unit = "second"
                    elif std_axis == StandardAxis.Z and seq.z_plan:
                        dim.scale = seq.z_plan.step
                        dim.unit = "micrometer"
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
        If the sequence would produce ragged dimensions.
    """

    # Check 1: Channel.do_stack=False with z_plan creates ragged z dimension
    if seq.z_plan and seq.channels:
        do_stack_values = {c.do_stack for c in seq.channels}
        if len(do_stack_values) > 1:
            raise NotImplementedError(
                "Sequences with mixed Channel.do_stack values are not supported. "
                "This creates ragged dimensions where different channels have "
                "different z-stack sizes."
            )

    # Check 2: Validate by actually iterating through events
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
    from useq import WellPlatePlan

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

    seen = {}
    for event in events:
        key = (event.index.get("p", 0), event.index.get("g"))
        if key not in seen:
            p_idx, g_idx = key
            plate_row, plate_col = None, None
            if event.pos_name not in (None, "None"):
                name = event.pos_name
                if is_well_plate:
                    well_name = event.pos_name.split("_")[0]
                    if match := re.match(r"([A-Za-z]+)(\d+)", well_name):
                        plate_row, plate_col = match.groups()
            else:
                name = f"{p_idx:04d}" if g_idx is not None else str(p_idx)

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
