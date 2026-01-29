from __future__ import annotations

import re
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
    """
    try:
        from useq import Axis, MDASequence
    except ImportError:
        raise ValueError("seq must be a useq.MDASequence") from None
    else:
        if not isinstance(seq, MDASequence):  # pragma: no cover
            raise ValueError("seq must be a useq.MDASequence")

    from useq import WellPlatePlan

    units = units or {}

    # WellPlatePlan positions never have subsequences, so skip the check
    is_well_plate = isinstance(seq.stage_positions, WellPlatePlan)
    has_position_subsequences = (
        False if is_well_plate else any(pos.sequence for pos in seq.stage_positions)
    )

    if has_position_subsequences:
        _validate_position_subsequences(seq)

    combined_positions = _build_positions(seq, has_position_subsequences, is_well_plate)
    position_insert_index: int | None = None

    # NOTE: v1 useq schema has a terminal bug:
    # certain MDASequences (e.g. time plans with interval=0) will trigger
    # a ZeroDivisionError on `seq.sizes`.  but they are broken upstream until v2.
    # with v2, we have better ways to look for unbounded dimensions.
    dims: list[Dimension] = []
    for ax_name, size in seq.sizes.items():
        if not size:  # pragma: no cover
            continue

        if combined_positions is not None:
            if ax_name == Axis.POSITION:
                position_insert_index = len(dims)
                continue
            if ax_name == Axis.GRID:
                continue

        _ax = "p" if ax_name == Axis.GRID else ax_name
        try:
            std_axis = StandardAxis(_ax)
        except ValueError:  # pragma: no cover
            raise ValueError(f"Unsupported axis for OME: {ax_name}") from None

        dim = std_axis.to_dimension(count=size, scale=1)

        if isinstance(dim, Dimension) and (_unit := units.get(ax_name)):
            dim.scale = _unit[0]
            dim.unit = _unit[1]

        dims.append(dim)

    if combined_positions is not None:
        insert_idx = position_insert_index if position_insert_index is not None else 0
        dims.insert(
            insert_idx, StandardAxis.POSITION.to_dimension(positions=combined_positions)
        )

    return [
        *dims,
        StandardAxis.Y.to_dimension(count=image_height, scale=pixel_size_um),
        StandardAxis.X.to_dimension(count=image_width, scale=pixel_size_um),
    ]


def _validate_position_subsequences(seq: useq.MDASequence) -> None:
    """Validate that position subsequences only contain grid_plan."""
    for pos in seq.stage_positions:
        if not pos.sequence:
            continue
        if not pos.sequence.grid_plan:
            raise NotImplementedError(
                "Position subsequences without grid_plan are not yet supported."
            )
        # Check that subsequence only contains grid_plan
        has_other = (
            pos.sequence.time_plan is not None
            or pos.sequence.z_plan is not None
            or pos.sequence.channels
            or pos.sequence.stage_positions
        )
        if has_other:
            raise NotImplementedError(
                "Position subsequences with plans other than grid_plan "
                "(e.g., time_plan, z_plan, channels) are not yet supported."
            )


def _build_positions(
    seq: useq.MDASequence, has_position_subsequences: bool, is_well_plate: bool
) -> list[Position] | None:
    """Build Position list from useq stage_positions, handling special cases.

    Handles: WellPlatePlan, position subsequences with grids, and sequence-level grids.
    Returns None for simple stage_positions that don't need special handling.
    """
    if is_well_plate:
        return _build_positions_from_well_plate_plan(seq)
    if has_position_subsequences:
        return _build_positions_from_subsequences(seq)
    if seq.grid_plan and seq.stage_positions:
        return _build_positions_with_grid(seq)
    return None


def _build_positions_from_subsequences(seq: useq.MDASequence) -> list[Position]:
    """Build positions from stage positions that have grid subsequences."""
    combined: list[Position] = []
    for idx, pos in enumerate(seq.stage_positions):
        pos_name = pos.name or f"{idx:04d}"
        plate_row, plate_col = _parse_plate_coords(pos)
        grid_plan = pos.sequence.grid_plan if pos.sequence else None

        if grid_plan:
            combined.extend(
                Position(
                    name=pos_name,
                    plate_row=plate_row,
                    plate_column=plate_col,
                    grid_row=g.row,
                    grid_column=g.col,
                )
                for g in grid_plan
            )
        else:
            combined.append(
                Position(name=pos_name, plate_row=plate_row, plate_column=plate_col)
            )
    return combined


def _build_positions_with_grid(seq: useq.MDASequence) -> list[Position]:
    """Build positions by expanding stage positions with sequence-level grid."""
    if seq.grid_plan is None:
        raise ValueError(
            "MDASequence grid_plan must be defined to build positions with grid."
        )
    return [
        Position(
            name=pos.name or f"{idx:04d}",
            plate_row=plate_row,
            plate_column=plate_col,
            grid_row=g.row,
            grid_column=g.col,
        )
        for idx, pos in enumerate(seq.stage_positions)
        for plate_row, plate_col in [_parse_plate_coords(pos)]
        for g in seq.grid_plan
    ]


def _build_positions_from_well_plate_plan(seq: useq.MDASequence) -> list[Position]:
    """Build positions from a WellPlatePlan, extracting well and grid coordinates."""
    from useq import GridRowsColumns, WellPlatePlan

    wpp = cast("WellPlatePlan", seq.stage_positions)
    well_names = [str(n) for n in wpp.selected_well_names]
    num_points = wpp.num_points_per_well

    # Get sequence-level grid coordinates if present (takes precedence)
    seq_grid = [(g.row, g.col) for g in seq.grid_plan] if seq.grid_plan else None

    # Get grid coordinates from well_points_plan only if no sequence-level grid
    grid_coords: list[tuple[int | None, int | None]] | None = None
    if seq_grid is None and isinstance(wpp.well_points_plan, GridRowsColumns):
        grid_coords = [(g.row, g.col) for g in wpp.well_points_plan]

    combined: list[Position] = []
    for i, pos in enumerate(wpp):
        well_idx, point_idx = divmod(i, num_points)
        well_name = well_names[well_idx]

        # Parse plate row/col from well name (e.g., 'A1' -> 'A', '1')
        match = re.compile(r"([A-Za-z]+)(\d+)").match(well_name)
        plate_row, plate_col = match.groups() if match else (None, None)
        pos_name = pos.name or f"{well_idx:04d}"

        # Expand with sequence-level grid if present, otherwise use well_points grid
        if seq_grid:
            combined.extend(
                Position(
                    name=pos_name,
                    plate_row=plate_row,
                    plate_column=plate_col,
                    grid_row=gr,
                    grid_column=gc,
                )
                for gr, gc in seq_grid
            )
        else:
            grid_row, grid_col = (
                grid_coords[point_idx]
                if grid_coords and point_idx < len(grid_coords)
                else (None, None)
            )
            combined.append(
                Position(
                    name=pos_name,
                    plate_row=plate_row,
                    plate_column=plate_col,
                    grid_row=grid_row,
                    grid_column=grid_col,
                )
            )

    return combined


def _parse_plate_coords(pos: useq.Position) -> tuple[str | None, str | None]:
    """Extract plate_row and plate_column from a useq Position if available."""
    plate_row = getattr(pos, "row", None)
    plate_col = getattr(pos, "col", None)
    if plate_row is not None:
        plate_row = str(plate_row)
    if plate_col is not None:
        plate_col = str(plate_col)
    return plate_row, plate_col
