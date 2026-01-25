from __future__ import annotations

from itertools import product
from typing import TYPE_CHECKING, cast

import numpy as np
import numpy.typing as npt

from ome_writers._schema import (
    Dimension,
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
        scale for the spatial dimensions..
    """
    try:
        from useq import Axis, MDASequence
    except ImportError:
        raise ValueError("seq must be a useq.MDASequence") from None
    else:
        if not isinstance(seq, MDASequence):  # pragma: no cover
            raise ValueError("seq must be a useq.MDASequence")

    units = units or {}
    has_position_subsequences = any(pos.sequence for pos in seq.stage_positions)

    if has_position_subsequences:
        _validate_position_subsequences(seq)

    combined_positions = _build_combined_positions(seq, has_position_subsequences)
    position_insert_index: int | None = None

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


def _build_combined_positions(
    seq: useq.MDASequence, has_position_subsequences: bool
) -> list[str] | None:
    """Build combined position names for grid+positions cases."""
    from useq import Axis

    if has_position_subsequences:
        combined = []
        for p_idx, pos in enumerate(seq.stage_positions):
            pos_name = pos.name or f"p{p_idx:03d}"
            if pos.sequence and pos.sequence.grid_plan:
                n_grid = pos.sequence.grid_plan.num_positions()
                combined.extend(f"{pos_name}_g{g_idx:03d}" for g_idx in range(n_grid))
            else:
                combined.append(pos_name)
        return combined

    if seq.grid_plan and seq.stage_positions:
        n_positions = seq.sizes.get(Axis.POSITION, 1)
        n_grid = seq.sizes.get(Axis.GRID) or seq.grid_plan.num_positions()
        return [
            f"p{p_idx:03d}g{g_idx:03d}"
            for p_idx in range(n_positions)
            for g_idx in range(n_grid)
        ]

    return None
