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
        scale for the spatial dimensions.
    """
    try:
        from useq import Axis, MDASequence
    except ImportError:
        # if we can't import MDASequence, then seq must not be a MDASequence
        raise ValueError("seq must be a useq.MDASequence") from None
    else:
        if not isinstance(seq, MDASequence):  # pragma: no cover
            raise ValueError("seq must be a useq.MDASequence")

    if any(pos.sequence for pos in seq.stage_positions):
        raise NotImplementedError(
            "Sequences with position sub-sequences are not supported."
        )

    units = units or {}
    has_grid = seq.grid_plan is not None
    has_positions = bool(seq.stage_positions)
    if has_grid and has_positions:
        raise NotImplementedError(
            "Sequences with both grid plans and stage positions are not yet supported."
        )

    # NOTE: v1 useq schema has a terminal bug:
    # certain MDASequences (e.g. time plans with interval=0) will trigger
    # a ZeroDivisionError on `seq.sizes`.  but they are broken upstream until v2.
    # with v2, we have better ways to look for unbounded dimensions.
    dims: list[Dimension] = []
    for ax_name, size in seq.sizes.items():
        if not size:  # pragma: no cover
            continue

        # convert useq Axis to StandardAxis
        # (they all have the same name except for GRID) ... which we convert to 'p',
        # having asserted above that we don't have both grid and stage positions.
        _ax = "p" if ax_name == Axis.GRID else ax_name
        try:
            std_axis = StandardAxis(_ax)
        except ValueError:  # pragma: no cover
            raise ValueError(f"Unsupported axis for OME: {ax_name}") from None

        dim = std_axis.to_dimension(count=size, scale=1)

        # if units are explicitly provided, set them on the dimension
        if isinstance(dim, Dimension):
            if _unit := units.get(ax_name):
                dim.scale = _unit[0]
                dim.unit = _unit[1]

        dims.append(dim)

    return [
        *dims,
        StandardAxis.Y.to_dimension(count=image_height, scale=pixel_size_um),
        StandardAxis.X.to_dimension(count=image_width, scale=pixel_size_um),
    ]
