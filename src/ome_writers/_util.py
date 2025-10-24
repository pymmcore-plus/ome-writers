from __future__ import annotations

import re
from itertools import product
from typing import TYPE_CHECKING, cast, get_args

import numpy as np
import numpy.typing as npt

from ome_writers import Dimension, DimensionLabel

if TYPE_CHECKING:
    from collections.abc import Iterator, Mapping

    import useq

    from ome_writers import Plate, UnitTuple


VALID_LABELS = get_args(DimensionLabel)
DEFAULT_UNITS: Mapping[DimensionLabel, UnitTuple | None] = {
    "t": (1.0, "s"),
    "z": (1.0, "um"),
    "y": (1.0, "um"),
    "x": (1.0, "um"),
    "c": None,
    "p": None,
}


def fake_data_for_sizes(
    sizes: Mapping[str, int],
    *,
    dtype: npt.DTypeLike = np.uint16,
    chunk_sizes: Mapping[str, int] | None = None,
) -> tuple[Iterator[np.ndarray], list[Dimension], np.dtype]:
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
    if not all(k in VALID_LABELS for k in sizes):  # pragma: no cover
        raise ValueError(
            f"Invalid dimension labels in sizes: {sizes.keys() - set(VALID_LABELS)}"
        )

    _chunk_sizes = dict(chunk_sizes or {})
    _chunk_sizes.setdefault("y", sizes["y"])
    _chunk_sizes.setdefault("x", sizes["x"])

    ordered_labels = [z for z in sizes if z not in "yx"]
    ordered_labels += ["y", "x"]
    dims = [
        Dimension(
            label=lbl,
            size=sizes[lbl],
            unit=DEFAULT_UNITS.get(lbl, None),
            chunk_size=_chunk_sizes.get(lbl, 1),
        )
        for lbl in cast("list[DimensionLabel]", ordered_labels)
    ]

    shape = [d.size for d in dims]
    dtype = np.dtype(dtype)
    if not np.issubdtype(dtype, np.integer):  # pragma: no cover
        raise ValueError(f"Unsupported dtype: {dtype}.  Must be an integer type.")

    # rng = np.random.default_rng()
    # data = rng.integers(0, np.iinfo(dtype).max, size=shape, dtype=dtype)
    data = np.ones(shape, dtype=dtype)

    def _build_plane_generator() -> Iterator[np.ndarray]:
        """Yield 2-D planes in y-x order."""
        i = 0
        if not (non_spatial_sizes := shape[:-2]):  # it's just a 2-D image
            yield data
        else:
            for idx in product(*(range(n) for n in non_spatial_sizes)):
                yield data[idx] * i
                i += 1

    return _build_plane_generator(), dims, dtype


def dims_from_useq(
    seq: useq.MDASequence,
    image_width: int,
    image_height: int,
    units: Mapping[str, UnitTuple | None] | None = None,
) -> list[Dimension]:
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
        An optional mapping of dimension labels to their units. If `None`, defaults to
        - "t" -> (1.0, "s")
        - "z" -> (1.0, "um")
        - "y" -> (1.0, "um")
        - "x" -> (1.0, "um")

    Examples
    --------
    A typical usage of ome-writers with useq-schema might look like this:

    ```python
    from ome_writers import create_stream, dims_from_useq

    width, height = however_you_get_expected_image_dimensions()
    dims = dims_from_useq(seq, image_width=width, image_height=height)

    with create_stream(
        path=...,
        dimensions=dims,
        dtype=np.uint16,
        backend=...,
    ) as stream:
        for frame in whatever_generates_your_data():
            stream.append(frame)
    ```
    """
    try:
        from useq import MDASequence
    except ImportError:
        # if we can't import MDASequence, then seq must not be a MDASequence
        raise ValueError("seq must be a useq.MDASequence") from None
    else:
        if not isinstance(seq, MDASequence):
            raise ValueError("seq must be a useq.MDASequence")

    _units: Mapping[str, UnitTuple | None] = {
        **DEFAULT_UNITS,  # type: ignore[dict-item]
        **(units or {}),
    }

    dims: list[Dimension] = []
    for ax, size in seq.sizes.items():
        if size:
            # all of the useq axes are the same as the ones used here.
            dim_label = cast("DimensionLabel", str(ax))
            if dim_label not in _units:
                raise ValueError(f"Unsupported axis for OME: {ax}")
            dims.append(Dimension(label=dim_label, size=size, unit=_units[dim_label]))

    return [
        *dims,
        Dimension(label="y", size=image_height, unit=_units["y"]),
        Dimension(label="x", size=image_width, unit=_units["x"]),
    ]


def plate_from_useq(seq: useq.MDASequence) -> Plate | None:
    """Extract plate metadata from a useq.MDASequence if it uses a WellPlatePlan.

    Parameters
    ----------
    seq : useq.MDASequence
        The `useq.MDASequence` to extract plate metadata from.

    Returns
    -------
    Plate | None
        A Plate object if the sequence uses a WellPlatePlan for stage positions,
        otherwise None.

    Examples
    --------
    ```python
    from useq import MDASequence, WellPlatePlan
    from ome_writers import plate_from_useq

    plate_plan = WellPlatePlan(
        plate="96-well",
        a1_center_xy=(0.0, 0.0),
        selected_wells=([0, 0, 1], [0, 1, 0]),  # A1, A2, B1
    )
    seq = MDASequence(stage_positions=plate_plan)
    plate = plate_from_useq(seq)
    # plate will contain the well layout information
    ```
    """
    try:
        from useq import MDASequence, WellPlatePlan
    except ImportError:
        raise ValueError("seq must be a useq.MDASequence") from None
    else:
        if not isinstance(seq, MDASequence):
            raise ValueError("seq must be a useq.MDASequence")

    from ome_writers import Plate, PlateAcquisition, WellPosition

    # check if stage_positions is a WellPlatePlan
    if not isinstance(seq.stage_positions, WellPlatePlan):
        return None

    plate_plan: WellPlatePlan = seq.stage_positions

    # get selected well names and corresponding indices from the plan
    wells: list[WellPosition] = []
    well_set = set()
    names = plate_plan.selected_well_names
    indices = plate_plan.selected_well_indices

    for name, idx in zip(names, indices, strict=True):
        row_index, col_index = int(idx[0]), int(idx[1])  # e.g. 30, 32
        row_letter, col_name = split_well_id(name)  # name e.g. AE33
        if row_letter is None or col_name is None:
            raise ValueError(f"Invalid well name: {name}")
        well_key = (row_index, col_index)
        if well_key not in well_set:
            well_set.add(well_key)
            path = f"{row_letter}/{col_name}"
            wells.append(
                WellPosition(path=path, row_index=row_index, column_index=col_index)
            )

    # Extract unique row labels from the first column of well names
    # all_well_names is 2D: [[A1, A2, ...], [B1, B2, ...], ...]
    # Extract just the row letter(s) from each first element
    rows = []
    for name in plate_plan.plate.all_well_names[:, 0]:
        row_label, _ = split_well_id(name)
        if row_label is None:
            raise ValueError(f"Invalid well name in plate definition: {name}")
        rows.append(row_label)
    num_cols = plate_plan.plate.columns
    # Create column labels (01, 02, 03, ...)
    columns = [str(i + 1).zfill(2) for i in range(num_cols)]

    field_count = plate_plan.num_points_per_well

    acquisitions = [
        PlateAcquisition(
            id=0,
            name="Acquisition 1",
            maximum_field_count=field_count,
        )
    ]

    return Plate(
        rows=rows,
        columns=columns,
        wells=wells,
        name=plate_plan.plate.name,
        field_count=field_count if field_count > 1 else None,
        acquisitions=acquisitions,
    )


def split_well_id(well: str) -> tuple[str | None, str | None]:
    match = re.match(r"([A-Z]+)(\d+)$", well, re.I)
    if match is None:
        return None, None
    if match.groups():
        row_letter, col_name = match.groups()
        if len(col_name) == 1:
            col_name = col_name.zfill(2)
        return row_letter, col_name
    return None, None
