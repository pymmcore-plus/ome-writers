from __future__ import annotations

from itertools import product
from typing import TYPE_CHECKING, cast, get_args

import numpy as np
import numpy.typing as npt

from ._dimensions import Dimension, DimensionLabel

if TYPE_CHECKING:
    from collections.abc import Iterator, Mapping, Sequence

    import useq
    from yaozarrs.v05 import PlateDef

    from ._dimensions import UnitTuple


VALID_LABELS = get_args(DimensionLabel)
DEFAULT_UNITS: Mapping[DimensionLabel, UnitTuple | None] = {
    "t": (1.0, "s"),
    "z": (1.0, "um"),
    "y": (1.0, "um"),
    "x": (1.0, "um"),
    "c": None,
    "p": None,
}
OME_NGFF_ORDER = {"t": 0, "c": 1, "z": 2, "y": 3, "x": 4}


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

    If the sequence includes a grid_plan, the grid positions ("g" axis) will be
    combined with stage positions ("p" axis) into a single position dimension,
    since OME-NGFF does not have a separate grid concept.

    Supports both global and per-position grid plans:
    - **Global grid_plan**: Applied to all positions.
      Total positions = num_stage_positions * num_grid_points
    - **Per-position grid_plan**: Each position can have its own grid via nested
      sequences. Total positions = count of unique (p, g) combinations

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

    With a global grid_plan, positions are multiplied:

    ```python
    import useq
    from ome_writers import dims_from_useq

    seq = useq.MDASequence(
        stage_positions=[(0, 0), (100, 100)],  # 2 stage positions
        grid_plan=useq.GridRowsColumns(rows=2, columns=2),  # 4 grid points each
    )
    dims = dims_from_useq(seq, image_width=512, image_height=512)
    # dims will include a position dimension with size=8 (2 * 4)
    ```

    With per-position grid plans, positions are counted from unique (p, g) pairs:

    ```python
    import useq
    from ome_writers import dims_from_useq

    seq = useq.MDASequence(
        stage_positions=[
            useq.Position(
                x=0,
                y=0,
                sequence=useq.MDASequence(
                    grid_plan=useq.GridRowsColumns(rows=1, columns=2)
                ),
            ),
            (100, 100),  # No grid
        ],
    )
    dims = dims_from_useq(seq, image_width=512, image_height=512)
    # dims will include a position dimension with size=3: (0,0), (0,1), (1,0)
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
    p_size = 0  # Track position dimension size
    g_size = 0  # Track grid dimension size
    p_index: int | None = None  # Track where 'p' was in the original order

    for _i, (ax, size) in enumerate(seq.sizes.items()):
        if size:
            ax_str = str(ax)
            # Handle grid axis by combining with position axis
            if ax_str == "g":
                g_size = size
                # If we haven't seen 'p' yet, this becomes the position index
                if p_index is None:
                    p_index = len(dims)
                continue  # Don't add 'g' as a separate dimension
            elif ax_str == "p":
                p_size = size
                p_index = len(dims)  # Mark where 'p' should go
                continue  # Will add combined p dimension later

            # all of the useq axes are the same as the ones used here.
            dim_label = cast("DimensionLabel", ax_str)
            if dim_label not in _units:
                raise ValueError(f"Unsupported axis for OME: {ax}")
            dims.append(Dimension(label=dim_label, size=size, unit=_units[dim_label]))

    # Add combined position dimension at the correct location if either p or g exists
    # We need to count actual unique (p, g) combinations for accuracy, as useq can have:
    # 1. Global grid (g_size > 0, all positions use same grid)
    # 2. Per-position grids (g_size = 0, nested sequences define grids)
    # 3. Mixed (g_size > 0, but some positions override with nested grids)
    # Counting unique combinations handles all cases correctly.
    if p_size > 0 or g_size > 0:
        unique_positions: set[tuple[int, int]] = set()
        for event in seq:
            p = event.index.get("p", 0)
            g = event.index.get("g", 0)
            unique_positions.add((p, g))
        total_positions = len(unique_positions)

        if p_index is not None:
            dims.insert(
                p_index, Dimension(label="p", size=total_positions, unit=_units["p"])
            )

    return [
        *dims,
        Dimension(label="y", size=image_height, unit=_units["y"]),
        Dimension(label="x", size=image_width, unit=_units["x"]),
    ]


def position_names_from_useq(seq: useq.MDASequence) -> dict[int, str]:
    """Generate position names from useq sequence with grid information.

    When a useq sequence includes grid plans, the positions are flattened into
    a single position dimension. This function returns a mapping from the flattened
    position index to a descriptive name that includes both position and grid info.

    Parameters
    ----------
    seq : useq.MDASequence
        The useq sequence to extract position information from.

    Returns
    -------
    dict[int, str]
        Mapping from flattened position index to position name (e.g., "p0g1").
        If the sequence has no positions or grids, returns an empty dict.

    Examples
    --------
    ```python
    import useq
    from ome_writers import dims_from_useq, position_names_from_useq

    seq = useq.MDASequence(
        stage_positions=[
            useq.Position(x=0, y=0, name="Origin"),
            useq.Position(x=100, y=100, name="Site2"),
        ],
        grid_plan=useq.GridRowsColumns(rows=2, columns=2),
    )

    dims = dims_from_useq(seq, image_width=512, image_height=512)
    names = position_names_from_useq(seq)
    # names = {0: "p0g0", 1: "p0g1", 2: "p0g2", ..., 7: "p1g3"}

    # Use these names for logging, metadata, or documentation
    for idx, name in names.items():
        print(f"Position {idx}: {name}")
    ```
    """
    try:
        from useq import MDASequence
    except ImportError:
        raise ValueError("position_names_from_useq requires useq-schema") from None
    else:
        if not isinstance(seq, MDASequence):
            raise ValueError("seq must be a useq.MDASequence")

    # Collect unique (p, g) combinations in the order they appear
    seen: dict[tuple[int, int], int] = {}
    position_order: list[tuple[int, int]] = []

    for event in seq:
        p = event.index.get("p", 0)
        g = event.index.get("g", 0)
        pg_tuple = (p, g)

        if pg_tuple not in seen:
            idx = len(position_order)
            seen[pg_tuple] = idx
            position_order.append(pg_tuple)

    # Build name mapping
    names: dict[int, str] = {}
    for idx, (p, g) in enumerate(position_order):
        # Check if there are any grid points
        has_grid = any(pg[1] > 0 for pg in position_order)

        if has_grid:
            # Include grid info when grids are present
            names[idx] = f"p{p}g{g}"
        else:
            # Just position index when no grids
            names[idx] = f"p{p}"

    return names


class DimensionIndexIterator:
    """Iterator that yields frame indices in acquisition order.

    Takes dimensions in acquisition order and yields (position_key, index_tuple)
    where index_tuple contains non-spatial dimension indices in acquisition order
    (excluding position and spatial dimensions).

    Frames are yielded in acquisition sequence, with index_tuples preserving that
    order. For example, if acquisition order is ["t", "p", "z", "c"], then
    index_tuple will be (t_index, z_index, c_index) for each frame.

    When no position dimension exists, position_key is 0 for all frames.

    Notes
    -----
    - storage_order_dimensions must not include "y", "x", or position label (the
    `position_key` argument)
    - Since data is stored in acquisition order, storage_order_dimensions should
    match the non-spatial acquisition dimensions (excluding position)
    - index_tuple preserves acquisition order of the dimensions

    Parameters
    ----------
    acquisition_order_dimensions : Sequence[Dimension]
        Dimensions in acquisition order (slowest to fastest varying).
        May include position dimension.
    storage_order_dimensions : Sequence[DimensionLabel]
        Labels for non-spatial dims in acquisition order (excluding position),
        e.g. ["t", "z", "c"]. Must not include "y", "x", or position label.
        These determine which dimensions appear in the index_tuple output.
    position_key : DimensionLabel, optional
        Label for position dimension. Defaults to "p".

    Examples
    --------
    >>> dims = [
    ...     Dimension(label="t", size=2, unit=(1.0, "s"), chunk_size=1),
    ...     Dimension(label="p", size=2, unit=None, chunk_size=1),
    ...     Dimension(label="z", size=3, unit=(1.0, "um"), chunk_size=1),
    ...     Dimension(label="c", size=2, unit=None, chunk_size=1),
    ...     Dimension(label="y", size=32, unit=None, chunk_size=1),
    ...     Dimension(label="x", size=32, unit=None, chunk_size=1),
    ... ]
    >>> it = DimensionIndexIterator(dims, storage_order_dimensions=["t", "c", "z"])
    >>> list(it)[:3]
    [(0, (0, 0, 0)), (0, (0, 1, 0)), (0, (0, 0, 1))]
    """

    def __init__(
        self,
        acquisition_order_dimensions: Sequence[Dimension],
        storage_order_dimensions: Sequence[DimensionLabel],
        position_key: DimensionLabel = "p",
    ) -> None:
        # 1) Validate storage labels
        forbidden = {"y", "x", position_key}  # forbidden: spatial + position dims
        if forbidden & set(storage_order_dimensions):
            raise ValueError(
                "storage_order_dimensions should not include 'y', 'x', or "
                f"{position_key}."
            )

        # Build the desired output order: position first, then storage dims
        needed_labels = {position_key, *storage_order_dimensions}

        # Filter dimensions to those in output order, preserving acquisition order
        acq_order_dims = acquisition_order_dimensions
        self._iter_dims = [d for d in acq_order_dims if d.label in needed_labels]

        # Check if there are any spatial dimensions in the input
        self._has_spatial = any(d.label in ("y", "x") for d in acq_order_dims)

        # Compute shape tuple for iteration in acquisition order
        self._shape = tuple(d.size for d in self._iter_dims)

        # Precompute label→index mapping
        acq_labels = [d.label for d in self._iter_dims]

        # Get position index from acquisition dims if present
        p_key = position_key
        self._pos_idx = acq_labels.index(p_key) if p_key in acq_labels else None

        # Get storage indices from acquisition dims
        sod = storage_order_dimensions
        self._stor_idx = [acq_labels.index(lbl) for lbl in sod if lbl in acq_labels]

    def __iter__(self) -> Iterator[tuple[int, tuple]]:
        """Yield indices in acquisition order, formatted in output order."""
        if not self._shape:
            # Special case: no non-spatial dimensions
            # If there are spatial dimensions (2D-only), yield one frame
            # If no dimensions at all, don't yield anything
            if self._has_spatial:
                yield 0, ()
            return

        # Iterate over all acquisition indices
        for acq_idx in np.ndindex(*self._shape):
            # Extract position index
            pos = int(acq_idx[self._pos_idx]) if self._pos_idx is not None else 0
            # Build storage-ordered tuple from acquisition indices
            out = tuple(int(acq_idx[i]) for i in self._stor_idx)
            yield pos, out

    def __len__(self) -> int:
        return int(np.prod(self._shape)) if self._shape else 0


def plate_from_useq_to_yaozarrs(
    plan: useq.WellPlatePlan,
) -> PlateDef:
    """Convert a useq.WellPlatePlan to a yaozarrs.v05.PlateDef.

    Parameters
    ----------
    plan : useq.WellPlatePlan
        The useq well plate plan to convert.

    Returns
    -------
    yaozarrs.v05.PlateDef
        Plate definition compatible with yaozarrs-based OME-Zarr streams.

    Examples
    --------
    ```python
    import useq
    from ome_writers import plate_from_useq_to_yaozarrs

    # Create a useq plate plan
    plate_plan = useq.WellPlatePlan(
        plate=96,
        a1_center_xy=(500, 200),
        selected_wells=[(0, 0), (0, 1), (1, 0), (1, 1)],
        well_points_plan=useq.GridRowsColumns(rows=2, columns=2),
    )

    # Convert to yaozarrs PlateDef
    plate_def = plate_from_useq_to_yaozarrs(plate_plan)

    # Use with yaozarrs stream
    from ome_writers import TensorStoreZarrStream

    stream = TensorStoreZarrStream()
    stream.create(..., plate=plate_def)
    ```
    """
    try:
        from useq import WellPlatePlan
        from yaozarrs import v05
    except ImportError as e:
        msg = "plate_from_useq_to_yaozarrs requires useq-schema and yaozarrs"
        raise ImportError(msg) from e

    if not isinstance(plan, WellPlatePlan):
        raise TypeError("plan must be a useq.WellPlatePlan")

    # Extract unique rows and columns from selected wells
    well_names = plan.selected_well_names
    rows_set: set[str] = set()
    cols_set: set[str] = set()

    for well_name in well_names:
        # Well names are like "A1", "B2", "AA10", etc.
        # Split at the letter/digit boundary
        i = 0
        while i < len(well_name) and well_name[i].isalpha():
            i += 1
        row_name = well_name[:i]  # Letter part
        col_name = well_name[i:]  # Number part
        rows_set.add(row_name)
        cols_set.add(col_name)

    # Sort rows alphabetically, columns numerically
    rows = sorted(rows_set)
    cols = sorted(cols_set, key=int)

    # Create Row and Column objects
    row_objects = [v05.Row(name=r) for r in rows]
    col_objects = [v05.Column(name=c) for c in cols]

    # Create PlateWell objects (deduplicate to avoid pydantic validation errors)
    wells_dict: dict[str, v05.PlateWell] = {}
    for well_name in well_names:
        # Split at the letter/digit boundary
        i = 0
        while i < len(well_name) and well_name[i].isalpha():
            i += 1
        row_name = well_name[:i]
        col_name = well_name[i:]
        row_idx = rows.index(row_name)
        col_idx = cols.index(col_name)

        well_key = f"{row_name}/{col_name}"
        if well_key not in wells_dict:
            wells_dict[well_key] = v05.PlateWell(
                path=well_key,
                rowIndex=row_idx,
                columnIndex=col_idx,
            )

    wells = list(wells_dict.values())

    # Determine field count from well_points_plan
    field_count = plan.num_points_per_well

    return v05.PlateDef(
        name=plan.plate.name or "Plate",
        rows=row_objects,
        columns=col_objects,
        wells=wells,
        field_count=field_count if field_count > 1 else None,
    )
