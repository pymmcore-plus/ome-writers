from __future__ import annotations

from itertools import product
from typing import TYPE_CHECKING, cast, get_args

import numpy as np
import numpy.typing as npt

from ._dimensions import Dimension, DimensionLabel

if TYPE_CHECKING:
    from collections.abc import Iterator, Mapping, Sequence

    import useq

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
