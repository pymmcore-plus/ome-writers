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


def reorder_to_ome_ngff(dims_order: list[Dimension]) -> list[Dimension]:
    """Reorder dimensions to OME-NGFF TCZYX order."""
    dims_order.sort(key=lambda d: OME_NGFF_ORDER.get(d.label, 5))
    return dims_order


class DimensionIndexIterator:
    """Yield indices for frames in acquisition order, formatted for storage.

    The iterator accepts dimensions in acquisition order (slowest- to
    fastest-varying) and a sequence of labels describing the desired storage
    order for the non-spatial axes (e.g. `["t", "c", "z"]`). It yields
    tuples of the form (position_key, index_tuple) where `position_key` is
    an integer identifying the position (`p`). When no position dimension is
    present in the provided `acquisition_order_dimensions`, the iterator will
    return `position_key` == 0 for every frame (single-position case).

    Notes
    -----
    - `storage_order_dimensions` must not include the spatial labels
      `"y"` or `"x"` nor the position label (default "p").
    - The iterator returns indices in acquisition order (so frames are yielded
      in the sequence they would be acquired), but each yielded index tuple
      is ordered according to the provided storage order labels.

    Parameters
    ----------
    acquisition_order_dimensions : Sequence[Dimension]
        Dimensions in acquisition order (slowest to fastest varying). This
        sequence may include a position dimension labelled by `position_key`.
    storage_order_dimensions : Sequence[DimensionLabel]
        Labels for the non-spatial dimensions in the desired storage order
        (e.g. `["t", "c", "z"]`). Should not include `"y"`, `"x"`,
        or the position label.
    position_key : DimensionLabel, optional
        Label used for the position dimension. Defaults to `"p"`.

    Example
    -------
    >>> dims = [
    ...     Dimension(label="t", size=2, unit=(1.0, "s"), chunk_size=1),
    ...     Dimension(label="p", size=2, unit=None, chunk_size=1),
    ...     Dimension(label="z", size=3, unit=(1.0, "um"), chunk_size=1),
    ...     Dimension(label="c", size=2, unit=None, chunk_size=1),
    ...     Dimension(label="y", size=32, unit=None, chunk_size=1),
    ...     Dimension(label="x", size=32, unit=None, chunk_size=1),
    ... ]
    >>> it = DimensionIndexIterator(dims, storage_order_dimensions=["t", "c", "z"])
    >>> for pos, idx in it:
    ...     print(pos, idx)
    0 (0, 0, 0)  # which means p=0, (t=0, c=0, z=0)
    0 (0, 1, 0)  # which means p=0, (t=0, c=1, z=0)
    ...
    If the "c" and "z" dimensions were swapped in storage order:
    >>> it = DimensionIndexIterator(dims, storage_order_dimensions=["t", "z", "c"])
    >>> for pos, idx in it:
    ...     print(pos, idx)
    0 (0, 0, 0)  # which means p=0, (t=0, z=0, c=0)
    0 (0, 0, 1)  # which means p=0, (t=0, z=0, c=1)
    ...
    """

    def __init__(
        self,
        acquisition_order_dimensions: Sequence[Dimension],
        storage_order_dimensions: Sequence[DimensionLabel],
        position_key: DimensionLabel = "p",
    ) -> None:
        # Validate that storage_order_dimensions does not include forbidden labels
        forbidden_labels = {"y", "x", position_key}
        if forbidden_labels & set(storage_order_dimensions):
            raise ValueError(
                "storage_order_dimensions should not include 'y', 'x', or "
                f"{position_key}."
            )

        # Store acquisition order labels and position key for __iter__ method
        self._acq_dims_labels = [d.label for d in acquisition_order_dimensions]
        self._position_key = position_key

        # Build the desired output order: position first, then storage dims
        output_labels_order = [position_key, *list(storage_order_dimensions)]

        # Filter dimensions to those in output order, preserving acquisition order
        acq_order_dims = acquisition_order_dimensions
        self.iter_dims = [d for d in acq_order_dims if d.label in output_labels_order]

        # Compute shape tuple for iteration in acquisition order
        self.shape = tuple(d.size for d in self.iter_dims)

        # Get labels of filtered dimensions in acquisition order
        acq_labels = [d.label for d in self.iter_dims]
        # Determine which output labels are present in acquisition dims
        field_names = [label for label in output_labels_order if label in acq_labels]
        # Map from output order indices to acquisition order indices
        self._output_to_acq = [acq_labels.index(f) for f in field_names]

    def __iter__(self) -> Iterator[tuple[int, tuple]]:
        """Yield indices in acquisition order, formatted in output order."""
        # If no dimensions to iterate over, return empty iterator
        if not self.shape:
            return

        # Determine position key depending if it was included in acquisition dims
        has_pos_key = False if self._position_key not in self._acq_dims_labels else True
        # Iterate over all possible flat indices
        for i in range(int(np.prod(self.shape))):
            # Unravel flat index to multi-dim indices (acquisition order)
            acq_indices = np.unravel_index(i, self.shape)
            # Map acquisition indices to output indices
            all_output_indices = tuple(int(acq_indices[j]) for j in self._output_to_acq)
            # If position dimension is present, extract position key from indices
            if has_pos_key:
                pos_key, *output_indices = all_output_indices
                # Yield position key and index tuple in storage order
                yield pos_key, tuple(output_indices)
            # Otherwise, just use the output indices as they are
            else:
                # Yield position key and index tuple in storage order
                yield 0, tuple(all_output_indices)

    def __len__(self) -> int:
        """Return total number of frames."""
        # Return product of dimension sizes, or 0 if no dimensions
        return int(np.prod(self.shape)) if self.shape else 0
