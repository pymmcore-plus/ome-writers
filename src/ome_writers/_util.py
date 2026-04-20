from __future__ import annotations

from itertools import product
from typing import TYPE_CHECKING, cast

import numpy as np
import numpy.typing as npt

from ome_writers._schema import Dimension, dims_from_standard_axes

if TYPE_CHECKING:
    from collections.abc import Iterator, Mapping, Sequence


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


def spatial_role_indices(dims: Sequence[Dimension]) -> dict[str, int]:
    """Return a `{"x"|"y"|"z": dim_index}` map for spatial roles in `dims`.

    This function exists because ome-writers and NGFF don't mandate canonical axis
    names or a dimension orders, but some metadata formats/fields *DO*.  This is our
    best-effort attempt to resolve the spatial roles of dimensions based on their names
    and positions.

    Resolution rules (shared by the zarr and OME-XML backends):

    1. **Name match (case-insensitive)** on `"x"`, `"y"`, `"z"` wins first.
       This is the NGFF canonical labeling and what the vast majority of
       users will have via `StandardAxis` / `useq_to_acquisition_settings`.
    2. **Positional fallback for X and Y only**: if a name match didn't
       resolve X, the last dim of `dims` plays the X role when space-typed;
       if a name match didn't resolve Y, the second-to-last dim plays the Y
       role when space-typed. This handles non-canonical names like
       `"row"`/`"col"` without guessing.
    3. **Z has no positional fallback.** ome-writers doesn't mandate canonical
       axis names and doesn't mandate a ZYX dim order (`storage_order` can
       be `"acquisition"` or a custom list), so there's no unambiguous
       positional slot for Z the way there is for the frame-last X/Y pair.
       A future schema change — e.g. an explicit `spatial_role` on
       `Dimension`, or a structured position-coord map keyed by dim name —
       could lift this restriction. Until then, per-position Z metadata
       requires naming the Z axis `"z"` (case-insensitive).

    Missing roles are simply absent from the returned dict.
    """
    result: dict[str, int] = {}
    for i, d in enumerate(dims):
        lower = d.name.lower()
        if lower in ("x", "y", "z") and lower not in result:
            result[lower] = i
    n = len(dims)
    if "x" not in result and n >= 1 and dims[-1].type == "space":
        result["x"] = n - 1
    if "y" not in result and n >= 2 and dims[-2].type == "space":
        result["y"] = n - 2
    return result
