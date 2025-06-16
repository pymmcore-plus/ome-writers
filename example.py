"""Basic example of using ome_writers."""

from __future__ import annotations

import shutil
from itertools import product
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from ome_writers import DimensionInfo, create_stream

if TYPE_CHECKING:
    from collections.abc import Iterator, Mapping


UNITS: dict[str, tuple[float, str]] = {
    "t": (1.0, "s"),
    "z": (1.0, "um"),
    "y": (1.0, "um"),
    "x": (1.0, "um"),
}


def create_data_and_dims(
    sizes: Mapping[str, int],
    *,
    dtype: np.typing.DTypeLike = np.uint16,
    chunk_sizes: Mapping[str, int] | None = None,
) -> tuple[Iterator[np.ndarray], list[DimensionInfo], np.dtype]:
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
    if not {"y", "x"} <= sizes.keys():
        raise ValueError("sizes must include both 'y' and 'x'")

    _chunk_sizes = dict(chunk_sizes or {})
    _chunk_sizes.setdefault("y", sizes["y"])
    _chunk_sizes.setdefault("x", sizes["x"])

    ordered_labels: list[str] = [z for z in sizes if z not in "yx"]
    ordered_labels += ["y", "x"]
    dims = [
        DimensionInfo(
            label=lbl,
            size=sizes[lbl],
            unit=UNITS.get(lbl, None),
            chunk_size=_chunk_sizes.get(lbl, 1),
        )
        for lbl in ordered_labels
    ]

    shape = [d.size for d in dims]
    dtype = np.dtype(dtype)
    if not np.issubdtype(dtype, np.integer):
        raise ValueError(f"Unsupported dtype: {dtype}")

    rng = np.random.default_rng()
    data = rng.integers(0, np.iinfo(dtype).max, size=shape, dtype=dtype)

    def _build_plane_generator() -> Iterator[np.ndarray]:
        """Yield 2-D planes in y-x order."""
        if not non_spatial:  # only y, x present
            yield data
        else:
            for idx in product(*(range(n) for n in non_spatial)):
                yield data[idx]  # successive indexing drops axes

    non_spatial = shape[:-2]  # everything except y, x
    return _build_plane_generator(), dims, dtype


def write_with_ome_writers(
    output: str,
    dtype: np.dtype,
    dims: list[DimensionInfo],
    backend: str,
    plane_iter: Iterator[np.ndarray],
) -> None:
    """Write data to a Zarr store using the specified ome-writers backend."""
    stream = create_stream(output, dtype, dims, backend=backend)

    for plane in plane_iter:
        stream.append(plane)
    stream.flush()


def write_directly_with_aq_zarr(
    output: str,
    dtype: np.dtype,
    dims: list[DimensionInfo],
    plane_iter: Iterator[np.ndarray],
) -> None:
    """Just uses acquire-zarr directly to write data."""
    from acquire_zarr import (
        DataType,
        Dimension,
        DimensionType,
        StreamSettings,
        ZarrStream,
        ZarrVersion,
    )

    def _dim_toaqz_dim(
        dim: DimensionInfo,
        shard_size_chunks: int = 1,
    ) -> Dimension:
        return Dimension(
            name=dim.label,
            type=getattr(DimensionType, dim.ome_dim_type.upper()),
            array_size_px=dim.size,
            chunk_size_px=(dim.chunk_size if dim.chunk_size is not None else dim.size),
            shard_size_chunks=shard_size_chunks,
        )

    settings = StreamSettings(
        store_path=output,
        data_type=getattr(DataType, np.dtype(dtype).name.upper()),
        version=ZarrVersion.V3,
        overwrite=True,
        output_key="",
    )
    settings.dimensions.extend([_dim_toaqz_dim(dim) for dim in dims])
    stream = ZarrStream(settings)
    for plane in plane_iter:
        stream.append(plane)
    del stream  # flush


if __name__ == "__main__":
    sizes = {"t": 10, "z": 10, "c": 1, "y": 256, "x": 256}
    chunks_sizes = {"y": 64, "x": 64}
    output = Path("~/Desktop/some_path_ts.zarr").expanduser()
    # backend = "acquire-zarr"
    backend = "tensorstore"
    shutil.rmtree(output, ignore_errors=True)  # remove existing output

    plane_iter, dims, dtype = create_data_and_dims(
        sizes=sizes, chunk_sizes=chunks_sizes
    )
    write_with_ome_writers(str(output), dtype, dims, backend, plane_iter)
    # write_directly_with_aq_zarr(str(output), dtype, dims, plane_iter)
    print("Data written successfully to", output)
