# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "acquire-zarr",
#     "numpy",
#     "yaozarrs[io]",
# ]
# ///

"""Basic example of using ome_writers."""

import json
from pathlib import Path

import numpy as np
import yaozarrs
from acquire_zarr import (
    ArraySettings,
    Dimension,
    DimensionType,
    StreamSettings,
    ZarrStream,
    ZarrVersion,
)


def write_zarr(output: Path) -> None:
    # NOTE!!!
    # TZCYX order
    dtype = np.dtype("uint16")
    data = np.random.randint(0, 65536, size=(10, 5, 2, 512, 512), dtype=dtype)
    nt, nz, nc, ny, nx = data.shape

    dimensions = [
        Dimension(
            name="t",
            kind=DimensionType.TIME,
            array_size_px=nt,
            chunk_size_px=1,
            shard_size_chunks=1,
        ),
        Dimension(
            name="z",
            kind=DimensionType.SPACE,
            array_size_px=nz,
            chunk_size_px=1,
            shard_size_chunks=1,
        ),
        Dimension(
            name="c",
            kind=DimensionType.CHANNEL,
            array_size_px=nc,
            chunk_size_px=1,
            shard_size_chunks=1,
        ),
        Dimension(
            name="y",
            kind=DimensionType.SPACE,
            array_size_px=ny,
            chunk_size_px=64,
            shard_size_chunks=1,
        ),
        Dimension(
            name="x",
            kind=DimensionType.SPACE,
            array_size_px=nx,
            chunk_size_px=64,
            shard_size_chunks=1,
        ),
    ]

    settings = StreamSettings(
        arrays=[
            ArraySettings(
                dimensions=dimensions,
                data_type=dtype,
            )
        ],
        overwrite=True,
        store_path=str(output),
        version=ZarrVersion.V3,
    )
    stream = ZarrStream(settings)

    for t, z, c in np.ndindex(nt, nz, nc):
        stream.append(data[t, z, c])
    stream.close()


def reorder_dims(output: Path) -> None:
    def swap_z_and_c(lst: list) -> None:
        lst[1], lst[2] = lst[2], lst[1]

    # This function is NOT idempotent!
    ome_md_file = output / "zarr.json"
    with open(ome_md_file) as fh:
        ome_md = json.load(fh)

    swap_z_and_c(ome_md["attributes"]["ome"]["multiscales"][0]["axes"])
    swap_z_and_c(
        ome_md["attributes"]["ome"]["multiscales"][0]["datasets"][0][
            "coordinateTransformations"
        ][0]["scale"]
    )

    with open(ome_md_file, "w") as fh:
        json.dump(ome_md, fh)

    array_md_file = output / "0" / "zarr.json"
    with open(array_md_file) as fh:
        array_md = json.load(fh)

    # swap dimensions
    swap_z_and_c(array_md["chunk_grid"]["configuration"]["chunk_shape"])
    swap_z_and_c(array_md["codecs"][0]["configuration"]["chunk_shape"])
    swap_z_and_c(array_md["dimension_names"])
    swap_z_and_c(array_md["shape"])

    # add transpose codec
    array_md["codecs"][0]["configuration"]["codecs"].insert(
        0, {"name": "transpose", "configuration": {"order": [0, 2, 1, 3, 4]}}
    )

    with open(array_md_file, "w") as fh:
        json.dump(array_md, fh)


def validate_zarr(output: Path) -> None:
    try:
        yaozarrs.validate_zarr_store(str(output))
    except Exception as e:
        print("Zarr store validation FAILED:", e)
    else:
        print("Zarr store validation PASSED")


if __name__ == "__main__":
    op = Path("example_az_transposed.zarr").expanduser()
    write_zarr(op)
    reorder_dims(op)
    validate_zarr(op)
