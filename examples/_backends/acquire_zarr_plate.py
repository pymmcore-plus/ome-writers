"""Basic example of using ome_writers."""

from __future__ import annotations

from pathlib import Path

import acquire_zarr as az
import numpy as np

AXIS_ORDER = ["t", "c", "z", "y", "x"]
SIZES = {"t": 10, "c": 2, "z": 5, "y": 512, "x": 512}
CHUNK_SHAPES = {"t": 1, "c": 1, "z": 1, "y": 64, "x": 64}
PLATE_FOVS = [("A", "2", "0"), ("B", "1", "0")]  # [(row_name, col_name, fov_path), ...]
DTYPE = np.uint16

# --------------------------------------------------
assert set(AXIS_ORDER[-2:]) == {"y", "x"}, "Last two axes must be YX"
SHAPE = tuple(SIZES[dim] for dim in AXIS_ORDER)
_extra = "".join(f"_{dim}{SIZES[dim]}" for dim in AXIS_ORDER[:-2])
DEST = Path(f"example_az_plate{_extra}.zarr")
_kind_map = {"t": az.DimensionType.TIME, "c": az.DimensionType.CHANNEL}

array_dimensions = [
    az.Dimension(
        name=dim,
        kind=_kind_map.get(dim, az.DimensionType.SPACE),
        array_size_px=SIZES[dim],
        chunk_size_px=CHUNK_SHAPES[dim],
        shard_size_chunks=1,
    )
    for dim in AXIS_ORDER
]

acq_0 = az.Acquisition(id=0)
wells = [
    az.Well(
        row_name=row_name,
        column_name=column_name,
        images=[
            az.FieldOfView(
                path=fov_path,
                array_settings=az.ArraySettings(
                    output_key=fov_path,
                    dimensions=array_dimensions,
                    data_type=DTYPE,
                ),
                acquisition_id=acq_0.id,
            )
        ],
    )
    for row_name, column_name, fov_path in PLATE_FOVS
]

plate = az.Plate(
    path="",
    name="My HCS Experiment",
    row_names=["A", "B", "C", "D"],
    column_names=["1", "2", "3", "4", "5", "6", "7", "8"],
    wells=wells,
    acquisitions=[acq_0],
)
settings = az.StreamSettings(
    hcs_plates=[plate],
    overwrite=True,
    store_path=str(DEST),
    version=az.ZarrVersion.V3,
)
stream = az.ZarrStream(settings)

i = 0
for well in wells:
    for fov in well.images:
        for _ in np.ndindex(SHAPE[:-2]):
            # Key must include full path: row/column/fov_name
            key = f"{well.row_name}/{well.column_name}/{fov.path}"
            frame = np.full(shape=SHAPE[-2:], fill_value=i, dtype=DTYPE)
            stream.append(frame, key=key)
stream.close()

print("Data written successfully to", DEST)
