"""Full integration testing, from schema declaration to on-disk file verification."""

from __future__ import annotations

import importlib.util
from typing import TYPE_CHECKING, cast

import numpy as np
import pytest
import yaozarrs
from yaozarrs import v05

from ome_writers import (
    AcquisitionSettings,
    ArraySettings,
    Dimension,
    Plate,
    Position,
    PositionDimension,
    create_stream,
)

if TYPE_CHECKING:
    from pathlib import Path

AVAILABLE_BACKENDS = []
if importlib.util.find_spec("tensorstore") is not None:
    AVAILABLE_BACKENDS.append("tensorstore")
if importlib.util.find_spec("zarr") is not None:
    AVAILABLE_BACKENDS.append("zarr")
if importlib.util.find_spec("acquire_zarr") is not None:
    AVAILABLE_BACKENDS.append("acquire-zarr")

# NOTES:
# - All root_paths will be replaced with temporary directories during testing.
D = Dimension  # alias, for brevity
CASES = [
    AcquisitionSettings(
        root_path="tmp",
        array_settings=ArraySettings(
            dimensions=[
                D(name="t", count=2, type="time"),
                D(name="c", count=3, type="channel"),
                D(name="z", count=4, type="space", scale=5),
                D(name="y", count=128, chunk_size=64, type="space", scale=0.1),
                D(name="x", count=128, chunk_size=64, type="space", scale=0.1),
            ],
            dtype="uint16",
        ),
    ),
    # transpose z and c ...
    # because we always write out valid OME-Zarr by default (i.e. TCZYX as of v0.5)
    # this exercises non-standard dimension orders
    # validation is ensured by yaozarrs.validate_zarr_store()
    AcquisitionSettings(
        root_path="tmp",
        array_settings=ArraySettings(
            dimensions=[
                D(name="t", count=2, type="time"),
                D(name="z", count=4, type="space", scale=5),
                D(name="c", count=3, type="channel"),
                D(name="y", count=128, chunk_size=64, type="space", scale=0.1),
                D(name="x", count=128, chunk_size=64, type="space", scale=0.1),
            ],
            dtype="uint16",
        ),
    ),
    # Multi-position case
    AcquisitionSettings(
        root_path="tmp",
        array_settings=ArraySettings(
            dimensions=[
                PositionDimension(positions=["Pos0", "Pos1"]),
                D(name="z", count=3, type="space"),
                D(name="y", count=128, chunk_size=64, type="space", scale=0.1),
                D(name="x", count=128, chunk_size=64, type="space", scale=0.1),
            ],
            dtype="uint16",
        ),
    ),
    # position interleaved with other dimensions
    AcquisitionSettings(
        root_path="tmp",
        array_settings=ArraySettings(
            dimensions=[
                D(name="t", count=3, type="time"),
                PositionDimension(positions=["Pos0", "Pos1"]),
                D(name="y", count=128, chunk_size=64, type="space", scale=0.1),
                D(name="x", count=128, chunk_size=64, type="space", scale=0.1),
            ],
            dtype="uint16",
        ),
    ),
    # Plate case
    AcquisitionSettings(
        root_path="tmp",
        array_settings=ArraySettings(
            dimensions=[
                D(name="t", count=10, chunk_size=1, type="time"),
                PositionDimension(
                    positions=[
                        Position(name="fov0", row="A", column="1"),
                        Position(name="fov0", row="A", column="2"),
                        Position(name="fov0", row="C", column="4"),
                        Position(name="fov1", row="C", column="4"),
                    ]
                ),
                D(name="c", count=2, chunk_size=1, type="channel"),
                D(name="z", count=5, chunk_size=1, type="space"),
                D(name="y", count=256, chunk_size=64, type="space"),
                D(name="x", count=256, chunk_size=64, type="space"),
            ],
            dtype="uint16",
        ),
        plate=Plate(
            name="Example Plate",
            row_names=["A", "B", "C", "D"],
            column_names=["1", "2", "3", "4", "5", "6", "7", "8"],
        ),
    ),
]


def _name_case(case: AcquisitionSettings) -> str:
    dims = case.array_settings.dimensions
    dim_names = "_".join(f"{d.name}{d.count}" for d in dims)
    plate_str = "plate-" if case.plate is not None else ""
    return f"{plate_str}{dim_names}-{case.array_settings.dtype}"


@pytest.mark.parametrize("case", CASES, ids=_name_case)
@pytest.mark.parametrize("backend", AVAILABLE_BACKENDS)
def test_cases_as_zarr(case: AcquisitionSettings, backend: str, tmp_path: Path) -> None:
    case.root_path = root = tmp_path / "output.zarr"
    case.backend = backend
    dims = case.array_settings.dimensions
    dtype = case.array_settings.dtype
    # currently, we enforce that the last 2 dimensions are the frame dimensions
    frame_shape = cast("tuple[int, ...]", tuple(d.count for d in dims[-2:]))

    # -------------- Write out all frames --------------

    with create_stream(case) as stream:
        router = stream._router
        num_positions = len(router.positions)
        stored_array_dims = router.array_storage_dimensions
        for f in range(router.num_frames):
            frame_data = np.full(frame_shape, f, dtype=dtype)
            stream.append(frame_data)

    # -------------- Validate the result --------------
    group = yaozarrs.validate_zarr_store(root)
    ome_meta = group.ome_metadata()

    # Assert group type (single image, multi-position, plate)
    # and collect image paths for further validation
    image_paths: list[Path] = []
    # Plate
    if case.plate is not None:
        assert isinstance(ome_meta, v05.Plate)
        image_paths = [root / p.row / p.column / p.name for p in router.positions]
    # Single image
    elif num_positions == 1:
        assert isinstance(ome_meta, v05.Image)
        image_paths = [root]
    # Multi-position
    else:
        assert isinstance(ome_meta, v05.Bf2Raw)
        ome_group = yaozarrs.validate_ome_uri(root / "OME")
        assert isinstance(ome_group.attributes.ome, v05.Series)
        image_paths = [root / pos.name for pos in router.positions]

    assert len(image_paths) == num_positions
    _validate_images(stored_array_dims, image_paths, dtype)


def _validate_images(
    storage_dims: list[D], image_paths: list[Path], dtype: str
) -> None:
    """Validate all images (multiscales groups) generated by a test."""
    for image_path in image_paths:
        group = yaozarrs.open_group(image_path)
        image = group.ome_metadata()
        assert isinstance(image, v05.Image), (
            f"Expected Image group at {image_path}, got {type(image)}"
        )

        # NOTE:
        # we're validating on disk data with tensorstore rather than zarr-python
        # due to zarr-python's dropped support for python versions that are still
        # before EOL (i.e. SPEC-0)
        if importlib.util.find_spec("tensorstore") is not None:
            _validate_array_tensorstore(group, storage_dims, dtype)
        else:
            _validate_array_zarr(group, storage_dims, dtype)


def _validate_array_tensorstore(
    group: yaozarrs.ZarrGroup, storage_dims: list[D], dtype: str
) -> None:
    """Validate an array stored on disk using tensorstore."""
    import tensorstore

    array0 = group["0"].to_tensorstore()  # type: ignore[possibly-unbound-attribute]
    assert isinstance(array0, tensorstore.TensorStore)

    # check on disk shape, dtype, dimension order and labels
    assert array0.shape == tuple(d.count for d in storage_dims)
    assert array0.dtype == np.dtype(dtype)
    expected_dim_order = [d.name for d in storage_dims]
    assert [d.label for d in array0.domain] == expected_dim_order  # type: ignore

    # validate chunking
    expected_chunk_shape = tuple(d.chunk_size or 1 for d in storage_dims)
    actual_chunk_shape = array0.chunk_layout.read_chunk.shape
    assert actual_chunk_shape == expected_chunk_shape, (
        f"expected {expected_chunk_shape}, got {actual_chunk_shape}"
    )

    # validate sharding
    # TODO


def _validate_array_zarr(
    group: yaozarrs.ZarrGroup, storage_dims: list[D], dtype: str
) -> None:
    """Validate an array stored on disk using zarr-python."""
    import zarr

    array0 = group["0"].to_zarr_python()  # type: ignore[possibly-unbound-attribute]
    assert isinstance(array0, zarr.Array)

    # check on disk shape, dtype
    assert array0.shape == tuple(d.count for d in storage_dims)
    assert str(array0.dtype) == dtype

    # validate chunking
    expected_chunk_shape = tuple(d.chunk_size or 1 for d in storage_dims)
    assert array0.chunks == expected_chunk_shape, (
        f"expected {expected_chunk_shape}, got {array0.chunks}"
    )

    # validate sharding
    # TODO
