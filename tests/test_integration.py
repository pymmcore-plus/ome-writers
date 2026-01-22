"""Full integration testing, from schema declaration to on-disk file verification."""

from __future__ import annotations

import contextlib
import importlib.util
import re
import time
from pathlib import Path
from typing import TypeAlias
from unittest.mock import Mock

import numpy as np
import pytest
import yaozarrs
from yaozarrs import v05

from ome_writers import (
    AcquisitionSettings,
    Dimension,
    Plate,
    Position,
    PositionDimension,
    _memory,
    create_stream,
)
from tests._frame_encoder import validate_encoded_frame_values, write_encoded_data

# NOTES:
# - All root_paths will be replaced with temporary directories during testing.
D = Dimension  # alias, for brevity
CASES = [
    AcquisitionSettings(
        root_path="tmp",
        dimensions=[
            D(name="y", count=64, chunk_size=64, type="space", scale=0.1),
            D(name="x", count=64, chunk_size=64, type="space", scale=0.1),
        ],
        dtype="uint16",
    ),
    AcquisitionSettings(
        root_path="tmp",
        dimensions=[
            D(name="c", count=2, type="channel"),
            D(name="y", count=64, chunk_size=64, type="space", scale=0.1),
            D(name="x", count=64, chunk_size=64, type="space", scale=0.1),
        ],
        dtype="uint8",
    ),
    AcquisitionSettings(
        root_path="tmp",
        dimensions=[
            D(name="t", count=2, type="time"),
            D(name="c", count=3, type="channel"),
            D(name="z", count=4, type="space", scale=5),
            D(name="y", count=128, chunk_size=64, type="space", scale=0.1),
            D(name="x", count=128, chunk_size=64, type="space", scale=0.1),
        ],
        dtype="uint16",
    ),
    # transpose z and c ...
    # because we always write out valid OME-Zarr by default (i.e. TCZYX as of v0.5)
    # this exercises non-standard dimension orders
    # validation is ensured by yaozarrs.validate_zarr_store()
    AcquisitionSettings(
        root_path="tmp",
        dimensions=[
            D(name="t", count=2, type="time"),
            D(name="z", count=4, type="space", scale=5),
            D(name="c", count=3, type="channel"),
            D(name="y", count=128, chunk_size=64, type="space", scale=0.1),
            D(name="x", count=128, chunk_size=64, type="space", scale=0.1),
        ],
        dtype="uint16",
    ),
    # Multi-position case
    AcquisitionSettings(
        root_path="tmp",
        dimensions=[
            PositionDimension(positions=["Pos0", "Pos1"]),
            D(name="z", count=3, type="space"),
            D(name="y", count=128, chunk_size=64, type="space", scale=0.1),
            D(name="x", count=128, chunk_size=64, type="space", scale=0.1),
        ],
        dtype="uint16",
    ),
    # position interleaved with other dimensions
    AcquisitionSettings(
        root_path="tmp",
        dimensions=[
            D(name="t", count=3, type="time"),
            PositionDimension(positions=["Pos0", "Pos1"]),
            D(name="y", count=128, chunk_size=64, type="space", scale=0.1),
            D(name="x", count=128, chunk_size=64, type="space", scale=0.1),
        ],
        dtype="uint16",
    ),
    # Plate case
    AcquisitionSettings(
        root_path="tmp",
        dimensions=[
            D(name="t", count=3, chunk_size=1, type="time"),
            PositionDimension(
                positions=[
                    Position(name="fov0", row="A", column="1"),
                    Position(name="fov0", row="C", column="4"),
                    Position(name="fov1", row="C", column="4"),
                ]
            ),
            D(name="c", count=2, chunk_size=1, type="channel"),
            D(name="z", count=4, chunk_size=1, type="space"),
            D(name="y", count=128, chunk_size=64, type="space"),
            D(name="x", count=128, chunk_size=64, type="space"),
        ],
        dtype="uint16",
        plate=Plate(
            name="Example Plate",
            row_names=["A", "B", "C", "D"],
            column_names=["1", "2", "3", "4", "5", "6", "7", "8"],
        ),
    ),
    # Unbounded first dimension (mimics runtime-determined acquisition length)
    AcquisitionSettings(
        root_path="tmp",
        dimensions=[
            D(name="t", count=None, chunk_size=1, type="time"),  # unbounded
            D(name="c", count=2, type="channel"),
            D(name="z", count=3, type="space"),
            D(name="y", count=128, chunk_size=64, type="space", scale=0.1),
            D(name="x", count=128, chunk_size=64, type="space", scale=0.1),
        ],
        dtype="uint16",
    ),
    # Unbounded with chunk buffering (tests resize with buffering enabled)
    AcquisitionSettings(
        root_path="tmp",
        dimensions=[
            D(name="t", count=None, chunk_size=1, type="time"),  # unbounded
            D(name="z", count=8, chunk_size=4, type="space"),  # chunked
            D(name="y", count=64, chunk_size=64, type="space", scale=0.1),
            D(name="x", count=64, chunk_size=64, type="space", scale=0.1),
        ],
        dtype="uint16",
    ),
    # Chunk buffering: 3D with chunk_size=4
    AcquisitionSettings(
        root_path="tmp",
        dimensions=[
            D(name="z", count=16, chunk_size=4, type="space"),
            D(name="y", count=64, chunk_size=64, type="space", scale=0.1),
            D(name="x", count=64, chunk_size=64, type="space", scale=0.1),
        ],
        dtype="uint16",
    ),
    # Chunk buffering with transposition (storage_order != acquisition)
    AcquisitionSettings(
        root_path="tmp",
        dimensions=[
            D(name="t", count=2, type="time"),
            D(name="z", count=8, chunk_size=4, type="space"),
            D(name="c", count=4, chunk_size=2, type="channel"),
            D(name="y", count=64, chunk_size=64, type="space", scale=0.1),
            D(name="x", count=64, chunk_size=64, type="space", scale=0.1),
        ],
        dtype="uint16",
    ),
    # Chunk buffering with partial chunks at finalize
    # (z=17 with non-divisible chunk_size=4)
    AcquisitionSettings(
        root_path="tmp",
        dimensions=[
            D(name="z", count=17, chunk_size=4, type="space"),
            D(name="y", count=64, chunk_size=64, type="space", scale=0.1),
            D(name="x", count=64, chunk_size=64, type="space", scale=0.1),
        ],
        dtype="uint16",
    ),
    # Chunk buffering with multiple positions
    AcquisitionSettings(
        root_path="tmp",
        dimensions=[
            PositionDimension(positions=["Pos0", "Pos1"]),
            D(name="z", count=8, chunk_size=4, type="space"),
            D(name="y", count=64, chunk_size=64, type="space", scale=0.1),
            D(name="x", count=64, chunk_size=64, type="space", scale=0.1),
        ],
        dtype="uint16",
    ),
]


def _name_case(case: AcquisitionSettings) -> str:
    dims = case.dimensions
    dim_names = "_".join(f"{d.name}{d.count}" for d in dims)
    plate_str = "plate-" if case.plate is not None else ""
    return f"{plate_str}{dim_names}-{case.dtype}"


UNBOUNDED_FRAME_COUNT = 2  # number of frames to write for unbounded dimensions

StorageIdxToFrame: TypeAlias = dict[tuple[int, ...], int]
FrameExpectation: TypeAlias = dict[int, StorageIdxToFrame]


@pytest.mark.parametrize("case", CASES, ids=_name_case)
def test_cases(case: AcquisitionSettings, any_backend: str, tmp_path: Path) -> None:
    is_tiff = any_backend == "tiff"
    ext = ".ome.tiff" if is_tiff else ".ome.zarr"
    # Use model_copy to avoid cached_property contamination across tests
    case = case.model_copy(
        update={
            "root_path": str(tmp_path / f"output{ext}"),
            "backend": any_backend,
        }
    )

    # -------------- Write out all frames --------------

    try:
        write_encoded_data(case, real_unbounded_count=UNBOUNDED_FRAME_COUNT)
    except NotImplementedError as e:
        if re.match("Backend .* does not support settings", str(e)):
            pytest.xfail(f"Backend does not support this configuration: {e}")
            return
        raise

    if is_tiff:
        _assert_valid_ome_tiff(case)
    else:
        _assert_valid_ome_zarr(case)


@pytest.mark.parametrize("fmt", ["tiff", "zarr"])
def test_auto_backend(tmp_path: Path, fmt: str) -> None:
    # just exercise the "auto" backend selection path
    suffix = f".{fmt}"
    settings = CASES[1].model_copy(
        update={
            "root_path": str(tmp_path / f"output.ome{suffix}"),
            "backend": "auto",
        }
    )
    frame_shape = tuple(d.count for d in settings.dimensions[-2:])
    try:
        stream = create_stream(settings)
    except Exception as e:
        if "No available backends" in str(e) or "Could not find compatible" in str(e):
            pytest.xfail(f"No available backend for format '{fmt}': {e}")
            return
        raise

    with stream:
        for _ in range(settings.num_frames or 1):
            stream.append(np.empty(frame_shape, dtype=settings.dtype))

    dest = Path(settings.root_path)
    assert dest.exists()
    assert dest.suffix == suffix
    assert dest.is_dir() == (fmt == "zarr")


def test_overwrite_safety(tmp_path: Path, any_backend: str) -> None:
    """Test that attempting to overwrite existing files raises an error."""
    ext = ".ome.tiff" if any_backend == "tiff" else ".ome.zarr"
    root_path = tmp_path / f"output{ext}"
    settings = AcquisitionSettings(
        root_path=str(root_path),
        dimensions=[
            D(name="z", count=2, type="space"),
            D(name="y", count=64, chunk_size=64, type="space", scale=0.1),
            D(name="x", count=64, chunk_size=64, type="space", scale=0.1),
        ],
        dtype="uint16",
        backend=any_backend,
    )

    # First write should succeed
    with create_stream(settings) as stream:
        for _ in range(2):
            stream.append(np.empty((64, 64), dtype=settings.dtype))

    # grab snapshot of tree complete tree-structure for later comparison
    root_mtime = root_path.stat().st_mtime

    # Second write should fail due to existing data
    with pytest.raises(FileExistsError):
        with create_stream(settings) as stream:
            for _ in range(2):
                stream.append(np.empty((64, 64), dtype=settings.dtype))

    assert root_path.stat().st_mtime == root_mtime, (
        "Directory modification time changed despite failed overwrite"
    )

    time.sleep(0.2)  # ensure mtime difference on Windows
    # add back overwrite=True to settings and verify it works
    settings = settings.model_copy(update={"overwrite": True})
    with create_stream(settings) as stream:
        for _ in range(2):
            stream.append(np.empty((64, 64), dtype=settings.dtype))

    new_stamp = root_path.stat().st_mtime
    assert new_stamp > root_mtime, (
        "Directory modification time not updated on overwrite"
    )


@pytest.mark.parametrize("avail_mem", [2_000_000_000, 100_000_000_000])
def test_chunk_memory_warning(
    avail_mem: int, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Test that large chunk buffering triggers memory warning with low memory."""
    # Mock available memory to be low (2 GB)
    # Config uses 64 chunks x 33.6MB = 2.15GB
    # With 80% threshold: 2GB * 0.8 = 1.6GB < 2.15GB → should warn
    mock_get_memory = Mock(return_value=avail_mem)
    monkeypatch.setattr(_memory, "_get_available_memory", mock_get_memory)
    monkeypatch.setattr(_memory.sys, "platform", "win32")  # Pretend we're on Windows

    ctx = (
        pytest.warns(UserWarning, match="Chunk buffering may use")
        if avail_mem < 5_000_000_000
        else contextlib.nullcontext()
    )
    with ctx:
        AcquisitionSettings(
            root_path=str(tmp_path / "output.ome.zarr"),
            dimensions=[
                D(name="z", count=8, chunk_size=4, type="space"),
                D(name="c", count=64, chunk_size=1, type="channel"),
                D(name="y", count=2048, chunk_size=64, type="space"),
                D(name="x", count=2048, chunk_size=64, type="space"),
            ],
            dtype="uint16",
            backend="zarr",
        )


# ---------------------- Helpers for validation ----------------------


def _assert_valid_ome_tiff(case: AcquisitionSettings) -> None:
    try:
        import ome_types
        import tifffile
    except ImportError:
        pytest.skip("ome-types and tifffile are required for OME-TIFF validation")
        return

    # Collect expected file paths based on position count
    if (num_positions := len(case.positions)) == 1:
        # Single file for single position
        file_paths = [Path(case.root_path)]
    else:
        # Multiple files for multiple positions
        # Follow the naming pattern from TiffBackend._prepare_files
        file_paths = [
            Path(case.root_path.replace(".ome.tiff", f"_p{p_idx:03d}.ome.tiff"))
            for p_idx in range(num_positions)
        ]

    assert len(file_paths) == num_positions

    # Validate each TIFF file
    storage_dims = case.array_storage_dimensions
    for pos_idx, file_path in enumerate(file_paths):
        assert file_path.exists(), f"Expected TIFF file not found: {file_path}"

        # Read and validate OME metadata
        ome_metadata = ome_types.from_tiff(file_path)
        assert ome_metadata is not None, f"Failed to read OME metadata from {file_path}"
        assert len(ome_metadata.images) > 0, f"No images in OME metadata: {file_path}"

        # Get the first image (each file has one image per position)
        pixels = ome_metadata.images[0].pixels

        # Validate basic dimensional properties
        expected_shape = {
            d.name.upper(): d.count or UNBOUNDED_FRAME_COUNT for d in storage_dims
        }

        # Reverse because OME dimension order has fastest-varying dimension on right,
        # but storage order has slowest-varying dimension first
        expected_order = "".join(d.name.upper() for d in reversed(storage_dims))
        assert pixels.dimension_order.value.startswith(expected_order)

        assert pixels.size_x == expected_shape.get("X", 1)
        assert pixels.size_y == expected_shape.get("Y", 1)
        assert pixels.size_z == expected_shape.get("Z", 1)
        assert pixels.size_c == expected_shape.get("C", 1)
        assert pixels.size_t == expected_shape.get("T", 1)

        # Validate dtype (basic check - OME types use uppercase enum names)
        assert pixels.type.numpy_dtype == case.dtype

        # Verify we can read the actual TIFF data
        data = tifffile.imread(file_path)
        assert np.prod(data.shape[:-2]) == pixels.size_t * pixels.size_c * pixels.size_z
        validate_encoded_frame_values(
            data, [d.name for d in storage_dims[:-2]], pos_idx
        )


def _assert_valid_ome_zarr(case: AcquisitionSettings) -> None:
    root = Path(case.root_path)
    group = yaozarrs.validate_zarr_store(root)
    ome_meta = group.ome_metadata()

    # Assert group type (single image, multi-position, plate)
    # and collect image paths for further validation
    image_paths: list[Path] = []
    num_positions = len(case.positions)
    # Plate
    if case.plate is not None:
        assert isinstance(ome_meta, v05.Plate)
        image_paths = [root / p.row / p.column / p.name for p in case.positions]  # ty: ignore[unsupported-operator]
    # Single image
    elif num_positions == 1:
        assert isinstance(ome_meta, v05.Image)
        image_paths = [root]
    # Multi-position
    else:
        assert isinstance(ome_meta, v05.Bf2Raw)
        ome_group = yaozarrs.validate_ome_uri(root / "OME")
        assert isinstance(ome_group.attributes.ome, v05.Series)
        image_paths = [root / pos.name for pos in case.positions]

    assert len(image_paths) == num_positions

    storage_dims = case.array_storage_dimensions
    for pos_idx, image_path in enumerate(image_paths):
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
            _validate_array_tensorstore(
                group,
                storage_dims,
                case.dtype,
                pos_idx,
                az_hack=case.backend == "acquire-zarr" and len(storage_dims) == 2,
            )
        else:
            _validate_array_zarr(group, storage_dims, case.dtype, pos_idx)


def _validate_array_tensorstore(
    group: yaozarrs.ZarrGroup,
    storage_dims: list[D],
    dtype: str,
    pos_idx: int,
    az_hack: bool = False,
) -> None:
    """Validate an array stored on disk using tensorstore."""
    import tensorstore

    array0 = group["0"].to_tensorstore()
    assert isinstance(array0, tensorstore.TensorStore)

    # check on disk shape, dtype, dimension order and labels
    stored_shape = array0.shape
    stored_labels = [d.label for d in array0.domain]  # type: ignore
    stored_chunk_shape = array0.chunk_layout.read_chunk.shape
    if az_hack:
        # adjust for phantom dimension added by acquire-zarr for 2D images
        # https://github.com/acquire-project/acquire-zarr/issues/183
        assert stored_labels[0] == "_singleton", "Hack not present?  Fix tests?"
        stored_shape = stored_shape[1:]
        stored_labels = stored_labels[1:]
        stored_chunk_shape = stored_chunk_shape[1:]  # ty: ignore

    assert stored_shape == tuple(d.count or UNBOUNDED_FRAME_COUNT for d in storage_dims)
    assert array0.dtype == np.dtype(dtype)
    assert stored_labels == [d.name for d in storage_dims]

    # validate chunking
    expected_chunk_shape = tuple(d.chunk_size or 1 for d in storage_dims)
    assert stored_chunk_shape == expected_chunk_shape, (
        f"expected {expected_chunk_shape}, got {stored_chunk_shape}"
    )

    validate_encoded_frame_values(
        array0.read().result(), [d.name for d in storage_dims[:-2]], pos_idx
    )


def _validate_array_zarr(
    group: yaozarrs.ZarrGroup,
    storage_dims: list[D],
    dtype: str,
    pos_idx: int,
) -> None:
    """Validate an array stored on disk using zarr-python."""
    import zarr

    array0 = group["0"].to_zarr_python()
    assert isinstance(array0, zarr.Array)

    # check on disk shape, dtype
    assert array0.shape == tuple(d.count or UNBOUNDED_FRAME_COUNT for d in storage_dims)
    assert str(array0.dtype) == dtype

    # validate chunking
    expected_chunk_shape = tuple(d.chunk_size or 1 for d in storage_dims)
    assert array0.chunks == expected_chunk_shape, (
        f"expected {expected_chunk_shape}, got {array0.chunks}"
    )

    validate_encoded_frame_values(
        array0[:], [d.name for d in storage_dims[:-2]], pos_idx
    )
