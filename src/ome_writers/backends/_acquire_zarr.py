"""Acquire-zarr backend for OME-Zarr v3 (sequential writes only)."""

from __future__ import annotations

import gc
import json
import shutil
from pathlib import Path
from typing import TYPE_CHECKING, Literal

from yaozarrs import v05

from ome_writers._backend import ArrayBackend

if TYPE_CHECKING:
    import numpy as np
    from pydantic import BaseModel

    from ome_writers._router import FrameRouter, Position, PositionInfo
    from ome_writers.schema import AcquisitionSettings, Dimension

try:
    import acquire_zarr as az
except ImportError as e:
    raise ImportError(
        f"{__name__} requires acquire-zarr: `pip install acquire-zarr`."
    ) from e


class AcquireZarrBackend(ArrayBackend):
    """OME-Zarr v3 backend using acquire-zarr (sequential writes only).

    Acquire-zarr is designed for high-performance streaming acquisition and only
    supports sequential writes. Frames must be appended in acquisition order.
    This backend requires storage_order="acquisition" in settings.
    """

    def __init__(self) -> None:
        self._stream: az.ZarrStream | None = None
        self._finalized = False
        self._root_path: Path | None = None
        self._storage_dims: tuple[Dimension, ...] = ()
        self._num_positions: int = 0
        self._is_hcs: bool = False
        self._used_2d_hack: bool = False

    def is_incompatible(self, settings: AcquisitionSettings) -> Literal[False] | str:
        """Check if settings are compatible with acquire-zarr.

        Acquire-zarr requires:
        - Root path ending with .zarr
        - storage_order="acquisition" (sequential writes only)
        """
        if not settings.root_path.endswith(".zarr"):  # pragma: no cover
            return "Root path must end with .zarr for AcquireZarrBackend."

        if settings.storage_index_permutation is not None:
            return (
                "AcquireZarrBackend does not currently support permuted storage order. "
                "Data may only be written in acquisition order."
            )

        return False

    def prepare(self, settings: AcquisitionSettings, router: FrameRouter) -> None:
        """Initialize acquire-zarr stream."""
        self._finalized = False
        self._root_path = Path(settings.root_path).expanduser().resolve()

        # Get positions and dimensions from router
        self._positions = positions = settings.positions
        self._num_positions = len(positions)
        self._storage_dims = storage_dims = settings.array_storage_dimensions

        # Handle overwrite
        if self._root_path.exists():
            if not settings.overwrite:
                raise FileExistsError(
                    f"Directory {self._root_path} already exists. "
                    "Use overwrite=True to overwrite it."
                )
            shutil.rmtree(self._root_path)

        # Convert to az dimensions
        # special-casing frame dimensions for default chunk sizes
        # TODO: maybe this should go in schema validation of the dimension list instead?
        # this can be extremely problematic if chunk_size stays at 1 for frame dims
        az_dims = [_to_acquire_dim(dim, False) for dim in storage_dims[:-2]]
        az_dims += [_to_acquire_dim(dim, True) for dim in storage_dims[-2:]]

        # Acquire-zarr requires at least 3 dimensions. For 2D images (Y, X only),
        # add a phantom Z dimension with size 1.
        if len(az_dims) == 2:
            self._used_2d_hack = True
            _inject_phantom_dim(az_dims)
        else:
            self._used_2d_hack = False

        # Check if this is a plate
        self._is_hcs = settings.plate is not None
        if self._is_hcs:
            # HCS Plate mode
            self._prepare_plate(settings, positions, az_dims)
        else:
            # Non-plate mode (single position or multi-position)
            self._prepare_non_plate(settings, positions, az_dims)

    def _prepare_non_plate(
        self, settings: AcquisitionSettings, positions: tuple, az_dims: list
    ) -> None:
        """Prepare acquire-zarr stream for non-plate acquisitions."""
        # Create ArraySettings for each position
        self._az_pos_keys = []
        az_arrays = []
        for pos in positions:
            # This is a hack to get a multiscales group, rather than a direct array
            # https://github.com/acquire-project/acquire-zarr/issues/181
            az_key = pos.name if self._num_positions > 1 else ""
            # this is a hack to get the proper hierarchy structure.
            # see https://github.com/acquire-project/acquire-zarr/issues/182
            downsample = az.DownsamplingMethod.MEAN if self._num_positions > 1 else None
            self._az_pos_keys.append(az_key)
            az_arrays.append(
                az.ArraySettings(
                    output_key=az_key,
                    dimensions=az_dims,
                    data_type=settings.dtype,
                    downsampling_method=downsample,
                )
            )

        # Create StreamSettings and ZarrStream
        stream_settings = az.StreamSettings(
            arrays=az_arrays,
            store_path=str(self._root_path),
            version=az.ZarrVersion.V3,
        )
        self._stream = az.ZarrStream(stream_settings)

    def _prepare_plate(
        self, settings: AcquisitionSettings, positions: tuple, az_dims: list
    ) -> None:
        """Prepare acquire-zarr stream for HCS plate acquisitions."""
        plate = settings.plate
        assert plate is not None

        # Create output keys for each position: "row/column/fov_name"
        # Use empty plate_path so the plate is at the root, not in a subdirectory
        self._az_pos_keys = [f"{pos.row}/{pos.column}/{pos.name}" for pos in positions]

        # Group positions by (row, column) to create wells
        # wells_map: {(row, col): [(pos_idx, Position), ...]}
        wells_map: dict[tuple[str, str], list[tuple[int, Position]]] = {}
        for idx, pos in enumerate(positions):
            wells_map.setdefault((pos.row, pos.column), []).append((idx, pos))

        # Create Wells with FieldOfViews
        az_wells = []
        for (row, col), pos_list in wells_map.items():
            fovs = []
            for _pos_idx, pos in pos_list:
                # Create ArraySettings for this FOV
                # Note: output_key should just be the FOV name since the well
                # structure already defines row/column
                # Use downsampling to create a multiscales group structure
                array_settings = az.ArraySettings(
                    output_key=pos.name,
                    dimensions=az_dims,
                    data_type=settings.dtype,
                    downsampling_method=az.DownsamplingMethod.MEAN,
                )
                # Create FieldOfView for each position in this well
                fov = az.FieldOfView(
                    path=pos.name,
                    acquisition_id=0,  # Default acquisition ID
                    array_settings=array_settings,
                )
                fovs.append(fov)

            # Create Well
            well = az.Well(row_name=row, column_name=col, images=fovs)
            az_wells.append(well)

        # Create Plate
        az_plate = az.Plate(
            path="",  # empty path so the plate is at the root of the store
            name=plate.name,
            row_names=plate.row_names,
            column_names=plate.column_names,
            wells=az_wells,
            acquisitions=[az.Acquisition(id=0, name="default")],
        )

        # Create StreamSettings with plate
        stream_settings = az.StreamSettings(
            hcs_plates=[az_plate],
            store_path=str(self._root_path),
            version=az.ZarrVersion.V3,
        )
        self._stream = az.ZarrStream(stream_settings)

    def write(
        self,
        position_info: PositionInfo,
        index: tuple[int, ...],
        frame: np.ndarray,
    ) -> None:
        """Write frame sequentially.

        Notes
        -----
        The index parameter is ignored because acquire-zarr is a sequential-only
        backend. Frames are written via append() in the order they arrive.
        """
        # Append sequentially - acquire-zarr doesn't use indices
        # The key matches the output_key we set in ArraySettings
        # the check on num positions matches the behavior above in prepare()
        az_pos_key = self._az_pos_keys[position_info[0]]
        self._stream.append(frame, key=az_pos_key)  # pyright: ignore (stream will not be None)

    def finalize(self) -> None:
        """Close stream and patch metadata."""
        if not self._finalized and self._stream is not None:
            self._stream.close()
            self._stream = None
            gc.collect()
            gc.collect()

            self._patch_metadata()
            self._finalized = True

    def _patch_metadata(self) -> None:
        """Patch Zarr metadata to NGFF v0.5 compliance.

        In certain scenarios, we need to slightly modify the metadata generated
        by acquire-zarr to ensure full compliance with OME-NGFF v0.5
        """
        if self._root_path is None:  # pragma: no cover
            return

        if not self._is_hcs and self._num_positions > 1:
            # create valid bioformats2raw series group
            # Create root zarr.json with bioformats2raw.layout
            _create_zarr3_group(
                self._root_path, v05.Bf2Raw.model_validate({"bioformats2raw.layout": 3})
            )

            # Create OME/zarr.json with series list
            ome_path = self._root_path / "OME"
            _create_zarr3_group(
                ome_path, v05.Series(series=[p.name for p in self._positions])
            )

        # Remove phantom Z dimension if we added one
        if self._used_2d_hack:
            _cleanup_2d_hack(self._root_path)


def _create_zarr3_group(
    dest_path: Path,
    ome_model: BaseModel,
    indent: int = 2,
) -> None:
    """Create a zarr group directory with optional OME metadata in zarr.json."""
    dest_path.mkdir(parents=True, exist_ok=True)
    zarr_json = {
        "zarr_format": 3,
        "node_type": "group",
        "attributes": {"ome": ome_model.model_dump(mode="json", exclude_none=True)},
    }
    (dest_path / "zarr.json").write_text(json.dumps(zarr_json, indent=indent))


def _to_acquire_dim(dim: Dimension, frame_dim: bool) -> az.Dimension:
    """Convert a Dimension to az.Dimension."""
    # Map dimension type to az DimensionType
    dim_type_map = {
        "time": az.DimensionType.TIME,
        "channel": az.DimensionType.CHANNEL,
        "space": az.DimensionType.SPACE,
    }

    if dim.chunk_size is not None:
        chunk_size = dim.chunk_size
    else:
        chunk_size = dim.count if frame_dim else 1

    return az.Dimension(
        name=dim.name,
        kind=dim_type_map.get(dim.type, az.DimensionType.OTHER),  # pyright: ignore[reportCallIssue]
        array_size_px=dim.count or 1,
        chunk_size_px=chunk_size,
        shard_size_chunks=dim.shard_size or 1,
    )


# -----------------
# code to deal with the fact that acquire-zarr doesn't directly support 2D images
# https://github.com/acquire-project/acquire-zarr/issues/183

HACK_AXIS_NAME = "singleton"


def _inject_phantom_dim(az_dims: list[az.Dimension]) -> None:
    az_dims.insert(
        0,
        az.Dimension(
            name=HACK_AXIS_NAME,
            kind=az.DimensionType.SPACE,
            array_size_px=1,
            chunk_size_px=1,
            shard_size_chunks=1,
        ),
    )


def _cleanup_2d_hack(root: Path) -> None:
    """Remove phantom Z dimension added for 2D images.

    https://github.com/acquire-project/acquire-zarr/issues/183
    """
    for zarr_json in list(root.rglob("zarr.json")):
        metadata = json.loads(zarr_json.read_text())
        if metadata.get("node_type") == "array":
            _fix_array_json_hack(metadata)
        elif metadata.get("node_type") == "group":
            _fix_group_json_hack(metadata)

        # Write back updated metadata
        zarr_json.write_text(json.dumps(metadata, indent=2))


def _fix_array_json_hack(metadata: dict) -> None:
    # https://github.com/acquire-project/acquire-zarr/issues/183
    # Remove phantom Z dimension from array meta
    if len(metadata["shape"]) > 2:
        metadata["shape"] = metadata["shape"][1:]
    if (
        (chunk_grid := metadata.get("chunk_grid"))
        and (chunk_config := chunk_grid.get("configuration"))
        and len(chunk_config.get("chunk_shape", [])) > 2
    ):
        chunk_config["chunk_shape"] = chunk_config["chunk_shape"][1:]
    if len(metadata.get("dimension_names", [])) > 2:
        metadata["dimension_names"] = metadata["dimension_names"][1:]
    for codec in metadata.get("codecs", []):
        if codec.get("name") == "sharding_indexed" and (
            cfg := codec.get("configuration")
        ):
            chunk_shape = cfg.get("chunk_shape", [])
            if len(chunk_shape) > 2:
                codec["configuration"]["chunk_shape"] = chunk_shape[1:]


def _fix_group_json_hack(metadata: dict) -> None:
    if not (ome_attrs := metadata.get("attributes", {}).get("ome")):  # pragma: no cover
        return

    # Update multiscales datasets and axes
    for multiscale in ome_attrs.get("multiscales", []):
        if "axes" in multiscale and len(axes := multiscale.get("axes", [])) > 2:
            if axes[0].get("name") == HACK_AXIS_NAME:
                multiscale["axes"] = axes[1:]

        for dataset in multiscale.get("datasets", []):
            for tform in dataset.get("coordinateTransformations", []):
                for type_ in ("scale", "translation"):
                    if type_ in tform and len(tform[type_]) > 2:
                        tform[type_] = tform[type_][1:]
