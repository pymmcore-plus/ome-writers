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

    from ome_writers._router import FrameRouter, PositionInfo
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
        self._storage_dims: list[Dimension] = []
        self._num_positions: int = 0
        self._is_hcs: bool = False

    def is_incompatible(self, settings: AcquisitionSettings) -> Literal[False] | str:
        """Check if settings are compatible with acquire-zarr.

        Acquire-zarr requires:
        - Root path ending with .zarr
        - storage_order="acquisition" (sequential writes only)
        """
        if not settings.root_path.endswith(".zarr"):
            return "Root path must end with .zarr for AcquireZarrBackend."

        # TODO
        # # Acquire-zarr only supports sequential writes in acquisition order
        # if settings.array_settings.storage_order != "acquisition":
        #     return (
        #         'AcquireZarrBackend requires storage_order="acquisition" '
        #         "(sequential writes only). "
        #         f"Got: {settings.array_settings.storage_order!r}"
        #     )

        return False

    def prepare(self, settings: AcquisitionSettings, router: FrameRouter) -> None:
        """Initialize acquire-zarr stream."""
        self._finalized = False
        self._root_path = Path(settings.root_path).expanduser().resolve()

        # Get positions and dimensions from router
        self._positions = positions = router.positions
        self._num_positions = len(positions)
        self._storage_dims = storage_dims = router.array_storage_dimensions

        # Handle overwrite
        if self._root_path.exists():
            if not settings.overwrite:
                raise FileExistsError(
                    f"Directory {self._root_path} already exists. "
                    "Use overwrite=True to overwrite it."
                )
            shutil.rmtree(self._root_path)

        self._is_hcs = False
        if settings.plate is not None:
            self._is_hcs = True
            raise NotImplementedError(
                "AcquireZarrBackend does not yet support HCS (plates)."
            )

        # Convert to az dimensions
        az_dims = [_to_acquire_dim(dim) for dim in storage_dims]

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
                    data_type=settings.array_settings.dtype,
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
        self._stream.append(frame, key=az_pos_key)

    def finalize(self) -> None:
        """Close stream and patch metadata."""
        if not self._finalized and self._stream is not None:
            self._stream.close()
            self._stream = None
            gc.collect()

            self._patch_metadata()
            self._finalized = True

    def _patch_metadata(self) -> None:
        """Patch Zarr metadata to NGFF v0.5 compliance.

        In certain scenarios, we need to slightly modify the metadata generated
        by acquire-zarr to ensure full compliance with OME-NGFF v0.5
        """
        if (
            self._root_path is None
            or not (zarr_json := self._root_path / "zarr.json").exists()
        ):
            return

        if self._is_hcs:
            ...
        elif self._num_positions > 1:
            # create valid bioformats2raw series group
            # Create root zarr.json with bioformats2raw.layout
            _create_zarr3_group(self._root_path, v05.Bf2Raw(bioformats2raw_layout=3))

            # Create OME/zarr.json with series list
            ome_path = self._root_path / "OME"
            _create_zarr3_group(
                ome_path, v05.Series(series=[p.name for p in self._positions])
            )

        # Validate that zarr.json is valid
        try:
            with open(zarr_json) as f:
                json.load(f)
        except json.JSONDecodeError:
            # If metadata is malformed, we can't patch it
            return


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


def _to_acquire_dim(dim: Dimension) -> az.Dimension:
    """Convert a Dimension to az.Dimension."""
    # Map dimension type to az DimensionType
    dim_type_map = {
        "time": az.DimensionType.TIME,
        "channel": az.DimensionType.CHANNEL,
        "space": az.DimensionType.SPACE,
    }

    return az.Dimension(
        name=dim.name,
        kind=dim_type_map.get(dim.type, az.DimensionType.OTHER),
        array_size_px=dim.count or 1,
        chunk_size_px=dim.chunk_size or 1,
        shard_size_chunks=dim.shard_size or 1,
    )
