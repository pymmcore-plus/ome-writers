# /// script
# requires-python = ">=3.11,<3.14"
# dependencies = [
#     "ome-writers[all]",
#     "zen-api-python",
# ]
#
# [tool.uv.sources]
# ome-writers = { path = "../" }
# ///
"""Example: streaming CZI pixel data from ZEN-API directly into OME-Zarr.

This example shows how to bridge the ZEN-API pixel stream (gRPC async)
into an OME-Zarr file using ome_writers.  Each 2D frame that arrives from
the microscope is appended to the Zarr store in real time, with per-frame
stage-position metadata preserved.

Two variants are shown:

1. **Known dimensions** – the experiment structure (T, C, Z, …) is
   declared up front and matches the ZEN experiment being run.
2. **Unbounded time-lapse** – only XY + channels are declared; the time
   axis is left open-ended (``count=None``) so frames are accepted until
   the experiment finishes.

Requirements
------------
* A running ZEN instance with the ZEN-API gRPC service enabled.
* ``zen-api-python`` installed (provides the auto-generated gRPC stubs).
* ``ome-writers`` installed (``pip install ome-writers[all]``).
* A ``config.ini`` next to this script (host/port/token for ZEN-API).

Adapt the constants in the ``if __name__`` block to match your experiment.
"""

from __future__ import annotations

import asyncio
import datetime
from pathlib import Path

import numpy as np

from ome_writers import AcquisitionSettings, Channel, Dimension, Position, create_stream

# ZEN-API gRPC stubs (auto-generated from the ZEN-API protobuf definitions)
from zen_api.acquisition.v1beta import (
    ExperimentServiceLoadRequest,
    ExperimentServiceStartExperimentRequest,
    ExperimentServiceStub,
    ExperimentStreamingServiceMonitorAllExperimentsRequest,
    ExperimentStreamingServiceMonitorExperimentRequest,
    ExperimentStreamingServiceStub,
)

# ZEN-API helper shipped with the OAD examples
from zen_api_utils.misc import initialize_zenapi


# ---------------------------------------------------------------------------
# Variant 1 – Known experiment dimensions
# ---------------------------------------------------------------------------


async def stream_to_ome_zarr(
    config: str | Path,
    output_path: str | Path,
    *,
    # Experiment shape – adapt to your .czexp
    num_timepoints: int = 10,
    num_channels: int = 2,
    num_zplanes: int = 5,
    frame_width: int = 1024,
    frame_height: int = 1024,
    pixel_type: np.dtype = np.dtype(np.uint16),
    pixel_size_um: float = 0.325,
    z_step_um: float = 1.0,
    channel_names: list[str] | None = None,
    # ZEN-API options
    experiment_name: str | None = None,
    experiment_id: str | None = None,
    channel_index: int | None = None,
) -> None:
    """Stream a ZEN experiment with known dimensions into OME-Zarr.

    Either pass *experiment_name* to start a new experiment via the API, or
    set *experiment_id* / leave both ``None`` to monitor an experiment that
    was already started from the ZEN UI.
    """
    channel, metadata = initialize_zenapi(str(config))

    # --- optionally start the experiment via ZEN-API ----------------------
    if experiment_name is not None:
        exp_svc = ExperimentServiceStub(channel=channel, metadata=metadata)
        exp = await exp_svc.load(
            ExperimentServiceLoadRequest(experiment_name=experiment_name)
        )
        await exp_svc.start_experiment(
            ExperimentServiceStartExperimentRequest(
                experiment_id=exp.experiment_id,
                output_name="zenapi_ome_zarr",
            )
        )
        experiment_id = exp.experiment_id

    # --- set up ome_writers -----------------------------------------------
    if channel_names is None:
        channel_names = [f"Ch{i}" for i in range(num_channels)]

    settings = AcquisitionSettings(
        root_path=str(output_path),
        # Dimension order must match the ZEN experiment acquisition order.
        # ZEN default: Time (slowest) → Z → Channel (fastest before XY).
        # Adjust if your experiment uses a different axis order.
        dimensions=[
            Dimension(
                name="t",
                count=num_timepoints,
                chunk_size=1,
                type="time",
            ),
            Dimension(
                name="z",
                count=num_zplanes,
                chunk_size=1,
                type="space",
                scale=z_step_um,
                unit="micrometer",
            ),
            Dimension(
                name="c",
                count=num_channels,
                chunk_size=1,
                type="channel",
                coords=[Channel(name=n) for n in channel_names],
            ),
            Dimension(
                name="y",
                count=frame_height,
                chunk_size=frame_height,
                type="space",
                scale=pixel_size_um,
                unit="micrometer",
            ),
            Dimension(
                name="x",
                count=frame_width,
                chunk_size=frame_width,
                type="space",
                scale=pixel_size_um,
                unit="micrometer",
            ),
        ],
        dtype=str(pixel_type),
        format="ome-zarr",
        overwrite=True,
    )

    # --- stream frames from ZEN into the Zarr store -----------------------
    streaming_svc = ExperimentStreamingServiceStub(
        channel=channel, metadata=metadata
    )

    if experiment_id is not None:
        responses = streaming_svc.monitor_experiment(
            ExperimentStreamingServiceMonitorExperimentRequest(
                experiment_id=experiment_id,
                channel_index=channel_index,
            )
        )
    else:
        # monitor whatever experiment is running (started from ZEN UI)
        responses = streaming_svc.monitor_all_experiments(
            ExperimentStreamingServiceMonitorAllExperimentsRequest(
                channel_index=channel_index,
            )
        )

    start = datetime.datetime.now()

    with create_stream(settings) as stream:
        async for response in responses:
            fd = response.frame_data

            # reconstruct the 2D numpy frame
            frame = np.frombuffer(
                fd.pixel_data.raw_data, dtype=pixel_type
            ).reshape(fd.frame_size.height, fd.frame_size.width)

            # flip + rotate to match ZEN orientation
            frame = np.ascontiguousarray(np.flipud(np.rot90(frame)))

            # per-frame metadata (stage coordinates and timing)
            meta = {
                "delta_t": (datetime.datetime.now() - start).total_seconds(),
                "position_x": fd.frame_stage_position.x * 1e6,  # m → µm
                "position_y": fd.frame_stage_position.y * 1e6,
                "position_z": fd.frame_stage_position.z * 1e6,
            }

            stream.append(frame, frame_metadata=meta)

            fp = fd.frame_position
            print(
                f"T={fp.t} Z={fp.z} C={fp.c} | "
                f"stage=({meta['position_x']:.1f}, {meta['position_y']:.1f}, "
                f"{meta['position_z']:.1f}) µm"
            )

    channel.close()
    print(f"Done – OME-Zarr written to {settings.output_path}")


# ---------------------------------------------------------------------------
# Variant 2 – Unbounded time-lapse (unknown number of timepoints)
# ---------------------------------------------------------------------------


async def stream_to_ome_zarr_unbounded(
    config: str | Path,
    output_path: str | Path,
    *,
    num_channels: int = 2,
    frame_width: int = 1024,
    frame_height: int = 1024,
    pixel_type: np.dtype = np.dtype(np.uint16),
    pixel_size_um: float = 0.325,
    channel_names: list[str] | None = None,
    channel_index: int | None = None,
) -> None:
    """Stream frames into OME-Zarr without knowing the number of timepoints.

    The time dimension is declared with ``count=None`` so the Zarr array
    grows automatically as frames arrive.  The stream ends when the ZEN
    experiment finishes (the async iterator completes).
    """
    channel, metadata = initialize_zenapi(str(config))

    if channel_names is None:
        channel_names = [f"Ch{i}" for i in range(num_channels)]

    settings = AcquisitionSettings(
        root_path=str(output_path),
        dimensions=[
            # count=None → unbounded; only the first dimension can be unbounded
            Dimension(name="t", count=None, chunk_size=1, type="time"),
            Dimension(
                name="c",
                count=num_channels,
                chunk_size=1,
                type="channel",
                coords=[Channel(name=n) for n in channel_names],
            ),
            Dimension(
                name="y",
                count=frame_height,
                chunk_size=frame_height,
                type="space",
                scale=pixel_size_um,
                unit="micrometer",
            ),
            Dimension(
                name="x",
                count=frame_width,
                chunk_size=frame_width,
                type="space",
                scale=pixel_size_um,
                unit="micrometer",
            ),
        ],
        dtype=str(pixel_type),
        format="ome-zarr",
        overwrite=True,
    )

    streaming_svc = ExperimentStreamingServiceStub(
        channel=channel, metadata=metadata
    )
    responses = streaming_svc.monitor_all_experiments(
        ExperimentStreamingServiceMonitorAllExperimentsRequest(
            channel_index=channel_index,
        )
    )

    start = datetime.datetime.now()
    frame_count = 0

    with create_stream(settings) as stream:
        async for response in responses:
            fd = response.frame_data

            frame = np.frombuffer(
                fd.pixel_data.raw_data, dtype=pixel_type
            ).reshape(fd.frame_size.height, fd.frame_size.width)
            frame = np.ascontiguousarray(np.flipud(np.rot90(frame)))

            meta = {
                "delta_t": (datetime.datetime.now() - start).total_seconds(),
                "position_x": fd.frame_stage_position.x * 1e6,
                "position_y": fd.frame_stage_position.y * 1e6,
                "position_z": fd.frame_stage_position.z * 1e6,
            }

            stream.append(frame, frame_metadata=meta)
            frame_count += 1

            fp = fd.frame_position
            print(
                f"frame {frame_count}: T={fp.t} C={fp.c} | "
                f"stage=({meta['position_x']:.1f}, {meta['position_y']:.1f}, "
                f"{meta['position_z']:.1f}) µm"
            )

    channel.close()
    print(f"Done – {frame_count} frames written to {settings.output_path}")


# ---------------------------------------------------------------------------
# Variant 3 – Multi-position (scenes/tiles) with per-position Zarr arrays
# ---------------------------------------------------------------------------


async def stream_to_ome_zarr_multiposition(
    config: str | Path,
    output_path: str | Path,
    *,
    num_timepoints: int = 10,
    num_channels: int = 2,
    num_positions: int = 3,
    frame_width: int = 1024,
    frame_height: int = 1024,
    pixel_type: np.dtype = np.dtype(np.uint16),
    pixel_size_um: float = 0.325,
    channel_names: list[str] | None = None,
    position_names: list[str] | None = None,
    channel_index: int | None = None,
) -> None:
    """Stream a multi-position ZEN experiment into OME-Zarr.

    Each ZEN scene/tile becomes a separate position array in the Zarr store,
    following the OME-NGFF multi-image convention.
    """
    channel, metadata = initialize_zenapi(str(config))

    if channel_names is None:
        channel_names = [f"Ch{i}" for i in range(num_channels)]
    if position_names is None:
        position_names = [f"Pos{i}" for i in range(num_positions)]

    settings = AcquisitionSettings(
        root_path=str(output_path),
        # ZEN multi-position order: Time → Position/Scene → Channel → YX
        dimensions=[
            Dimension(name="t", count=num_timepoints, chunk_size=1, type="time"),
            Dimension(
                name="p",
                type="position",
                coords=[Position(name=n) for n in position_names],
            ),
            Dimension(
                name="c",
                count=num_channels,
                chunk_size=1,
                type="channel",
                coords=[Channel(name=n) for n in channel_names],
            ),
            Dimension(
                name="y",
                count=frame_height,
                chunk_size=frame_height,
                type="space",
                scale=pixel_size_um,
                unit="micrometer",
            ),
            Dimension(
                name="x",
                count=frame_width,
                chunk_size=frame_width,
                type="space",
                scale=pixel_size_um,
                unit="micrometer",
            ),
        ],
        dtype=str(pixel_type),
        format="ome-zarr",
        overwrite=True,
    )

    streaming_svc = ExperimentStreamingServiceStub(
        channel=channel, metadata=metadata
    )
    responses = streaming_svc.monitor_all_experiments(
        ExperimentStreamingServiceMonitorAllExperimentsRequest(
            channel_index=channel_index,
        )
    )

    start = datetime.datetime.now()

    with create_stream(settings) as stream:
        async for response in responses:
            fd = response.frame_data

            frame = np.frombuffer(
                fd.pixel_data.raw_data, dtype=pixel_type
            ).reshape(fd.frame_size.height, fd.frame_size.width)
            frame = np.ascontiguousarray(np.flipud(np.rot90(frame)))

            meta = {
                "delta_t": (datetime.datetime.now() - start).total_seconds(),
                "position_x": fd.frame_stage_position.x * 1e6,
                "position_y": fd.frame_stage_position.y * 1e6,
                "position_z": fd.frame_stage_position.z * 1e6,
            }

            stream.append(frame, frame_metadata=meta)

            fp = fd.frame_position
            print(
                f"T={fp.t} P={fp.s} C={fp.c} | "
                f"stage=({meta['position_x']:.1f}, {meta['position_y']:.1f}, "
                f"{meta['position_z']:.1f}) µm"
            )

    channel.close()
    print(f"Done – OME-Zarr written to {settings.output_path}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # ---- Adapt these to your ZEN experiment setup ----
    CONFIG = Path(__file__).parent / "config.ini"

    # Variant 1: known dimensions (T=10, Z=5, C=2)
    asyncio.run(
        stream_to_ome_zarr(
            config=CONFIG,
            output_path="zenapi_known_dims",
            num_timepoints=10,
            num_channels=2,
            num_zplanes=5,
            frame_width=1024,
            frame_height=1024,
            pixel_type=np.dtype(np.uint16),
            pixel_size_um=0.325,
            z_step_um=1.0,
            channel_names=["DAPI", "GFP"],
            # Set to None to monitor an experiment started from the ZEN UI;
            # or provide experiment_name="MyExp" to start one via the API.
            experiment_name=None,
        )
    )

    # Variant 2: unbounded time-lapse (uncomment to use)
    # asyncio.run(
    #     stream_to_ome_zarr_unbounded(
    #         config=CONFIG,
    #         output_path="zenapi_unbounded",
    #         num_channels=2,
    #         frame_width=1024,
    #         frame_height=1024,
    #         channel_names=["DAPI", "GFP"],
    #     )
    # )

    # Variant 3: multi-position (uncomment to use)
    # asyncio.run(
    #     stream_to_ome_zarr_multiposition(
    #         config=CONFIG,
    #         output_path="zenapi_multipos",
    #         num_timepoints=10,
    #         num_channels=2,
    #         num_positions=3,
    #         frame_width=1024,
    #         frame_height=1024,
    #         channel_names=["DAPI", "GFP"],
    #         position_names=["Pos0", "Pos1", "Pos2"],
    #     )
    # )
