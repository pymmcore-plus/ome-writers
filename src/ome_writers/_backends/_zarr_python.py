from __future__ import annotations

from typing import TYPE_CHECKING, Literal

from ome_writers._backends._yaozarrs import YaozarrsBackend

if TYPE_CHECKING:
    from ome_writers._router import FrameRouter
    from ome_writers._schema import AcquisitionSettings


class ZarrBackend(YaozarrsBackend):
    """OME-Zarr writer using zarr-python via yaozarrs."""

    def _get_writer(self) -> Literal["zarr"]:
        return "zarr"


class ZarrsBackend(ZarrBackend):
    """OME-Zarr writer using zarr-python via yaozarrs."""

    def prepare(self, settings: AcquisitionSettings, router: FrameRouter) -> None:
        import zarr

        self._previous_pipeline = zarr.config.get("codec_pipeline.path")
        zarr.config.set({"codec_pipeline.path": "zarrs.ZarrsCodecPipeline"})
        super().prepare(settings, router)

    def finalize(self) -> None:
        super().finalize()
        if (pipeline := getattr(self, "_previous_pipeline", None)) is not None:
            import zarr

            zarr.config.set({"codec_pipeline.path": pipeline})
            del self._previous_pipeline
