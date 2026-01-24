from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar, Literal

from ome_writers._backends._yaozarrs import YaozarrsBackend

if TYPE_CHECKING:
    from donfig.config_obj import ConfigSet

    from ome_writers._router import FrameRouter
    from ome_writers._schema import AcquisitionSettings


class ZarrBackend(YaozarrsBackend):
    """OME-Zarr writer using zarr-python via yaozarrs."""

    _config_set: ConfigSet | None = None
    _zarr_config: ClassVar[dict[str, object]] = {
        # Disable write_empty_chunks optimization
        "array.write_empty_chunks": True,
    }

    def _get_yaozarrs_writer(self) -> Literal["zarr"]:
        return "zarr"

    def prepare(self, settings: AcquisitionSettings, router: FrameRouter) -> None:
        import zarr

        self._config_set = zarr.config.set(self._zarr_config)
        super().prepare(settings, router)

    def finalize(self) -> None:
        super().finalize()
        if self._config_set is not None:
            self._config_set.__exit__(None, None, None)


class ZarrsBackend(ZarrBackend):
    """OME-Zarr writer using zarrs-python Rust codecs via yaozarrs."""

    _zarr_config: ClassVar[dict[str, object]] = {
        **ZarrBackend._zarr_config,
        "codec_pipeline.path": "zarrs.ZarrsCodecPipeline",
    }
