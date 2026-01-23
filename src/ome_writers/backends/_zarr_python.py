from __future__ import annotations

from typing import Literal

from ome_writers.backends._yaozarrs import YaozarrsBackend


class ZarrBackend(YaozarrsBackend):
    """OME-Zarr writer using zarr-python via yaozarrs."""

    def _get_writer(self) -> Literal["zarr"]:
        return "zarr"
