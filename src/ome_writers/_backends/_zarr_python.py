from __future__ import annotations

from typing import TYPE_CHECKING, Literal

from ome_writers._backends._yaozarrs import YaozarrsBackend

if TYPE_CHECKING:
    from ome_writers._schema import AcquisitionSettings


class ZarrBackend(YaozarrsBackend):
    """OME-Zarr writer using zarr-python via yaozarrs."""

    def _get_writer(self) -> Literal["zarr"]:
        return "zarr"

    def is_incompatible(self, settings: AcquisitionSettings) -> Literal[False] | str:
        if not settings.root_path.endswith(".zarr"):  # pragma: no cover
            return "Root path must end with .zarr for ZarrBackend."
        return False
