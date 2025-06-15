from __future__ import annotations

from typing import Literal, NamedTuple

OME_DIM_TYPE = {"y": "space", "x": "space", "z": "space", "t": "time", "c": "channel"}
OME_UNIT = {"um": "micrometer", "ml": "milliliter", "s": "second", None: "unknown"}


class DimensionInfo(NamedTuple):
    label: Literal["x", "y", "z", "t", "c", "p", "other"]
    size: int
    unit: tuple[float, str] | None = None
    # None or 0 indicates no constraint.
    # -1 indicates that the chunk size should equal the full extent of the domain.
    chunk_size: int | None = 1

    @property
    def ome_dim_type(self) -> Literal["space", "time", "channel", "other"]:
        return OME_DIM_TYPE.get(self.label, "other")  # type: ignore

    @property
    def ome_unit(self) -> str:
        if isinstance(self.unit, tuple):
            return OME_UNIT.get(self.unit[1], "unknown")
        return "unknown"

    @property
    def ome_scale(self) -> float:
        if isinstance(self.unit, tuple):
            return self.unit[0]
        return 1.0
