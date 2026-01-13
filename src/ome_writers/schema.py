from dataclasses import dataclass
from enum import Enum
from typing import Literal

import numpy as np
import numpy.typing as npt
from typing_extensions import Self

DimensionType = Literal["space", "time", "channel", "other"]
StandardAxisKey = Literal["x", "y", "z", "c", "t", "p"]


class StandardAxis(str, Enum):
    X = "x"
    Y = "y"
    Z = "z"
    CHANNEL = "c"
    TIME = "t"
    POSITION = "p"

    def __str__(self) -> str:
        return self.value

    def dimension_type(self) -> DimensionType:
        if self in {StandardAxis.X, StandardAxis.Y, StandardAxis.Z}:
            return "space"
        if self == StandardAxis.TIME:
            return "time"
        if self == StandardAxis.CHANNEL:
            return "channel"
        return "other"

    def unit(self) -> str | None:
        if self in {StandardAxis.X, StandardAxis.Y, StandardAxis.Z}:
            return "micrometer"
        if self == StandardAxis.TIME:
            return "ssecond"
        return None


@dataclass(slots=True, frozen=True)
class Dimension:
    name: str
    count: int | None  # must be passed, but can be None for unlimited axes
    chunk_size: int | None = None
    shard_size: int | None = None
    type: DimensionType | None = None
    unit: str | None = None
    scale: float | None = None
    translation: float | None = None


@dataclass(slots=True, frozen=True)
class Position:
    """A single acquisition position."""

    name: str
    row: str | None = None
    column: str | None = None


@dataclass(slots=True)
class PositionDimension:
    """Positions (meta-dimension) in acquisition order.

    Unlike Dimension, positions don't become an array axis—they become
    separate arrays/files (this is currently true for both OME-Zarr and OME-TIFF).
    The position of PositionDimension in the dimensions list determines when positions
    are visited during acquisition.
    """

    positions: list[Position]
    name: str = "p"

    @property
    def count(self) -> int:
        """Number of positions."""
        return len(self.positions)

    @property
    def names(self) -> list[str]:
        """Position names in acquisition order."""
        return [p.name for p in self.positions]


@dataclass(slots=True, frozen=True)
class ArraySettings:
    # PositionDimension can appear anywhere in the list to control acquisition order.
    # Its location determines when positions are visited relative to other dimensions.
    dimensions: list[Dimension | PositionDimension]
    dtype: np.dtype
    compression: str | None = None
    position_name: str | None = None
    storage_order: Literal["acquisition", "ngff"] | list[str] = "acquisition"
    # downsampling: str | None = None

    def __post_init__(self) -> None:
        dim_names = [dim.name for dim in self.dimensions]
        if len(dim_names) != len(set(dim_names)):
            raise ValueError("Dimension names must be unique within an ArraySettings.")

        # only the first dimension is allowed to have a count of None (unlimited)
        for dim in self.dimensions[1:]:
            if dim.count is None:
                raise ValueError(
                    "Only the first dimension may have count=None (unlimited)."
                )

    @property
    def shape(self) -> tuple[int | None, ...]:
        return tuple(dim.count for dim in self.dimensions)

    @classmethod
    def from_standard_axes(
        cls,
        sizes: dict[StandardAxisKey, int | None],
        dtype: npt.DTypeLike,
        axis_order: list[StandardAxisKey] | None = None,
        chunk_shapes: dict[StandardAxisKey | StandardAxis, int] | None = None,
    ) -> Self:
        """Convenience constructor for standard axes.

        Standard axes are "special" axis names {'x', 'y', 'z', 'c', 't', and 'p'}.
        With this constructor, dimension types and units are inferred from these names.
        Chunk shapes are set to 1 for non-XY dimensions by default.
        Axis order is optional, and will default to the order of the `sizes` dict.

        Examples
        --------
        >>> settings = ArraySettings.from_standard_axes(
        ...     sizes={"t": 10, "c": 2, "z": 5, "y": 512, "x": 512},
        ...     dtype=np.uint16,
        ...     chunk_shapes={"y": 64, "x": 64},
        ... )
        """
        if axis_order is None:
            axis_order = list(sizes.keys())

        try:
            std_axes = [StandardAxis(axis) for axis in axis_order]
        except ValueError as e:
            raise ValueError(
                f"All axes in axis_order must be one of {StandardAxis._member_names_}."
            ) from e

        # standardize chunk shapes -> all axes included, defaulted as needed
        x_or_y = {StandardAxis.X, StandardAxis.Y}
        chunk_shapes = chunk_shapes or {}
        for axis in std_axes:
            if axis not in chunk_shapes:
                if axis in x_or_y:
                    chunk_shapes[axis] = sizes.get(axis.value) or 1
                else:
                    chunk_shapes[axis] = 1

        dim_list = [
            Dimension(
                name=axis.value,
                count=sizes[axis.value],
                type=axis.dimension_type(),
                unit=axis.unit(),
                chunk_size=chunk_shapes[axis],
            )
            for axis in std_axes
        ]
        return cls(dimensions=dim_list, dtype=np.dtype(dtype))


@dataclass(slots=True, frozen=True)
class Plate:
    """Plate structure for OME metadata.

    This defines the plate geometry (rows/columns) for metadata generation.
    Acquisition order is determined by PositionDimension in ArraySettings,
    not by this class.

    Parameters
    ----------
    row_names
        Row labels (e.g., ["A", "B", "C", ...]).
    column_names
        Column labels (e.g., ["1", "2", "3", ...]).
    name
        Optional plate name for metadata.
    """

    row_names: list[str]
    column_names: list[str]
    name: str | None = None


@dataclass(slots=True, frozen=True)
class AcquisitionSettings:
    root_path: str
    arrays: list[ArraySettings] | None = None
    plate: Plate | None = None
    overwrite: bool = False
    backend: str = "auto"

    def __post_init__(self) -> None:
        if isinstance(self.arrays, ArraySettings):
            object.__setattr__(self, "arrays", [self.arrays])
        if (self.arrays is None) == (self.plate is None):
            raise ValueError("Either arrays or plate must be provided, but not both.")
