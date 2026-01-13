"""Pydantic-based schema for ome-writers."""

from __future__ import annotations

from enum import Enum
from typing import TYPE_CHECKING, Annotated, Any, Literal, cast

from pydantic import AfterValidator, BaseModel, ConfigDict, model_validator

if TYPE_CHECKING:
    from collections.abc import Mapping

DimensionType = Literal["space", "time", "channel", "other"]
StandardAxisKey = Literal["x", "y", "z", "c", "t", "p"]


class _BaseModel(BaseModel):
    """Base model with frozen config."""

    model_config = ConfigDict(
        validate_default=True,
        validate_assignment=True,
        extra="forbid",
    )


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
            return "second"
        return None


class Dimension(_BaseModel):
    """A single array dimension."""

    name: str
    count: int | None  # None for unlimited (first dimension only)
    chunk_size: int | None = None
    shard_size: int | None = None
    type: DimensionType | None = None
    unit: str | None = None
    scale: float | None = None
    translation: float | None = None


class Position(_BaseModel):
    """A single acquisition position."""

    name: str
    row: str | None = None
    column: str | None = None

    @model_validator(mode="before")
    @classmethod
    def _cast(cls, value: Any) -> Any:
        """Allow casting from string to Position."""
        if isinstance(value, str):
            return {"name": value}
        return value


class PositionDimension(_BaseModel):
    """Positions (meta-dimension) in acquisition order.

    Unlike Dimension, positions don't become an array axis—they become
    separate arrays/files (this is currently true for both OME-Zarr and OME-TIFF).
    The position of PositionDimension in the dimensions list determines when
    positions are visited during acquisition.
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


def _validate_dims_list(
    dims: list[Dimension | PositionDimension],
) -> list[Dimension | PositionDimension]:
    dim_names = [dim.name for dim in dims]
    if len(dim_names) != len(set(dim_names)):
        raise ValueError("Dimension names must be unique.")

    # no more than one unlimited dimension (count=None):
    unlimited_dims = [
        dim for dim in dims if isinstance(dim, Dimension) and dim.count is None
    ]
    if len(unlimited_dims) > 1:
        raise ValueError("Only one dimension may be unlimited (count=None).")

    # ensure at least 2 spatial dimensions at the end
    spatial_dims = [d for d in dims if isinstance(d, Dimension) and d.type == "space"]
    if len(spatial_dims) < 2 or dims[-2:] != spatial_dims[-2:]:
        raise ValueError("The last two dimensions must be spatial (Y and X).")
    return dims


class ArraySettings(_BaseModel):
    """Settings for a single array/image."""

    # PositionDimension can appear anywhere in the list to control acquisition order.
    # Its location determines when positions are visited relative to other dimensions.
    dimensions: Annotated[
        list[Dimension | PositionDimension], AfterValidator(_validate_dims_list)
    ]
    dtype: str
    compression: str | None = None
    position_name: str | None = None
    storage_order: Literal["acquisition", "ngff"] | list[str] = "acquisition"

    @property
    def shape(self) -> tuple[int | None, ...]:
        """Shape of the array (count for each dimension)."""
        return tuple(dim.count for dim in self.dimensions)


class Plate(_BaseModel):
    """Plate structure for OME metadata.

    This defines the plate geometry (rows/columns) for metadata generation.
    Acquisition order is determined by PositionDimension in ArraySettings,
    not by this class.
    """

    row_names: list[str]
    column_names: list[str]
    name: str | None = None


class AcquisitionSettings(_BaseModel):
    """Top-level acquisition settings."""

    root_path: str
    arrays: list[ArraySettings] | ArraySettings | None = None
    plate: Plate | None = None
    overwrite: bool = False
    backend: str = "auto"

    @model_validator(mode="after")
    def _validate_arrays_or_plate(self) -> AcquisitionSettings:
        """Ensure exactly one of arrays or plate is provided."""
        # Coerce single ArraySettings to list
        if isinstance(self.arrays, ArraySettings):
            object.__setattr__(self, "arrays", [self.arrays])

        if (self.arrays is None) == (self.plate is None):
            raise ValueError("Either arrays or plate must be provided, but not both.")
        return self


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------


def dims_from_standard_axes(
    sizes: Mapping[str, int | list[str | Position]],
    chunk_shapes: Mapping[str | StandardAxis, int] | None = None,
) -> list[Dimension | PositionDimension]:
    """Create dimensions from standard axis names.

    Standard axes are {'x', 'y', 'z', 'c', 't', 'p'}. Dimension types and units
    are inferred from these names. Chunk shapes default to 1 for non-XY dimensions.

    For positions ('p'), the value can be:
    - int: creates PositionDimension with names "0", "1", ...
    - list[str | Position]: creates PositionDimension with those names or Position
      objects

    Parameters
    ----------
    sizes
        Mapping of axis name to size. Order determines dimension order.
        For 'p', value can be int or list of position names.
    chunk_shapes
        Optional chunk sizes per axis. Defaults to full size for X/Y, 1 for others.

    Returns
    -------
    list[Dimension | PositionDimension]
        Dimensions in the order specified by sizes.

    Examples
    --------
    >>> dims = dims_from_standard_axes({"t": 10, "c": 2, "y": 512, "x": 512})
    >>> dims = dims_from_standard_axes({"t": 10, "p": ["A1", "B2"], "y": 512, "x": 512})
    """
    try:
        std_axes = [StandardAxis(axis) for axis in sizes]
    except ValueError as e:
        raise ValueError(
            f"All axes must be one of {StandardAxis._member_names_}."
        ) from e

    # Default chunk shapes: full size for X/Y, 1 for others
    x_or_y = {StandardAxis.X, StandardAxis.Y}
    chunk_shapes = dict(chunk_shapes) if chunk_shapes else {}
    for axis in std_axes:
        if axis not in chunk_shapes and axis != StandardAxis.POSITION:
            size = sizes.get(axis.value)
            chunk_shapes[axis] = size if isinstance(size, int) and axis in x_or_y else 1

    dims: list[Dimension | PositionDimension] = []
    for axis in std_axes:
        value = sizes[axis.value]
        if axis == StandardAxis.POSITION:
            if isinstance(value, list):
                positions = [Position.model_validate(n) for n in value]
            elif isinstance(value, int):
                positions = [Position(name=str(i)) for i in range(value)]
            else:
                raise ValueError(f"Invalid position value: {value}")
            dims.append(PositionDimension(positions=positions))
        else:
            dims.append(
                Dimension(
                    name=axis.value,
                    count=cast("int", value),
                    type=axis.dimension_type(),
                    unit=axis.unit(),
                    chunk_size=chunk_shapes[axis],
                )
            )
    return dims
