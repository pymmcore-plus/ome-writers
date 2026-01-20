"""Pydantic-based schema for ome-writers."""

from __future__ import annotations

import warnings
from enum import Enum
from typing import TYPE_CHECKING, Annotated, Any, Literal, TypeAlias, cast

import numpy as np
from annotated_types import Len
from pydantic import (
    AfterValidator,
    BaseModel,
    BeforeValidator,
    ConfigDict,
    Field,
    PositiveInt,
    model_validator,
)

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

BackendName: TypeAlias = Literal["acquire-zarr", "tensorstore", "zarr", "tiff"]
DimensionType: TypeAlias = Literal["space", "time", "channel", "other"]
StandardAxisKey: TypeAlias = Literal["x", "y", "z", "c", "t", "p"]


class _BaseModel(BaseModel):
    """Base model with frozen config."""

    model_config = ConfigDict(
        validate_default=True,
        validate_assignment=True,
        extra="forbid",
    )


def _validate_dtype(dtype: Any) -> str:
    """Validate dtype is a valid string."""
    try:
        return np.dtype(dtype).name
    except Exception as e:
        raise ValueError(f"Invalid dtype: {dtype!r}: {e}") from e


class StandardAxis(str, Enum):
    """Standard axis names.

    Dimension names are fundamentally arbitrary strings, but these standard
    names are commonly used and have well-defined meanings in terms of dimension
    type and units.  This is used in `dims_from_standard_axes` helper function.
    """

    X = "x"
    Y = "y"
    Z = "z"
    CHANNEL = "c"
    TIME = "t"
    POSITION = "p"

    def dimension_type(self) -> DimensionType:
        """Return dimension type for this standard axis."""
        if self in {StandardAxis.X, StandardAxis.Y, StandardAxis.Z}:
            return "space"
        if self == StandardAxis.TIME:
            return "time"
        if self == StandardAxis.CHANNEL:
            return "channel"
        return "other"  # pragma: no cover

    def unit(self) -> str | None:
        """Return unit for this standard axis, or None if unknown."""
        if self in {StandardAxis.X, StandardAxis.Y, StandardAxis.Z}:
            return "micrometer"
        if self == StandardAxis.TIME:
            return "second"
        return None

    def to_dimension(
        self,
        *,
        count: int | None = None,
        positions: list[str | Position] | None = None,
        chunk_size: int | None = None,
        shard_size: int | None = None,
        scale: float | None = None,
    ) -> Dimension | PositionDimension:
        """Convert to Dimension or PositionDimension with given count."""
        if self == StandardAxis.POSITION:
            if positions:
                positions = [Position.model_validate(n) for n in positions]
            elif count:
                if not isinstance(count, int):
                    raise ValueError(f"Invalid position value: {count}.")
                positions = [Position(name=str(i)) for i in range(count)]
            else:  # pragma: no cover
                raise ValueError(
                    "Either count or positions must be provided for PositionDimension."
                )
            return PositionDimension(positions=positions)

        return Dimension(
            name=self.value,
            count=count,
            type=self.dimension_type(),
            unit=self.unit(),
            chunk_size=chunk_size,
            shard_size=shard_size,
            scale=scale,
        )


class Dimension(_BaseModel):
    """A single array dimension."""

    name: Annotated[str, Len(min_length=1)] = Field(
        description="User-defined name. Can be anything, but prefer using standard "
        "names like 'x', 'y', 'z', 'c', 't' where possible. Must be unique across all "
        "dimensions in an acquisition.",
    )
    count: PositiveInt | None = Field(
        default=None,
        description="Size of this dimension (in number of elements/pixels)."
        "None indicates an unbounded (unlimited) 'append' dimension. "
        "Only the first dimension in may be unbounded.",
    )
    chunk_size: PositiveInt | None = Field(
        default=None,
        description="Number of elements in a chunk for this dimension, for storage "
        "backends that support chunking (e.g. Zarr). If None, defaults to full size "
        "(i.e. `count`) for the last two 'frame' dimensions, and 1 for others.",
    )
    shard_size: PositiveInt | None = Field(
        default=None,
        description="Number of chunks per shard, for storage backends that "
        "support sharding (e.g. Zarr v3). If not specified, no sharding is used.",
    )
    type: DimensionType | None = Field(
        default=None,
        description="Type of this dimension. Must be one of 'space', 'time', 'channel' "
        "or 'other'. If `None`, type _may_ be inferred from the name for standard "
        "names like 'x', 'y', 'z', 'c', 't'.",
    )
    unit: str | None = Field(
        default=None,
        description="Physical unit for this dimension, e.g. 'micrometer' for spatial "
        "dimensions or 'second' for time.  Prefer using [ome-ngff unit naming "
        "conventions](https://ngff.openmicroscopy.org/latest/index.html#axes-md)"
        " where possible.",
    )
    scale: float | None = Field(
        default=None,
        description="Physical size of a single element along this dimension, "
        "in the specified `unit`. For spatial dimensions, this is often referred to "
        "as 'pixel size'.  For time dimensions, this would be the time interval.",
    )
    translation: float | None = Field(
        default=None,
        description="Physical offset of the first element along this dimension, "
        "in the specified `unit`. (e.g. the physical coordinate of the first pixel "
        "or timepoint, in some XYZ stage or other coordinate system).",
    )


class Position(_BaseModel):
    """A single acquisition position."""

    name: Annotated[str, Len(min_length=1)] = Field(
        description="Unique name for this position. Within a list of positions, "
        "names must be unique within each `(row, column)` pair.",
    )
    row: str | None = Field(
        default=None,
        description="Row name for plate position.",
    )
    column: str | None = Field(
        default=None,
        description="Column name for plate position.",
    )
    # TODO
    # These could be used to specify the coordinateTransform.translate for
    # different positions
    # x_translation: float | None = None
    # y_translation: float | None = None

    @model_validator(mode="before")
    @classmethod
    def _validate_position(cls, value: Any) -> Any:
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

    positions: list[Position] = Field(
        description="List of positions in acquisition order.  String literals "
        "are also accepted and will be converted to Position objects with the "
        "given name.",
    )
    name: Annotated[str, Len(min_length=1)] = Field(
        default="p",
        description="Name of this position dimension. Default is 'p'.",
    )

    @property
    def count(self) -> int:
        """Number of positions in the list."""
        return len(self.positions)

    @property
    def names(self) -> list[str]:
        """Position names in acquisition order."""
        return [p.name for p in self.positions]

    @property
    def unit(self) -> None:
        """Unit is always None for PositionDimension.

        Provided for symmetry with Dimension.
        """
        return None

    @property
    def scale(self) -> Literal[1]:
        """Scale is always None for PositionDimension.

        Provided for symmetry with Dimension.
        """
        return 1


def _validate_dims_list(
    dims: tuple[Dimension | PositionDimension, ...],
) -> tuple[Dimension | PositionDimension, ...]:
    """Validate dimensions list for AcquisitionSettings."""
    # ensure at most one PositionDimension and 2-5 non-position dimensions.
    # ensure unique position names within each well (row/column combination)
    has_pos = False
    n_dims = 0
    name_counts: dict[str, int] = {}
    for idx, dim in enumerate(dims):
        name_counts.setdefault(dim.name, 0)
        name_counts[dim.name] += 1
        if isinstance(dim, PositionDimension):
            if has_pos:
                raise ValueError("Only one PositionDimension is allowed.")
            _validate_unique_names_per_well(dim.positions)
            has_pos = True
        else:
            n_dims += 1
            # only the first dimension can be unbounded
            if dim.count is None and idx != 0:
                raise ValueError(
                    "Only the first dimension may be unbounded (count=None)."
                )

    if n_dims < 2:
        raise ValueError(
            "At least 2 non-position dimensions are required (usually Y and X)."
        )
    if n_dims > 5:
        raise ValueError(
            "At most 5 non-position dimensions are allowed for both OME-Zarr and "
            f"OME-TIFF, got {n_dims}."
        )

    # ensure all dimension names are unique
    if dupe_names := [name for name, count in name_counts.items() if count > 1]:
        raise ValueError(f"Dimension names must be unique, duplicates: {dupe_names}")

    # ensure at least 2 spatial dimensions at the end
    for dim in dims[-2:]:
        if not isinstance(dim, Dimension) or dim.type not in {"space", None}:
            raise ValueError(
                "The last two dimensions must be spatial dimensions (type='space')."
            )
        if dim.type is None:
            if dim.name.lower() not in {"x", "y"}:
                warnings.warn(
                    "The last two dimensions are expected to have type='space'. Setting"
                    f"type='space' for non-standard dimension name '{dim.name}'.",
                    stacklevel=2,
                )
            dim.type = "space"

    return tuple(dims)


def _validate_unique_names_per_well(positions: list[Position]) -> None:
    """Validate position names are unique within each well.

    For positions with row/column defined, names must be unique within each
    (row, column) group. This allows the same name across different wells,
    but not multiple positions with the same name in the same well.
    """
    # Group positions by (row, column) - only for positions with both defined
    wells = {}
    for pos in positions:
        key = (pos.row, pos.column)
        wells.setdefault(key, []).append(pos.name)

    # Check for duplicates within each well
    for (row, col), names in wells.items():
        if len(names) != len(set(names)):
            seen: set[str] = set()
            duplicates = [n for n in names if n in seen or seen.add(n)]
            if row is None and col is None:
                raise ValueError(
                    "All positions without row/column must have unique names."
                )
            else:
                raise ValueError(
                    f"Position names must be unique within each well. "
                    f"Well ({row}, {col}) has duplicate names: {duplicates}"
                )


DimensionsList: TypeAlias = Annotated[
    tuple[Dimension | PositionDimension, ...], AfterValidator(_validate_dims_list)
]
"""Assembled list of Dimensions and PositionDimensions."""


class Plate(_BaseModel):
    """Plate structure for OME metadata.

    This defines the plate geometry (rows/columns) for metadata generation.
    Acquisition order is determined by `PositionDimension` in `AcquisitionSettings`,
    not by this class.
    """

    row_names: list[str] = Field(
        description="List of *all* row names in the plate, e.g. "
        "`['A', 'B', 'C', ...]`. This is used to indicate the full plate structure in "
        "OME metadata, even if not all wells are acquired.",
    )
    column_names: list[str] = Field(
        description="List of *all* column names in the plate, e.g. "
        "`['1', '2', '3', ...]`. This is used to indicate the full plate structure in "
        "OME metadata, even if not all wells are acquired.",
    )
    name: str | None = Field(
        default=None,
        description="Optional name for the plate.",
    )


class AcquisitionSettings(_BaseModel):
    """Top-level acquisition settings.

    This is the main schema object for defining an acquisition to be written
    using ome-writers.  It includes the output path, dimensions, data type,
    compression, storage order, plate structure, and backend selection.

    Pass this object to [ome_writers.create_stream][] to create a data stream
    for writing acquisition data.

    !!! note
        This is a frozen model.  Use `.model_copy(update={...})` to create modified
        copies.
    """

    model_config = ConfigDict(frozen=True, validate_default=True)

    root_path: Annotated[str, BeforeValidator(str)] = Field(
        description="Root output path for the acquisition data.  This may be a "
        "directory (for OME-Zarr) or a file path (for OME-TIFF). It is customary "
        "to use an `.ome.zarr` extension for OME-Zarr directories and `.ome.tiff` "
        "for OME-TIFF files.",
    )
    dimensions: DimensionsList = Field(
        description="List of dimensions in order of acquisition. Must include at least "
        "two spatial 'in-frame' dimensions (usually Y and X) at the end. May not "
        "include more than 5 non-position dimensions total. May include one "
        "`PositionDimension` to specify multiple acquisition positions. Only the first "
        "dimension may be unbounded (count=None).",
    )
    dtype: Annotated[str, BeforeValidator(_validate_dtype)] = Field(
        description="Data type of the pixel data to be written, e.g. 'uint8', "
        "'uint16', 'float32', etc. Must be a valid numpy DTypeLike string.",
    )
    compression: str | None = None
    # "ome" means "spec-compliant" storage order.
    # It MAY depend on the output format (e.g. OME-Zarr vs OME-TIFF) and
    # version, and backends may have different restrictions.
    storage_order: Literal["acquisition", "ome"] | list[str] = Field(
        default="ome",
        description="Storage order for non-frame dimensions (if different from "
        "acquisition order).  May be 'acquisition' (same as acquisition order), "
        "a list of dimension names to specify a custom storage order. Or 'ome' to "
        "use a format-specific OME-compliant canonical order (e.g. TCZYX for OME-Zarr "
        "v0.5), or any [TCZ]YX variation for OME-TIFF. The last two 'frame' dimensions "
        "(usually Y and X) may not be reordered. Default is 'ome'.",
    )
    plate: Plate | None = Field(
        default=None,
        description="Plate structure for OME metadata. If specified, requires a "
        "`PositionDimension` in `dimensions`, and all positions must have "
        "row/column defined. Presence of this field indicates plate mode.",
    )
    overwrite: bool = Field(
        default=False,
        description="Whether to overwrite existing data at `root_path`. If False "
        "and data already exists at the path, an error will be raised when "
        "creating the stream.",
    )
    backend: BackendName | Literal["auto"] = Field(
        default="auto",
        description="Storage backend to use for writing data.  May be one of "
        "'acquire-zarr', 'tensorstore', 'zarr', or 'tiff'.  If 'auto', the backend "
        "will be chosen based on the `root_path` extension and available dependencies. "
        "Default is 'auto'.",
    )

    @property
    def format(self) -> Literal["tiff", "zarr"]:
        """Inferred file format.  Either (OME) 'tiff' or 'zarr'."""
        if self.backend == "tiff" or (
            self.backend == "auto"
            and self.root_path.lower().endswith((".tiff", ".tif"))
        ):
            return "tiff"
        return "zarr"

    @property
    def shape(self) -> tuple[int | None, ...]:
        """Shape of the array (count for each dimension)."""
        return tuple(dim.count for dim in self.dimensions)

    @property
    def is_unbounded(self) -> bool:
        """Whether the acquisition has an unbounded (None) dimension."""
        return self.dimensions[0].count is None

    @property
    def num_frames(self) -> int | None:
        """Return total number of frames, or None if unlimited dimension present."""
        _non_frame_sizes = tuple(d.count for d in self.dimensions[:-2])
        total = 1
        for size in _non_frame_sizes:
            if size is None:
                return None
            total *= size
        return total

    @property
    def positions(self) -> tuple[Position, ...]:
        """Position objects in acquisition order."""
        for dim in self.dimensions[:-2]:  # last 2 dims may never be positions
            if isinstance(dim, PositionDimension):
                return tuple(dim.positions)
        return (Position(name="0"),)  # single default position

    @property
    def position_dimension_index(self) -> int | None:
        """Index of PositionDimension in dimensions, or None if not present."""
        for i, dim in enumerate(self.dimensions[:-2]):
            if isinstance(dim, PositionDimension):
                return i
        return None

    @property
    def frame_dimensions(self) -> tuple[Dimension, ...]:
        """In-frame dimensions, currently always last two dims (usually (Y,X))."""
        return cast("tuple[Dimension, ...]", self.dimensions[-2:])

    @property
    def index_dimensions(self) -> tuple[Dimension, ...]:
        """All NON-frame Dimensions, excluding PositionDimension dimensions."""
        return tuple(dim for dim in self.dimensions[:-2] if isinstance(dim, Dimension))

    @property
    def array_dimensions(self) -> tuple[Dimension, ...]:
        """All Dimensions excluding PositionDimension dimensions."""
        return tuple(
            dim for dim in self.dimensions if not isinstance(dim, PositionDimension)
        )

    @property
    def storage_index_dimensions(self) -> tuple[Dimension, ...]:
        """NON-frame Dimensions in storage order."""
        return _sort_dims_to_storage_order(
            self.index_dimensions, self.storage_order, self.format
        )

    @property
    def storage_index_permutation(self) -> tuple[int, ...] | None:
        """Permutation to convert acquisition index to storage index, if different."""
        storage_dims = self.storage_index_dimensions
        perm = _compute_permutation(self.index_dimensions, storage_dims)
        return perm if perm != tuple(range(len(perm))) else None

    @property
    def array_storage_dimensions(self) -> tuple[Dimension, ...]:
        """All Dimensions (excluding PositionDimension) in storage order."""
        return self.storage_index_dimensions + self.frame_dimensions

    # --------- Validators ---------

    @model_validator(mode="after")
    def _validate_storage_order(self) -> AcquisitionSettings:
        """Validate storage_order value."""
        if isinstance(self.storage_order, list):
            # manual sort orders are still not allowed to permute the last 2 dims
            if set(self.storage_order[-2:]) != {
                self.dimensions[-2].name,
                self.dimensions[-1].name,
            }:
                raise ValueError(
                    "storage_order may not (yet) permute the last two dimensions."
                )
            dim_names = {dim.name for dim in self.dimensions}
            if set(self.storage_order) != set(dim_names):
                raise ValueError(
                    f"storage_order names {self.storage_order!r} don't match "
                    f"acquisition dimension names {list(dim_names)}"
                )

        return self

    @model_validator(mode="after")
    def _validate_plate_positions(self) -> AcquisitionSettings:
        """Validate plate mode requirements."""
        if self.plate is not None:
            # Ensure there is a PositionDimension
            # and that all positions have row/column assigned
            for dim in self.dimensions:
                if isinstance(dim, PositionDimension):
                    missing = [
                        pos.name
                        for pos in dim.positions
                        if pos.row is None or pos.column is None
                    ]
                    if missing:
                        raise ValueError(
                            f"All positions must have row and column for plate mode. "
                            f"Missing row/column for positions: {missing}"
                        )
                    break
            else:
                raise ValueError(
                    "Plate mode requires a PositionDimension in dimensions."
                )

        return self


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------


def dims_from_standard_axes(
    sizes: Mapping[str, int | Sequence[str | Position] | None],
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
        if axis == StandardAxis.POSITION and isinstance(value, list):
            kwargs = {"positions": value}
        else:
            kwargs = {"count": value}
        dims.append(axis.to_dimension(chunk_size=chunk_shapes.get(axis), **kwargs))
    return dims


def _ngff_sort_key(dim: Dimension) -> tuple[int, int]:
    """Sort key for NGFF canonical order: time, channel, space (z, y, x)."""
    if dim.type == "time":
        return (0, 0)
    if dim.type == "channel":
        return (1, 0)
    if dim.type == "space":
        return (2, {"z": 0, "y": 1, "x": 2}.get(dim.name, -1))
    return (3, 0)


def _ome_tiff_sort_key(dim: Dimension) -> tuple[int, int]:
    """Sort key for OME-TIFF canonical order: [TCZ]YX variations."""
    # FIXME: ... if even possible
    # here we have a problem... because OME-TIFF didn't have the concept
    # of dimension types, it uses strict "XYXCZT" naming conventions.
    # so if the user has non-standard names, we can't really map to OME-TIFF
    # storage order correctly.
    # however... OME-Tiff storage order is more flexible in practice (within
    # the strict 5-dim limit).  So for now we just return the acquisition order
    # and ... pray?
    order_priority = {"t": 1, "c": 1, "z": 1, "y": 3, "x": 4}
    return (order_priority.get(dim.name.lower(), 5), 0)


def _sort_dims_to_storage_order(
    index_dims: list[Dimension],
    storage_order: str | list[str],
    format: Literal["tiff", "zarr"],
) -> tuple[Dimension, ...]:
    """Resolve storage_order setting to explicit list of dimension names.

    NOTE: this function is the only place in the schema that cares about format.
    While it seems a little like a backend concern, putting it here lets us have
    immediate validation of Zarr/Tiff-specific storage order rules.  Additionally,
    the "rules" are relatively constant across backends.
    """
    if storage_order == "acquisition":
        return tuple(index_dims)
    elif storage_order == "ome":
        if format == "zarr":
            return tuple(sorted(index_dims, key=_ngff_sort_key))
        else:
            return tuple(sorted(index_dims, key=_ome_tiff_sort_key))
    elif isinstance(storage_order, list):
        dims_map = {dim.name: dim for dim in index_dims}
        return tuple(dims_map[name] for name in storage_order if name in dims_map)
    else:
        raise ValueError(  # pragma: no cover (unreachable due to prior validation)
            f"Invalid storage_order: {storage_order!r}. Must be 'acquisition', 'ome', "
            "or list of names."
        )


def _compute_permutation(
    acq_dims: list[Dimension], storage_names: list[Dimension]
) -> tuple[int, ...]:
    """Compute permutation to convert acquisition indices to storage indices."""
    dim_names = [dim.name for dim in acq_dims]
    return tuple(dim_names.index(dim.name) for dim in storage_names)
