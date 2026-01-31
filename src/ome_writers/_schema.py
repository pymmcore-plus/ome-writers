"""Pydantic-based schema for ome-writers."""

from __future__ import annotations

import warnings
from collections import Counter, defaultdict
from enum import Enum
from typing import (
    TYPE_CHECKING,
    Annotated,
    Any,
    Literal,
    TypeAlias,
    cast,
    get_args,
)

import numpy as np
from annotated_types import MinLen
from pydantic import (
    AfterValidator,
    BaseModel,
    BeforeValidator,
    ConfigDict,
    Field,
    PositiveInt,
    model_validator,
)
from pydantic_extra_types.color import Color  # noqa: TC002  (used by pydantic)

from ome_writers._memory import warn_if_high_memory_usage
from ome_writers._stream import AVAILABLE_BACKENDS
from ome_writers._units import cast_unit_to_ngff

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

FileFormat: TypeAlias = Literal["ome-tiff", "ome-zarr"]
TiffBackendName: TypeAlias = Literal["tifffile"]
ZarrBackendName: TypeAlias = Literal[
    "acquire-zarr", "tensorstore", "zarrs-python", "zarr-python"
]
BackendName: TypeAlias = TiffBackendName | ZarrBackendName
DimensionType: TypeAlias = Literal["space", "time", "channel", "other"]
StandardAxisKey: TypeAlias = Literal["x", "y", "z", "c", "t", "p"]


class _BaseModel(BaseModel):
    """Base model with common configuration."""

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
        shard_size_chunks: int | None = None,
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
            shard_size_chunks=shard_size_chunks,
            scale=scale,
        )


class Channel(_BaseModel):
    """A single channel in an acquisition.

    This object may be used (instead of a plain string) in the
    [`Dimension.coords`][ome_writers.Dimension] list
    for dimensions of `type='channel'` when you want to specify more than just channel
    name.

    This metadata is propagated to `OME.Image.Pixels.Channel` for OME-XML, and
    to `omero.channels` for OME-Zarr.
    """

    name: Annotated[str, MinLen(1)] = Field(
        description="A name for the channel, suitable for presentation to the user.",
    )

    excitation_wavelength_nm: PositiveInt | None = Field(
        default=None,
        description="Excitation wavelength in nanometers.",
    )
    emission_wavelength_nm: PositiveInt | None = Field(
        default=None,
        description="Emission wavelength in nanometers.",
    )
    fluorophore: str | None = Field(
        default=None,
        description="Name of the fluorophore used in this channel. No validation done, "
        "prefer names from [FPbase](https://www.fpbase.org/) where possible.",
    )
    color: Color | None = Field(
        default=None,
        description="A color recommended when rendering this channel. Input may be a "
        "string, or a 3-4 tuple of RGB(A) values (0-255).",
    )

    @model_validator(mode="before")
    @classmethod
    def _cast_values(cls, data: Any) -> Any:
        """Cast input value to Channel dict."""
        if isinstance(data, str):
            return {"name": data}
        return data

    # In case you're looking for more fields...
    # OME-XML supports a lot more detail about channels, detectors, lights, filters, etc
    # https://ome-types.readthedocs.io/en/latest/API/ome_types.model/#ome_types._autogenerated.ome_2016_06.Channel
    # OME-Zarr NGFF currently has limited channel metadata support, via the "omero" key
    # it leans more towards display suggestions over acquisition settings.
    # See `yaozarrs.v05.Omero`
    #
    # open an issue to request more channel fields if needed.
    # (one can always use `stream.get/update_metadata()` to manually update metadata)


class Dimension(_BaseModel):
    """A single array dimension."""

    name: Annotated[str, MinLen(1)] = Field(
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
    shard_size_chunks: PositiveInt | None = Field(
        default=None,
        description="Number of chunks per shard (*NOT* number of pixels per shard), "
        "for storage backends that support sharding (e.g. Zarr v3). If not specified, "
        "no sharding is used (i.e. chunks are the unit of storage).",
    )
    type: DimensionType | None = Field(
        default=None,
        description="Type of this dimension. Must be one of `'space'`, `'time'`, "
        "`'channel'` or `'other'`. If `None`, type _may_ be inferred from the provided "
        "`unit`, or from standard dimension `names` like `'x'`, `'y'`, `'z'`, `'c'`, "
        "`'t'`.",
    )
    unit: str | None = Field(
        default=None,
        description="Physical unit for this dimension. "
        "If `type` is `'space'` or `'time'`, this MUST be a valid unit of length or "
        "time. Both [OME-NGFF unit "
        "names](https://ngff.openmicroscopy.org/latest/index.html#axes-md) (_e.g._, "
        "`'micrometer'`, `'millisecond'`), and [OME-XML "
        "abbreviations](https://www.openmicroscopy.org/Schemas/Documentation/Generated/OME-2016-06/ome_xsd.html#UnitsLength)"
        " (_e.g._, `'um'`, `'ms'`) are accepted. "
        "If `type` is `None`, then it will be inferred from the unit if a recognized "
        "unit is provided.",
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
    coords: list[str | float | Channel] | None = Field(
        default=None,
        description="Explicit coordinate values for each element along this dimension."
        "This is primarily a convenience for specifying categorical coordinates, such "
        "as channel names, and may also be used to explicitly list non-uniform spatial "
        "or temporal coordinates. If provided, the length of this list must match "
        "the `count` of this dimension. If `count` is `None`, the length of this list "
        "will be used as the count. Item types may additionally be validated based on "
        "the dimension `type` (e.g. coords may be a list of `Channel` objects, but "
        "only if the dimension type is 'channel').",
    )

    @model_validator(mode="before")
    @classmethod
    def _validate_model(cls, data: Any) -> Any:
        """Validate that unit is NGFF-compliant if specified.

        Ensure that unit matches dim type, and infer dim type from unit if missing.

        After validation, Dimensions with type "space" or "time" are guaranteed to
        to have valid NGFF units.  Those with type "channel", "other", or None may
        still have arbitrary units.
        """
        if isinstance(data, dict):
            if (unit := data.get("unit")) is not None:
                cast_unit, dim_type = cast_unit_to_ngff(unit, data.get("type"))
                data["unit"] = cast_unit
                data["type"] = dim_type

            if (coords := data.get("coords")) is not None:
                count = data.get("count")
                try:
                    coords = list(coords)
                except TypeError:
                    raise ValueError("`coords` must be an iterable type.") from None
                if count is None:
                    data["count"] = len(coords)
                elif len(coords) != count:
                    raise ValueError(
                        f"Length of coords ({len(coords)}) does not match count "
                        f"({count})."
                    )

                # validate item type by dimension type
                dim_type = data.get("type")
                if dim_type == "channel":
                    data["coords"] = [Channel.model_validate(c) for c in coords]

        return data


class Position(_BaseModel):
    """A single acquisition position.

    This represents a physical position in space associated with a single camera frame
    or field of view.  Optional fields such as `grid/plate_row/column` indicate that the
    position is a member of a larger coordinate system (e.g. well plate or grid).  The
    `x_coord`, `y_coord`, and `z_coord` fields represent physical coordinates.  Units
    should match those used in the spatial Dimensions of the acquisition.
    """

    name: Annotated[str, MinLen(1)] = Field(
        description="Unique name for this position. Within a list of positions, "
        "names must be unique within each `(plate_row, plate_column)` or "
        "`(grid_row, grid_column)` pair.",
    )
    plate_row: str | None = Field(
        default=None,
        description="Row name for plate position.",
    )
    plate_column: str | None = Field(
        default=None,
        description="Column name for plate position.",
    )
    grid_row: int | None = Field(
        default=None,
        description="Row index for this position in a grid layout (if any).",
    )
    grid_column: int | None = Field(
        default=None,
        description="Column index for this position in a grid layout (if any).",
    )
    x_coord: float | None = Field(
        default=None,
        description="Physical X coordinate of this position (e.g., stage coordinate).",
    )
    y_coord: float | None = Field(
        default=None,
        description="Physical Y coordinate of this position (e.g., stage coordinate).",
    )
    z_coord: float | None = Field(
        default=None,
        description="Physical Z coordinate of this position (e.g., stage coordinate).",
    )

    @model_validator(mode="before")
    @classmethod
    def _validate_position(cls, value: Any) -> Any:
        """Allow casting from string to Position."""
        if isinstance(value, str):
            return {"name": value}
        return value


def _validate_unique_names_per_group(positions: list[Position]) -> list[Position]:
    """Validate position names are unique within each hierarchical group.

    For positions with plate coordinates or grid coordinates, names must be unique
    within each group. This allows the same name across different wells/cells,
    but not multiple positions with the same name in the same group.

    Positions without any hierarchical coordinates (plate or grid) must have
    globally unique names.
    """
    # Group positions by their hierarchical coordinates
    counters: dict[tuple, Counter] = defaultdict(Counter)
    for pos in positions:
        key = (pos.plate_row, pos.plate_column, pos.grid_row, pos.grid_column)
        counters[key][pos.name] += 1

    # Check for duplicates within each group
    for key, counts in counters.items():
        duplicates = [name for name, count in counts.items() if count > 1]

        if duplicates:
            # All values None means no hierarchical structure
            if all(v is None for v in key):
                raise ValueError(
                    "All positions without row/column must have unique names."
                )
            else:
                # Format error based on coordinate type
                plate_row, plate_column, grid_row, grid_column = key
                parts = []
                if plate_row is not None and plate_column is not None:
                    parts.append(f'well ("{plate_row}", "{plate_column}")')
                if grid_row is not None and grid_column is not None:
                    parts.append(f"grid position ({grid_row}, {grid_column})")

                group_desc = ", ".join(parts) if parts else f"group {key}"
                raise ValueError(
                    f"Position names must be unique within each group. "
                    f"Duplicate names in {group_desc}: {duplicates}"
                )
    return positions


PositionList: TypeAlias = Annotated[
    list[Position], AfterValidator(_validate_unique_names_per_group)
]


class PositionDimension(_BaseModel):
    """Positions (meta-dimension) in acquisition order.

    Unlike Dimension, positions don't become an array axis—they become
    separate arrays/files (this is currently true for both OME-Zarr and OME-TIFF).
    The position of PositionDimension in the dimensions list determines when
    positions are visited during acquisition.
    """

    positions: PositionList = Field(
        description="List of positions in acquisition order.  String literals "
        "are also accepted and will be converted to Position objects with the "
        "given name.",
    )
    name: Annotated[str, MinLen(1)] = Field(
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


TiffCompression: TypeAlias = Literal["lzw", "none"]
ZarrCompression: TypeAlias = Literal["blosc-zstd", "blosc-lz4", "zstd", "none"]
Compression: TypeAlias = Literal["blosc-zstd", "blosc-lz4", "zstd", "lzw", "none"]


class OmeTiffFormat(_BaseModel):
    """Settings specific to OME-TIFF format."""

    name: Literal["ome-tiff"] = Field(
        default="ome-tiff",
        description="File format identifier for OME-TIFF.",
    )
    backend: TiffBackendName | Literal["auto"] = Field(
        default="auto",
        description="Storage backend to use for writing data.  Must be one of 'auto', "
        "or 'tifffile'. If 'auto' (the default), the backend will be chosen based on "
        "available dependencies. Currently, 'tifffile' is the only supported backend.",
    )
    suffix: str = Field(
        default=".ome.tiff",
        description="File suffix/extension to use for OME-TIFF files. Default is "
        "'.ome.tiff'.",
    )


class OmeZarrFormat(_BaseModel):
    """Settings specific to OME-Zarr format."""

    name: Literal["ome-zarr"] = Field(
        default="ome-zarr",
        description="File format identifier for OME-Zarr.",
    )
    backend: ZarrBackendName | Literal["auto"] = Field(
        default="auto",
        description="Storage backend to use for writing data.  Must be one of 'auto', "
        "'tensorstore', 'acquire-zarr', 'zarrs-python', or 'zarr-python'. "
        "If 'auto' (the default), the backend will be chosen based on the "
        "available dependencies, in the order: "
        "tensorstore, acquire-zarr, zarrs-python, zarr-python.",
    )
    suffix: str = Field(
        default=".ome.zarr",
        description="Directory suffix/extension to use for OME-Zarr directories. "
        "Default is '.ome.zarr'.",
    )


def _cast_format(value: Any) -> Any:
    # _backend_str possibly passed from _pick_auto_format
    suffix = ""
    if isinstance(value, dict) and (backend_str := value.pop("_backend_str", None)):
        suffix = value.get("suffix", "")
        value = backend_str  # Fall through to string handling below

    if isinstance(value, str):
        kwargs = {"suffix": suffix} if suffix else {}
        match value.lower():
            case "ome-tiff" | "tiff":
                return OmeTiffFormat(**kwargs)
            case "ome-zarr" | "zarr":
                return OmeZarrFormat(**kwargs)
            case "tensorstore":
                return OmeZarrFormat(backend="tensorstore", **kwargs)
            case "acquire-zarr":
                return OmeZarrFormat(backend="acquire-zarr", **kwargs)
            case "zarrs-python":
                return OmeZarrFormat(backend="zarrs-python", **kwargs)
            case "zarr-python":
                return OmeZarrFormat(backend="zarr-python", **kwargs)
            case "tifffile":
                return OmeTiffFormat(backend="tifffile", **kwargs)

    return value


Format: TypeAlias = Annotated[
    OmeTiffFormat | OmeZarrFormat, BeforeValidator(_cast_format)
]


class AcquisitionSettings(_BaseModel):
    """Top-level acquisition settings.

    This is the main schema object for defining an acquisition to be written
    using ome-writers.  It includes the output path, dimensions, data type,
    compression, storage order, plate structure, and backend selection.

    Pass this object to [ome_writers.create_stream][] to create a data stream
    for writing acquisition data.
    """

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
    format: Format = Field(  # type: ignore
        default="auto",
        description="Desired output format/backend. Can be a simple string: 'ome-tiff' "
        "or 'ome-zarr', in which case the first available format-appropriate backend "
        "will be used; Or it may be a full format specification dict/object "
        "([`ome_writers.OmeTiff`][] or [`ome_writers.OmeZarr`][]), to configure "
        "format-specific options such as backend selection.",
    )
    compression: Compression | None = Field(
        default=None,
        description="Compression algorithm for the storage backend. "
        "Zarr backends support: 'blosc-zstd', 'blosc-lz4', 'zstd', 'none'. "
        "TIFF backend supports: 'lzw', 'none'. "
        "If None, no compression is applied.",
    )
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

    @property
    def output_path(self) -> str:
        """Output path for the acquisition data.

        This is the `root_path` provided by the user, resolved by the format, possibly
        with appropriate suffix/extension added.
        """
        ome_stem, _ = _ome_stem_suffix(self.root_path)
        return ome_stem + self.format.suffix

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
            self.index_dimensions, self.storage_order, self.format.name
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
    def _validate_format_compression(self) -> AcquisitionSettings:
        """Validate compression is supported for selected format."""
        if self.compression is None:
            return self
        # TODO: move this to Format classes?
        if self.format.name == "ome-tiff":
            tiff_args = get_args(TiffCompression)
            if self.compression not in tiff_args:  # pragma: no cover
                raise ValueError(
                    f"Compression '{self.compression}' is not supported for OME-TIFF. "
                    f"Supported: {tiff_args}."
                )
        elif self.format.name == "ome-zarr":
            zarr_args = get_args(ZarrCompression)
            if self.compression not in zarr_args:  # pragma: no cover
                raise ValueError(
                    f"Compression '{self.compression}' is not supported for OME-Zarr. "
                    f"Supported: {zarr_args}."
                )

        return self

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
                    names_without_row_col = [
                        pos.name
                        for pos in dim.positions
                        if pos.plate_row is None or pos.plate_column is None
                    ]
                    if names_without_row_col:
                        raise ValueError(
                            f"All positions must have row and column for plate mode. "
                            f"Missing row/column for positions: {names_without_row_col}"
                        )

                    names_with_bad_coords = [
                        pos.name
                        for pos in dim.positions
                        if ((r := pos.plate_row) and r not in self.plate.row_names)
                        or (
                            (c := pos.plate_column) and c not in self.plate.column_names
                        )
                    ]
                    if names_with_bad_coords:
                        warnings.warn(
                            f"Some positions have row/column values not in the plate "
                            f"definition: {names_with_bad_coords}. "
                            "These may be omitted from the plate metadata on disk.",
                            stacklevel=3,
                        )

                    break
            else:
                raise ValueError(
                    "Plate mode requires a PositionDimension in dimensions."
                )

        return self

    @model_validator(mode="after")
    def _warn_chunk_buffer_memory(self) -> AcquisitionSettings:
        """Warn if chunk buffering may use excessive memory (Windows only)."""
        warn_if_high_memory_usage(self)
        return self

    @model_validator(mode="before")
    @classmethod
    def _pick_auto_format(cls, data: Any) -> Any:
        """If format is 'auto', pick first available format/backend."""
        if isinstance(data, dict):
            if "backend" in data:  # pragma: no cover
                raise ValueError(
                    "`backend` is no longer a top-level field in AcquisitionSettings. "
                    "Please specify backend within the `format` field, or pass the "
                    "backend name directly as the `format` value."
                )

            root = data.get("root_path", "")
            _stem, suffix = _ome_stem_suffix(root)
            fmt = data.get("format", "auto")
            if isinstance(fmt, dict):
                fmt.setdefault("suffix", suffix)
            elif fmt == "auto":
                # suffix-based inference
                if suffix.endswith((".tiff", ".tif")):
                    data["format"] = {"name": "ome-tiff", "suffix": suffix}
                elif suffix.endswith(".zarr"):
                    data["format"] = {"name": "ome-zarr", "suffix": suffix}
                else:  # pick first available backend
                    backend = next(iter(AVAILABLE_BACKENDS.values()))
                    warnings.warn(
                        f"\n\nOutput format could not be inferred from root_path "
                        f"{root!r}. \nPicking the first available format/backend: "
                        f"{backend.format!r}/{backend.name!r}. "
                        "\nThis may not be what you want, and may be an error in "
                        "future versions.\n"
                        "Please specify the desired format explicitly (e.g. "
                        "format='ome-zarr') or via the extension of `root_path`.\n",
                        stacklevel=3,
                    )
                    data["format"] = backend.name
            elif isinstance(fmt, str):
                # Format is a string like "zarr-python", "tifffile", etc.
                # Pass the suffix from root_path so _cast_format can use it
                data["format"] = {"_backend_str": fmt, "suffix": suffix}
        return data


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------


def dims_from_standard_axes(
    sizes: Mapping[str, int | Sequence[str | Position] | None],
    chunk_shapes: Mapping[str | StandardAxis, int] | None = None,
    shard_shapes: Mapping[str | StandardAxis, int] | None = None,
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
        Optional mapping of axis_name to chunk size. Defaults to full size for X/Y,
        1 for others.
    shard_shapes
        Optional mapping of axis_name to shard size (in number of chunks). Defaults to
        None (no sharding).

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
        allowed = [x.value for x in StandardAxis]
        raise ValueError(
            f"Standard axes names must be one of {allowed}. Got: {list(sizes)}"
        ) from e

    # Default chunk shapes: full size for X/Y, 1 for others
    x_or_y = {StandardAxis.X, StandardAxis.Y}
    chunk_shapes = dict(chunk_shapes) if chunk_shapes else {}
    for axis in std_axes:
        if axis not in chunk_shapes and axis != StandardAxis.POSITION:
            size = sizes.get(axis.value)
            chunk_shapes[axis] = size if isinstance(size, int) and axis in x_or_y else 1

    shard_shapes = dict(shard_shapes) if shard_shapes else {}
    dims: list[Dimension | PositionDimension] = []
    for axis in std_axes:
        value = sizes[axis.value]
        if axis == StandardAxis.POSITION and isinstance(value, list):
            kwargs = {"positions": value}
        else:
            kwargs = {"count": value}
        dims.append(
            axis.to_dimension(
                chunk_size=chunk_shapes.get(axis),
                shard_size_chunks=shard_shapes.get(axis),
                **kwargs,
            )
        )
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
    format: FileFormat,
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
        if format == "ome-zarr":
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


def _ome_stem_suffix(path: str) -> tuple[str, str]:
    """Return the stem of an OME file, removing .ome and image container suffixes.

    Examples
    --------
    >>> _ome_stem_suffix("data/image.tiff")
    ('data/image', '.tiff')
    >>> _ome_stem_suffix("data/image.ome.tiff")
    ('data/image', '.ome.tiff')

    # other periods are preserved
    >>> _ome_stem_suffix("data/image.test.ome.tiff")
    ('data/image.test', '.ome.tiff')

    # only *trailing* [.ome].ext is removed
    >>> _ome_stem_suffix("data/image.ome.test.zarr")
    ('data/image.ome.test', '.zarr')
    >>> _ome_stem_suffix("data/image.ome.test")
    ('data/image.ome.test', '')
    """
    str_path = str(path)
    lower = str_path.lower()
    for ext in (".tiff", ".tif", ".zarr"):
        if lower.endswith(ext):
            result = str_path[: -len(ext)]
            if result.lower().endswith(".ome"):
                return result[:-4], f".ome{ext}"
            return result, ext
    return path, ""
