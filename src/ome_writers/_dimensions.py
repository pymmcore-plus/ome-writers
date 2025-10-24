from __future__ import annotations

__all__ = [
    "Dimension",
    "DimensionLabel",
    "UnitTuple",
    "dims_to_ngff_v5",
    "dims_to_ome",
    "dims_to_yaozarrs_v5",
    "ome_meta_v5",
]

import uuid
from pathlib import Path
from typing import TYPE_CHECKING, Literal, NamedTuple, TypeAlias

import numpy as np

from ome_writers import __version__

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    import ome_types
    import yaozarrs.v05 as yao_v05

OME_DIM_TYPE = {"y": "space", "x": "space", "z": "space", "t": "time", "c": "channel"}
OME_UNIT = {"um": "micrometer", "ml": "milliliter", "s": "second", None: "unknown"}


# Recognized dimension labels
DimensionLabel: TypeAlias = Literal["x", "y", "z", "t", "c", "p", "other"]
# UnitTuple is a tuple of (scale, unit); e.g. (1, "s")
UnitTuple: TypeAlias = tuple[float, str]


class Dimension(NamedTuple):
    label: DimensionLabel
    size: int
    unit: UnitTuple | None = None
    # None or 0 indicates no constraint.
    # -1 indicates that the chunk size should equal the full extent of the domain.
    chunk_size: int | None = None

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


def dims_to_ome(
    dims: Sequence[Dimension],
    dtype: np.typing.DTypeLike,
    tiff_file_name: str | None = None,
) -> ome_types.OME:
    """Convert a sequence of Dimension objects to an OME object.

    This creates an OME representing a 5D image with the specified dimensions.

    Parameters
    ----------
    dims : Sequence[Dimension]
        The dimensions to convert.
    dtype : np.typing.DTypeLike
        The data type of the image.
    tiff_file_name : str | None
        Optional TIFF file name for metadata.

    Returns
    -------
    ome_types.OME
        The OME metadata object.

    Raises
    ------
    ImportError
        If ome-types is not installed.
    """
    try:
        from ome_types import model as m
    except ImportError as e:
        raise ImportError(
            "The `ome-types` package is required to use this method. "
            "Please install it via `pip install ome-types` or use the `tiff` extra."
        ) from e

    # Find the position dimension, if any
    if any(dim.label not in "tczyxp" for dim in dims):
        raise NotImplementedError("Only dimensions t, c, z, y, x, and p are supported.")

    dims_sizes = {dim.label: dim.size for dim in dims}
    n_positions = dims_sizes.pop("p", 1)

    _dim_names = "".join(reversed(dims_sizes)).upper()
    dim_order = next(
        (x for x in m.Pixels_DimensionOrder if x.value.startswith(_dim_names)),
        m.Pixels_DimensionOrder.XYCZT,
    )

    images: list[m.Image] = []
    channels = [
        m.Channel(
            id=f"Channel:{i}",
            name=f"Channel {i + 1}",
            samples_per_pixel=1,  # TODO
        )
        for i in range(dims_sizes.get("c", 0))
    ]

    uuid_ = f"urn:uuid:{uuid.uuid4()}"

    for p in range(n_positions):
        planes: list[m.Plane] = []
        tiff_blocks: list[m.TiffData] = []
        ifd = 0

        # iterate over ordered cartesian product of tcz sizes
        tcz_dims = [(d.label, d.size) for d in dims if d.label in "tcz"]
        if tcz_dims:
            labels, sizes = zip(*tcz_dims, strict=False)
            has_z, has_t, has_c = "z" in labels, "t" in labels, "c" in labels
        else:
            labels, sizes = (), ()
            has_z, has_t, has_c = False, False, False

        for index in np.ndindex(*sizes) if sizes else [()]:
            plane = m.Plane(
                the_z=index[labels.index("z")] if has_z else 0,
                the_t=index[labels.index("t")] if has_t else 0,
                the_c=index[labels.index("c")] if has_c else 0,
            )
            planes.append(plane)
            if tiff_file_name is not None:
                tiff_data = m.TiffData(
                    ifd=ifd,
                    uuid=m.TiffData.UUID(value=uuid_, file_name=tiff_file_name),
                    first_c=plane.the_c,
                    first_z=plane.the_z,
                    first_t=plane.the_t,
                    plane_count=1,
                )
                tiff_blocks.append(tiff_data)
            ifd += 1

        md_only = None if tiff_blocks else m.MetadataOnly()
        pix_type = m.PixelType(np.dtype(dtype).name)  # try/catch
        pixels = m.Pixels(
            id=f"Pixels:{p}",
            channels=channels,
            planes=planes,
            tiff_data_blocks=tiff_blocks,
            metadata_only=md_only,
            dimension_order=dim_order,
            type=pix_type,
            # significant_bits=..., # TODO
            size_x=dims_sizes.get("x", 1),
            size_y=dims_sizes.get("y", 1),
            size_z=dims_sizes.get("z", 1),
            size_c=dims_sizes.get("c", 1),
            size_t=dims_sizes.get("t", 1),
            # physical_size_x=voxel_size.x,
            # physical_size_y=voxel_size.y,
            # physical_size_z = voxel_size.z
            # physical_size_x_unit=UnitsLength.MICROMETER,
            # physical_size_y_unit=UnitsLength.MICROMETER,
            # physical_size_z_unit = UnitsLength.MICROMETER
        )

        base_name = Path(tiff_file_name).stem if tiff_file_name else f"Image_{p}"
        images.append(
            m.Image(
                # objective_settings=...
                id=f"Image:{p}",
                name=base_name + (f" (Series {p})" if n_positions > 1 else ""),
                pixels=pixels,
                # acquisition_date=acquisition_date,
            )
        )

    ome = m.OME(images=images, creator=f"ome_writers v{__version__}")
    return ome


def dims_to_yaozarrs_v5(array_dims: Mapping[str, Sequence[Dimension]]) -> yao_v05.Image:
    """Create OME NGFF v0.5 metadata as a yaozarrs Image object.

    Parameters
    ----------
    array_dims : Mapping[str, Sequence[Dimension]]
        A mapping of array paths to their corresponding dimension information.
        Each key is the path to a zarr array, and the value is a sequence of
        Dimension objects describing the dimensions of that array.

    Returns
    -------
    yaozarrs.v05.Image
        The OME-Zarr NGFF v0.5 Image object.

    Raises
    ------
    ImportError
        If yaozarrs is not installed.

    Example
    -------
    >>> from ome_writers import Dimension, dims_to_yaozarrs_v5
    >>> array_dims = {
    ...     "0": [
    ...         Dimension(label="t", size=1, unit=(1.0, "s")),
    ...         Dimension(label="c", size=1),
    ...         Dimension(label="z", size=1, unit=(1.0, "um")),
    ...         Dimension(label="y", size=1, unit=(1.0, "um")),
    ...         Dimension(label="x", size=1, unit=(1.0, "um")),
    ...     ],
    ... }
    >>> image = dims_to_yaozarrs_v5(array_dims)
    """
    try:
        from yaozarrs import v05
    except ImportError as e:
        raise ImportError(
            "The `yaozarrs` package is required to use this function. "
            "Please install it via `pip install yaozarrs`."
        ) from e

    # Group arrays by their axes to create multiscales entries
    multiscales_dict: dict[str, dict] = {}

    for array_path, dims in array_dims.items():
        axes, scales = _ome_axes_scales(dims)

        # Sort axes according to NGFF v0.5 spec: time, channel, space
        # Create tuples of (axis, scale, original_index) for sorting
        axis_order = {"time": 0, "channel": 1, "space": 2}
        sorted_items = sorted(
            zip(axes, scales, range(len(axes)), strict=True),
            key=lambda x: (axis_order.get(x[0]["type"], 3), x[2])
        )
        axes = [item[0] for item in sorted_items]
        scales = [item[1] for item in sorted_items]

        # Convert axes dicts to yaozarrs axis objects
        yao_axes = []
        for ax in axes:
            if ax["type"] == "time":
                yao_axes.append(
                    v05.TimeAxis(name=ax["name"], type="time", unit=ax.get("unit"))
                )
            elif ax["type"] == "space":
                yao_axes.append(
                    v05.SpaceAxis(name=ax["name"], type="space", unit=ax.get("unit"))
                )
            elif ax["type"] == "channel":
                yao_axes.append(v05.ChannelAxis(name=ax["name"], type="channel"))
            else:
                yao_axes.append(
                    v05.CustomAxis(
                        name=ax["name"], type=ax.get("type"), unit=ax.get("unit")
                    )
                )

        # Create dataset with coordinate transformations
        ct = v05.ScaleTransformation(type="scale", scale=scales)
        ds = v05.Dataset(path=array_path, coordinateTransformations=[ct])

        # Create a hashable key from axes for grouping
        axes_key = str(axes)

        # Get or create multiscale entry
        if axes_key not in multiscales_dict:
            multiscales_dict[axes_key] = {
                "axes": yao_axes,
                "datasets": []
            }

        # Add the dataset to the corresponding group
        multiscales_dict[axes_key]["datasets"].append(ds)

    # Create Multiscale objects
    multiscales = [
        v05.Multiscale(axes=ms_data["axes"], datasets=ms_data["datasets"])
        for ms_data in multiscales_dict.values()
    ]

    return v05.Image(version="0.5", multiscales=multiscales)


def _ome_axes_scales(dims: Sequence[Dimension]) -> tuple[list[dict], list[float]]:
    """Return ome axes meta.

    The length of "axes" must be between 2 and 5 and MUST be equal to the
    dimensionality of the zarr arrays storing the image data. The "axes" MUST
    contain 2 or 3 entries of "type:space" and MAY contain one additional
    entry of "type:time" and MAY contain one additional entry of
    "type:channel" or a null / custom type. The order of the entries MUST
    correspond to the order of dimensions of the zarr arrays. In addition, the
    entries MUST be ordered by "type" where the "time" axis must come first
    (if present), followed by the "channel" or custom axis (if present) and
    the axes of type "space".
    """
    axes: list[dict] = []
    scales: list[float] = []
    for dim in dims:
        axes.append(
            {"name": dim.label, "type": dim.ome_dim_type, "unit": dim.ome_unit},
        )
        scales.append(dim.ome_scale)
    return axes, scales


# Alias for backward compatibility
ome_meta_v5 = dims_to_yaozarrs_v5
