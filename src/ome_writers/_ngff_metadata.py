from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from ome_writers._dimensions import Dimension


def ome_meta_v5(array_dims: Mapping[str, Sequence[Dimension]]) -> dict:
    """Create OME NGFF v0.5 metadata.

    Parameters
    ----------
    array_dims : Mapping[str, Sequence[DimensionInfo]]
        A mapping of array paths to their corresponding dimension information.
        Each key is the path to a zarr array, and the value is a sequence of
        DimensionInfo objects describing the dimensions of that array.

    Example
    -------
    >>> from ome_writers import DimensionInfo, ome_meta_v5
    >>> array_dims = {
        "0": [
            DimensionInfo(label="t", size=1, unit=(1.0, "s")),
            DimensionInfo(label="c", size=1, unit=(1.0, "s")),
            DimensionInfo(label="z", size=1, unit=(1.0, "s")),
            DimensionInfo(label="y", size=1, unit=(1.0, "s")),
            DimensionInfo(label="x", size=1, unit=(1.0, "s")),
        ],
    }
    >>> ome_meta = ome_meta_v5(array_dims)
    """
    # Group arrays by their axes to create multiscales entries
    multiscales: dict[str, dict] = {}

    for array_path, dims in array_dims.items():
        axes, scales = _ome_axes_scales(dims)
        ct = {"scale": scales, "type": "scale"}
        ds = {"path": array_path, "coordinateTransformations": [ct]}

        # Create a hashable key from axes for grouping
        axes_key = str(axes)
        # Create a new entry for this axes configuration if it doesn't exist
        # (in the case where multiple arrays share the same axes, we want to
        # create multiple datasets under the same multiscale entry, rather than
        # creating a new multiscale entry with a single dataset each time)
        multiscale = multiscales.setdefault(axes_key, {"axes": axes, "datasets": []})

        # Add the dataset to the corresponding group
        multiscale["datasets"].append(ds)

    attrs = {"ome": {"version": "0.5", "multiscales": list(multiscales.values())}}
    return attrs


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


def build_yaozarrs_image_metadata_v05(dims: Sequence[Dimension]) -> Any:
    """Build a yaozarrs v05 Image metadata model from Dimension objects.

    Parameters
    ----------
    dims : Sequence[Dimension]
        Sequence of dimensions describing the image axes.

    Returns
    -------
    Any
        A yaozarrs v05 Image model with a single multiscale.
    """
    try:
        from yaozarrs import v05
    except ImportError as e:
        raise ImportError(
            "The `yaozarrs` package is required to use this function. "
            "Please install it via `pip install yaozarrs`."
        ) from e

    axes = []
    scales = []

    for dim in dims:
        axis = dim_to_yaozarrs_axis_v05(dim)
        axes.append(axis)
        scales.append(dim.ome_scale)

    # Create the Image model with a single multiscale (single resolution level)
    image = v05.Image(
        multiscales=[
            v05.Multiscale(
                axes=axes,
                datasets=[
                    v05.Dataset(
                        path="0",
                        coordinateTransformations=[
                            v05.ScaleTransformation(scale=scales)
                        ],
                    )
                ],
            )
        ],
    )
    return image


def dim_to_yaozarrs_axis_v05(dim: Dimension) -> Any:
    """Convert a Dimension to a yaozarrs v05 Axis object.

    Parameters
    ----------
    dim : Dimension
        The dimension to convert.

    Returns
    -------
    Any
        A yaozarrs v05 Axis object (TimeAxis | SpaceAxis | ChannelAxis | CustomAxis).
    """
    try:
        from yaozarrs import v05
    except ImportError as e:
        raise ImportError(
            "The `yaozarrs` package is required to use this function. "
            "Please install it via `pip install yaozarrs`."
        ) from e

    label = dim.label
    unit = dim.ome_unit if dim.ome_unit != "unknown" else None

    if label == "t":
        return v05.TimeAxis(name=label, unit=unit)
    elif label == "c":
        return v05.ChannelAxis(name=label)
    elif label in ("x", "y", "z"):
        return v05.SpaceAxis(name=label, unit=unit)
    else:
        return v05.CustomAxis(name=label)
