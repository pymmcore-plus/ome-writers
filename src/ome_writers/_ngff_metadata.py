from __future__ import annotations

from typing import TYPE_CHECKING

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
