from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from ome_writers._dimensions import DimensionInfo


def ome_meta_v5(array_dims: Mapping[str, Sequence[DimensionInfo]]) -> dict:
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
    multiscales = []
    for array_path, dims in array_dims.items():
        axes, scales = _ome_axes_scales(dims)
        ct = {"scale": scales, "type": "scale"}
        ds = {"path": array_path, "coordinateTransformations": [ct]}
        multiscales.append({"axes": axes, "datasets": [ds]})
    attrs = {"ome": {"version": "0.5", "multiscales": multiscales}}
    return attrs


def _ome_axes_scales(dims: Sequence[DimensionInfo]) -> tuple[list[dict], list[float]]:
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
