from __future__ import annotations

import uuid
from pathlib import Path
from typing import TYPE_CHECKING, Literal, NamedTuple

import numpy as np
from ome_types import model as m
from typing_extensions import TypeAlias

from ome_writers import __version__

if TYPE_CHECKING:
    from collections.abc import Sequence

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
) -> m.OME:
    """Convert a sequence of Dimension objects to an OME object.

    This creates an OME representing a 5D image with the specified dimensions.
    """
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
        labels, sizes = zip(*[(d.label, d.size) for d in dims if d.label in "tcz"])
        has_z, has_t, has_c = "z" in labels, "t" in labels, "c" in labels
        for index in np.ndindex(*sizes):
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
