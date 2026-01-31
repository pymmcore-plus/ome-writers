from __future__ import annotations

import itertools
import uuid
import warnings
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, NamedTuple

import ome_types
import ome_types.model as ome
import tifffile

from ome_writers._schema import Channel, Dimension, Position, PositionDimension
from ome_writers._units import ngff_to_ome_unit

if TYPE_CHECKING:
    from ome_writers._schema import AcquisitionSettings

BinaryOnly = ome.OME.BinaryOnly
COMPANION_IDX = -1  # special index for .companion.ome files


class MetadataMode(Enum):
    """TIFF structure modes."""

    SINGLE_FILE = "single_file"
    """All series in a single TIFF file with full OME-XML metadata."""
    MULTI_REDUNDANT = "multi_redundant"
    """TIFF file per series; each with full OME-XML metadata."""
    MULTI_MASTER_TIFF = "multi_master_tiff"
    """TIFF file per series; full OME-XML in master TIFF, BinData in rest."""
    MULTI_MASTER_COMPANION = "multi_master_companion"
    """TIFF file per series with only BinData, full OME-XML in companion file."""


class OmeXMLMirror:
    """In-memory mirror of an on-disk OME-XML structure, with dirty tracking.

    Parameters
    ----------
    path : str
        Complete path to the OME-XML file or TIFF file that will contain the OME-XML
        metadata.
    model : ome.OME | None, optional
        Initial OME model to use. If None, an empty OME model is created.
    """

    def __init__(
        self, path: str | Path, pos_idx: int | None, model: ome.OME | None = None
    ) -> None:
        self.path: str = str(path)
        self.pos_idx: int = COMPANION_IDX if pos_idx is None else pos_idx
        self.model = model or ome.OME()
        self._dirty: bool = False

    def mark_dirty(self) -> None:
        """Mark the OME-XML as dirty (modified)."""
        self._dirty = True

    @property
    def is_tiff(self) -> bool:
        """Whether the OME-XML is stored in a TIFF file."""
        return self.path.lower().endswith((".tiff", ".tif"))

    def flush(self, *, force: bool = False) -> None:
        """Mark the OME-XML as clean (not modified)."""
        if not self._dirty and not force:  # pragma: no cover
            return

        xml_bytes = self.model.to_xml().encode("utf-8")
        if self.is_tiff:
            try:
                tifffile.tiffcomment(self.path, comment=xml_bytes)
            except FileNotFoundError:  # pragma: no cover
                warnings.warn(
                    f"TIFF file {self.path} not found when writing OME-XML comment.",
                    stacklevel=2,
                )
        else:
            # companion file
            with open(self.path, mode="wb") as f:
                f.write(xml_bytes)
                print("✓ Wrote OME-XML to", self.path)

        self._dirty = False


def prepare_metadata(
    settings: AcquisitionSettings, mode: MetadataMode = MetadataMode.MULTI_REDUNDANT
) -> dict[str, OmeXMLMirror]:
    """Create OME-XML mirrors based on the acquisition settings and metadata mode.

    Parameters
    ----------
    settings : AcquisitionSettings
        The acquisition settings containing file paths.
    mode : MetadataMode
        The metadata mode to use.

    Returns
    -------
    dict[str, OmeXMLMirror]
        Mapping of file paths to their OME-XML mirrors.
    """
    if any(
        dim.name.lower() not in "tczyx"
        for dim in settings.dimensions
        if not isinstance(dim, PositionDimension)
    ):  # pragma: no cover
        raise ValueError("Dimension names must be one of 't', 'c', 'z', 'y', 'x'")

    root = Path(settings.output_path).expanduser().resolve()
    if len(settings.positions) <= 1:
        mode = MetadataMode.SINGLE_FILE

    # Generate file info (path + UUID) for each position
    file_infos = _generate_file_infos(root, settings.positions, mode)
    for info in file_infos:
        # Handle overwrite
        path = Path(info.path)
        if path.exists():
            if not settings.overwrite:
                raise FileExistsError(
                    f"File {path} already exists. Use overwrite=True to overwrite it."
                )
            path.unlink()

        # Ensure parent directory exists
        path.parent.mkdir(parents=True, exist_ok=True)

    # Build the complete OME model with all series
    full_model = _build_full_model(settings, file_infos, mode)

    # Create mirrors based on mode
    mirrors: dict[str, OmeXMLMirror] = {}

    if mode == MetadataMode.SINGLE_FILE:
        # Single file contains everything
        info = file_infos[0]
        mirrors[info.path] = OmeXMLMirror(
            path=info.path, pos_idx=info.pos_idx, model=full_model
        )

    elif mode == MetadataMode.MULTI_REDUNDANT:
        # Each file gets a full copy, differing only in root UUID
        for info in file_infos:
            mirrors[info.path] = OmeXMLMirror(
                path=info.path,
                pos_idx=info.pos_idx,
                model=full_model.model_copy(deep=True, update={"uuid": info.uuid}),
            )

    elif mode == MetadataMode.MULTI_MASTER_TIFF:
        # First file is master with full metadata
        master_info = file_infos[0]
        full_model.uuid = master_info.uuid
        mirrors[master_info.path] = OmeXMLMirror(
            path=master_info.path,
            pos_idx=master_info.pos_idx,
            model=full_model,
        )

        # Other files get BinaryOnly
        for info in file_infos[1:]:
            mirrors[info.path] = OmeXMLMirror(
                path=info.path,
                pos_idx=info.pos_idx,
                model=ome_types.OME(
                    uuid=info.uuid,
                    binary_only=BinaryOnly(
                        metadata_file=master_info.path,
                        uuid=master_info.uuid,
                    ),
                ),
            )

    elif mode == MetadataMode.MULTI_MASTER_COMPANION:
        companion_path = str(root.with_suffix(".companion.ome"))
        companion_uuid = _make_uuid()

        # Companion file gets full metadata
        full_model.uuid = companion_uuid
        mirrors[companion_path] = OmeXMLMirror(
            path=companion_path,
            pos_idx=None,
            model=full_model,
        )

        # All TIFF files get BinaryOnly
        for info in file_infos:
            mirrors[info.path] = OmeXMLMirror(
                path=info.path,
                pos_idx=info.pos_idx,
                model=ome_types.OME(
                    uuid=info.uuid,
                    binary_only=BinaryOnly(
                        metadata_file=companion_path,
                        uuid=companion_uuid,
                    ),
                ),
            )

    return mirrors


# ---------------------------------------------------------------------------
# Helper types and functions
# ---------------------------------------------------------------------------


class FileInfo(NamedTuple):
    """Simple container for file path and UUID."""

    path: str
    uuid: str
    pos_idx: int | None  # None for companion file


def _make_uuid() -> str:
    """Generate a URN-formatted UUID."""
    return f"urn:uuid:{uuid.uuid4()}"


def _generate_file_infos(
    root: Path, positions: list[Position], mode: MetadataMode
) -> list[FileInfo]:
    """Generate file paths and UUIDs for each position."""

    if mode == MetadataMode.SINGLE_FILE:
        # Single file for all positions
        return [FileInfo(path=str(root), uuid=_make_uuid(), pos_idx=0)]

    # XXX: if we switch to using pos.name directly, we need to sanitize it
    # def _safename(name: str) -> str:
    #     """Sanitize a string to be safe for filenames."""
    #     if str.isdigit(name):
    #         return f"p{name:04}"
    #     return name.replace("/", "_").replace("\\", "_")

    # Multi-file: one file per position
    extension = "".join(root.suffixes)
    stem = root.stem.removesuffix(".ome")
    return [
        FileInfo(
            path=str(root.parent / f"{stem}_p{idx:03}{extension}"),
            uuid=_make_uuid(),
            pos_idx=idx,
        )
        for idx, pos in enumerate(positions)
    ]


def _build_full_model(
    settings: AcquisitionSettings, file_infos: list[FileInfo], mode: MetadataMode
) -> ome_types.OME:
    """Build complete OME model with all series/images."""
    dims = [d for d in settings.dimensions if not isinstance(d, PositionDimension)]
    dimension_order = _get_dimension_order(dims)
    channel_dim = next(
        (d for d in dims if d.type == "channel" or d.name.lower() == "c"), None
    )
    pixel_sizes = {"z": 1, "c": 1, "t": 1}
    pixel_sizes.update({d.name.lower(): d.count or 1 for d in dims})
    size_t = pixel_sizes["t"]
    size_c = pixel_sizes["c"]
    size_z = pixel_sizes["z"]

    single_file = mode == MetadataMode.SINGLE_FILE

    # Track cumulative IFD offset for single-file mode
    ifd_offset = 0
    planes_per_series = size_t * size_c * size_z

    images: list[ome.Image] = []
    for i, pos in enumerate(settings.positions):
        # For single file, all series reference the same file (no UUID children)
        # For multi-file, each series references all files via UUID children
        if single_file:
            tiff_data = ome.TiffData(ifd=ifd_offset, plane_count=planes_per_series)
            ifd_offset += planes_per_series
        else:
            # Create TiffData for this series pointing to its file
            # Each series lives in one file, so one TiffData with UUID
            file_info = file_infos[i]
            relative_path = Path(file_info.path).name
            tiff_data = ome.TiffData(
                plane_count=planes_per_series,
                uuid=ome.TiffData.UUID(file_name=relative_path, value=file_info.uuid),
            )

        if channel_dim and channel_dim.coords:
            # Use full channel information if
            channels = [
                _cast_channel(omw_channel=c, id=f"Channel:{i}:{cidx}")
                for cidx, c in enumerate(channel_dim.coords)
            ]
        else:
            channels = [ome.Channel(id=f"Channel:{i}:{c}") for c in range(size_c)]

        physical_sizes = _get_physical_sizes(dims)
        pixels = ome.Pixels(
            id=f"Pixels:{i}",
            dimension_order=dimension_order,
            size_x=pixel_sizes["x"],
            size_y=pixel_sizes["y"],
            size_z=size_z,
            size_c=size_c,
            size_t=size_t,
            **physical_sizes,
            type=settings.dtype,
            # big_endian=False,
            channels=channels,
            tiff_data_blocks=[tiff_data],
        )
        image = ome.Image(
            id=f"Image:{i}",
            name=pos.name,
            pixels=pixels,
            acquisition_date=datetime.now(timezone.utc),
        )
        images.append(image)

    plates = _build_plates(settings)
    return ome_types.OME(uuid=_make_uuid(), images=images, plates=plates)


VALID_ORDERS = [x.value for x in ome.Pixels_DimensionOrder]


def _get_dimension_order(dims: list[Dimension]) -> str:
    # suffix is some combination of ZCT in order of appearance
    # reversed to match OME dimension order conventions (f)
    suffix = "".join(d.name.upper() for d in dims if d.name.upper() not in "XY")[::-1]
    for order in VALID_ORDERS:
        if order[2:].startswith(suffix):
            return order
    raise ValueError(f"No valid order matches {suffix}")  # pragma: no cover


def _get_physical_sizes(dims: list[Dimension]) -> dict:
    output = {}
    dims_map = {dim.name.lower(): dim for dim in dims}
    for axis in ["x", "y", "z"]:
        if (dim := dims_map.get(axis)) and dim.scale is not None:
            output[f"physical_size_{axis}"] = dim.scale
            # if dim.type is space, it's guaranteed to be a valid NGFF unit
            if (
                dim.type == "space"
                and dim.unit
                and (ome_unit := ngff_to_ome_unit(dim.unit))
            ):
                output[f"physical_size_{axis}_unit"] = ome_unit
    return output


def _build_plates(settings: AcquisitionSettings) -> list[ome.Plate]:
    """Build OME Plate with Wells and WellSamples linking to Images.

    Each position maps to a WellSample, which links to an Image via ImageRef.
    Wells are determined by unique (plate_row, plate_column) combinations.
    """
    if not (positions := settings.positions) or not (plate := settings.plate):
        return []

    # all known (row, column) keys based on plate definition
    valid_keys = set(itertools.product(plate.row_names, plate.column_names))

    # Group positions by well (row, column), filtering out invalid coordinates
    wells_map: dict[tuple[str, str], list[tuple[int, Position]]] = {}
    for idx, pos in enumerate(positions):
        # in AcquisitionSettings._validate_plate_positions, we already warned the user
        # that positions with with plate_row/plate_column that aren't represented
        # in the plate definition will be skipped from the metadata.
        # So here we just skip them here silently.
        key = (pos.plate_row, pos.plate_column)
        if key in valid_keys:
            wells_map.setdefault(key, []).append((idx, pos))

    # Build Well objects with WellSamples
    wells: list[ome.Well] = []
    for (row_name, col_name), positions in wells_map.items():
        row_idx = plate.row_names.index(row_name)
        col_idx = plate.column_names.index(col_name)

        well_samples = [
            ome.WellSample(
                id=f"WellSample:{idx}",
                index=idx,
                image_ref=ome.ImageRef(id=f"Image:{idx}"),
            )
            for idx, _pos in positions
        ]

        wells.append(
            ome.Well(
                id=f"Well:{row_idx}_{col_idx}",
                row=row_idx,
                column=col_idx,
                well_samples=well_samples,
            )
        )

    row_conv = _infer_naming_convention(plate.row_names)
    col_conv = _infer_naming_convention(plate.column_names)

    return [
        ome.Plate(
            id="Plate:0",
            name=plate.name,
            rows=len(plate.row_names),
            columns=len(plate.column_names),
            row_naming_convention=row_conv,
            column_naming_convention=col_conv,
            wells=wells,
        )
    ]


def _infer_naming_convention(names: list[str]) -> ome.NamingConvention | None:
    """Infer naming convention from a list of names."""
    if all(n.isalpha() for n in names):
        return ome.NamingConvention.LETTER
    if all(n.isdigit() for n in names):
        return ome.NamingConvention.NUMBER
    return None  # pragma: no cover


def _cast_channel(omw_channel: Channel, id: str) -> ome.Channel:
    """Cast an ome_writers Channel to an ome_types.model Channel."""
    color = {"color": omw_channel.color.as_rgb_tuple()} if omw_channel.color else {}
    return ome.Channel(
        id=id,
        name=omw_channel.name,
        **color,
        emission_wavelength=omw_channel.emission_wavelength_nm,
        emission_wavelength_unit=ome.UnitsLength.NANOMETER,
        excitation_wavelength=omw_channel.excitation_wavelength_nm,
        excitation_wavelength_unit=ome.UnitsLength.NANOMETER,
        fluor=omw_channel.fluorophore,
    )
