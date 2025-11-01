"""Plate metadata for organizing multi-well/multi-position acquisitions."""

from __future__ import annotations

__all__ = [
    "Plate",
    "PlateAcquisition",
    "WellPosition",
    "plate_to_acquire_zarr",
    "plate_to_ome_types",
    "plate_to_yaozarrs_v5",
]

from typing import TYPE_CHECKING, NamedTuple

if TYPE_CHECKING:
    from collections.abc import Sequence

    import acquire_zarr as aqz
    import numpy as np
    import ome_types.model as ome
    import yaozarrs.v05 as yao_v05


class PlateAcquisition(NamedTuple):
    """Represents metadata for a plate acquisition.

    Parameters
    ----------
    id : int
        A unique identifier for this acquisition within the plate context.
    name : str | None
        Optional name for the acquisition.
    description : str | None
        Optional description of the acquisition.
    start_time : int | None
        Start timestamp (epoch time in seconds).
    end_time : int | None
        End timestamp (epoch time in seconds).
    maximum_field_count : int | None
        Maximum number of fields of view for this acquisition.
    """

    id: int
    name: str | None = None
    description: str | None = None
    start_time: int | None = None
    end_time: int | None = None
    maximum_field_count: int | None = None


class WellPosition(NamedTuple):
    """Represents a well position in a plate.

    Parameters
    ----------
    path : str
        Path to the well subgroup (e.g., "A/01", "B/03").
    row_index : int
        Zero-based row index.
    column_index : int
        Zero-based column index.
    """

    path: str
    row_index: int
    column_index: int


class Plate(NamedTuple):
    """Represents plate metadata for multi-well experiments.

    This structure is used to define plate layouts with rows, columns, and wells.
    It can be converted to both OME-TIFF metadata (ome_types.Plate) and
    OME-Zarr NGFF v0.5 metadata (yaozarrs.v05.Plate).

    Parameters
    ----------
    rows : Sequence[str]
        Row names (e.g., ["A", "B", "C"]).
    columns : Sequence[str]
        Column names (e.g., ["01", "02", "03"]).
    wells : Sequence[WellPosition]
        Well positions in the plate.
    name : str | None
        Optional name for the plate.
    field_count : int | None
        Maximum number of fields per well.
    acquisitions : Sequence[PlateAcquisition] | None
        Optional acquisition metadata.

    Examples
    --------
    >>> from ome_writers import Plate, WellPosition, PlateAcquisition
    >>> plate = Plate(
    ...     rows=["A", "B"],
    ...     columns=["01", "02"],
    ...     wells=[
    ...         WellPosition("A/01", 0, 0),
    ...         WellPosition("A/02", 0, 1),
    ...         WellPosition("B/01", 1, 0),
    ...         WellPosition("B/02", 1, 1),
    ...     ],
    ...     name="My Plate",
    ...     acquisitions=[PlateAcquisition(id=0, name="Scan1")],
    ... )
    """

    rows: Sequence[str]
    columns: Sequence[str]
    wells: Sequence[WellPosition]
    name: str | None = None
    field_count: int | None = None
    acquisitions: Sequence[PlateAcquisition] | None = None


def plate_to_ome_types(plate: Plate) -> ome.Plate:
    """Convert a Plate to an ome_types.Plate object.

    Parameters
    ----------
    plate : Plate
        The plate metadata to convert.

    Returns
    -------
    ome.Plate
        The plate metadata as an ome_types.Plate object.

    Raises
    ------
    ImportError
        If ome-types is not installed.
    """
    try:
        from ome_types import model as m
    except ImportError as e:
        raise ImportError(
            "The `ome-types` package is required to use this function. "
            "Please install it via `pip install ome-types`."
        ) from e

    # Create wells with well samples
    ome_wells: list[m.Well] = []
    for well_pos in plate.wells:
        # For OME-TIFF, each well can have multiple well samples (fields)
        # We'll create one sample per well for now, referencing the image
        img_idx = well_pos.row_index * len(plate.columns) + well_pos.column_index
        well_samples = [
            m.WellSample(
                id=f"WellSample:{well_pos.row_index}:{well_pos.column_index}:0",
                index=0,
                image_ref=m.ImageRef(id=f"Image:{img_idx}"),
            )
        ]

        ome_wells.append(
            m.Well(
                id=f"Well:{well_pos.row_index}:{well_pos.column_index}",
                row=well_pos.row_index,
                column=well_pos.column_index,
                well_samples=well_samples,
            )
        )

    # Create plate annotations if acquisitions are provided
    # Note: ome-types expects datetime objects for start_time/end_time
    # but we store epoch timestamps for compatibility with NGFF
    plate_acquisitions: list[m.PlateAcquisition] | None = None
    if plate.acquisitions:
        from datetime import datetime, timezone

        plate_acquisitions = [
            m.PlateAcquisition(
                id=f"PlateAcquisition:{acq.id}",
                name=acq.name,
                description=acq.description,
                start_time=(
                    datetime.fromtimestamp(acq.start_time, tz=timezone.utc)
                    if acq.start_time is not None
                    else None
                ),
                end_time=(
                    datetime.fromtimestamp(acq.end_time, tz=timezone.utc)
                    if acq.end_time is not None
                    else None
                ),
                maximum_field_count=acq.maximum_field_count,
            )
            for acq in plate.acquisitions
        ]

    plate_obj = m.Plate(
        id="Plate:0",
        name=plate.name,
        rows=len(plate.rows),
        columns=len(plate.columns),
        wells=ome_wells,
    )

    if plate_acquisitions is not None:
        plate_obj.plate_acquisitions = plate_acquisitions

    return plate_obj


def plate_to_yaozarrs_v5(plate: Plate) -> yao_v05.Plate:
    """Convert a Plate to a yaozarrs v0.5 Plate object.

    Parameters
    ----------
    plate : Plate
        The plate metadata to convert.

    Returns
    -------
    yaozarrs.v05.Plate
        The plate metadata as a yaozarrs v0.5 Plate object.

    Raises
    ------
    ImportError
        If yaozarrs is not installed.
    """
    try:
        from yaozarrs import v05
    except ImportError as e:
        raise ImportError(
            "The `yaozarrs` package is required to use this function. "
            "Please install it via `pip install yaozarrs`."
        ) from e

    # Create row and column metadata
    yao_rows = [v05.Row(name=row_name) for row_name in plate.rows]
    yao_columns = [v05.Column(name=col_name) for col_name in plate.columns]

    # Create well metadata
    yao_wells = [
        v05.PlateWell(
            path=well_pos.path,
            rowIndex=well_pos.row_index,
            columnIndex=well_pos.column_index,
        )
        for well_pos in plate.wells
    ]

    # Create acquisition metadata if provided
    yao_acquisitions: list[v05.Acquisition] | None = None
    if plate.acquisitions:
        yao_acquisitions = [
            v05.Acquisition(
                id=acq.id,
                name=acq.name,
                description=acq.description,
                starttime=acq.start_time,
                endtime=acq.end_time,
                maximumfieldcount=acq.maximum_field_count,
            )
            for acq in plate.acquisitions
        ]

    # Create the plate definition
    plate_def = v05.PlateDef(
        rows=yao_rows,
        columns=yao_columns,
        wells=yao_wells,
        name=plate.name,
        field_count=plate.field_count,
        acquisitions=yao_acquisitions,
    )

    return v05.Plate(version="0.5", plate=plate_def)


def plate_to_acquire_zarr(
    plate: Plate,
    az_dims: list[aqz.Dimension],
    dtype: np.dtype,
) -> aqz.Plate:
    """Convert ome_writers Plate to acquire-zarr Plate object.

    This method creates an acquire-zarr Plate object with proper well and
    field of view structure from an ome_writers Plate object.

    Parameters
    ----------
    plate : Plate
        The ome_writers plate object to convert.
    az_dims : list[aqz.Dimension]
        The acquire-zarr dimensions for the arrays.
    dtype : np.dtype
        The NumPy data type for the image data.

    Returns
    -------
    aqz.Plate
        The created acquire-zarr Plate object.
    """
    try:
        import acquire_zarr as aqz
    except ImportError as e:
        raise ImportError(
            "The `acquire-zarr` package is required to use this function. "
            "Install it with `pip install acquire-zarr`."
        ) from e

    fields_per_well = plate.field_count or 1

    # Create acquisition metadata if provided
    acquisitions = []
    if plate.acquisitions:
        for acq in plate.acquisitions:
            acquisitions.append(
                aqz.Acquisition(
                    id=acq.id,
                    name=acq.name or f"Acquisition {acq.id}",
                    start_time=acq.start_time,
                    end_time=acq.end_time,
                )
            )

    # Create wells with fields of view
    wells = []
    for well_pos in plate.wells:
        row_name, col_name = well_pos.path.split("/")

        # Create field of view entries for this well
        fov_entries = []
        for fov_idx in range(fields_per_well):
            fov_key = f"fov{fov_idx}"
            # Create a new ArraySettings for each FOV to ensure proper configuration
            fov_array = aqz.ArraySettings(
                output_key=fov_key,
                dimensions=az_dims,
                data_type=dtype,
            )
            fov_entries.append(
                aqz.FieldOfView(
                    path=fov_key,
                    acquisition_id=acquisitions[0].id if acquisitions else None,
                    array_settings=fov_array,
                )
            )

        well = aqz.Well(
            row_name=row_name,
            column_name=col_name,
            images=fov_entries,
        )
        wells.append(well)

    # Create the HCS plate
    # remove any spaces from the plate name for acquire-zarr
    plate_path = (plate.name or "plate").replace(" ", "_")
    plate_aqz = aqz.Plate(
        path=plate_path,
        name=plate.name,
        row_names=list(plate.rows),
        column_names=list(plate.columns),
        wells=wells,
        acquisitions=acquisitions if acquisitions else None,
    )

    return plate_aqz
