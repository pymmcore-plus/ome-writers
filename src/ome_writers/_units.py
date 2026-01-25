"""Unit conversion utilities for OME-NGFF and OME-XML compatibility.

IMPORTANT: Users MUST use NGFF-compliant units when specifying Dimension.unit.
Valid units are defined in the yaozarrs ValidSpaceUnit and ValidTimeUnit type aliases.

Examples of NGFF-compliant units:
- Spatial: "micrometer", "nanometer", "millimeter", "meter", etc.
- Temporal: "second", "millisecond", "microsecond", "minute", "hour", etc.

This module handles automatic conversion from NGFF units to OME-XML symbols
for the TIFF backend (e.g., "micrometer" → "µm").

The mappings are based on:
- yaozarrs ValidSpaceUnit and ValidTimeUnit type aliases
  https://github.com/tlambert03/yaozarrs/blob/main/src/yaozarrs/_axis.py
- ome-types UnitsLength and UnitsTime enums
  https://ome-types.readthedocs.io/en/latest/API/ome_types.model/
"""

from __future__ import annotations

# Mapping from NGFF spatial unit names to OME-XML length symbols
# All 26 NGFF ValidSpaceUnit values → OME UnitsLength
_NGFF_TO_OME_LENGTH: dict[str, str] = {
    "angstrom": "Å",
    "attometer": "am",
    "centimeter": "cm",
    "decimeter": "dm",
    "exameter": "Em",
    "femtometer": "fm",
    "foot": "ft",
    "gigameter": "Gm",
    "hectometer": "hm",
    "inch": "in",
    "kilometer": "km",
    "megameter": "Mm",
    "meter": "m",
    "micrometer": "µm",
    "mile": "mi",
    "millimeter": "mm",
    "nanometer": "nm",
    "parsec": "pc",
    "petameter": "Pm",
    "picometer": "pm",
    "terameter": "Tm",
    "yard": "yd",
    "yoctometer": "ym",
    "yottameter": "Ym",
    "zeptometer": "zm",
    "zettameter": "Zm",
}

# Mapping from NGFF temporal unit names to OME-XML time symbols
# All 23 NGFF ValidTimeUnit values → OME UnitsTime
_NGFF_TO_OME_TIME: dict[str, str] = {
    "attosecond": "as",
    "centisecond": "cs",
    "day": "d",
    "decisecond": "ds",
    "exasecond": "Es",
    "femtosecond": "fs",
    "gigasecond": "Gs",
    "hectosecond": "hs",
    "hour": "h",
    "kilosecond": "ks",
    "megasecond": "Ms",
    "microsecond": "µs",
    "millisecond": "ms",
    "minute": "min",
    "nanosecond": "ns",
    "petasecond": "Ps",
    "picosecond": "ps",
    "second": "s",
    "terasecond": "Ts",
    "yoctosecond": "ys",
    "yottasecond": "Ys",
    "zeptosecond": "zs",
    "zettasecond": "Zs",
}


def validate_ngff_unit(unit: str | None) -> str | None:
    """Validate that a unit is NGFF-compliant.

    This validator ensures that any unit specified in a Dimension
    is a valid NGFF unit name from the yaozarrs specifications.

    Parameters
    ----------
    unit : str | None
        Unit name to validate

    Returns
    -------
    str | None
        The validated unit (unchanged if valid), or None if unit was None

    Raises
    ------
    ValueError
        If unit is not a valid NGFF unit name

    Examples
    --------
    >>> validate_ngff_unit("micrometer")
    'micrometer'
    >>> validate_ngff_unit("second")
    'second'
    >>> validate_ngff_unit(None)
    >>> validate_ngff_unit("invalid")  # doctest: +SKIP
    Traceback (most recent call last):
        ...
    ValueError: Invalid NGFF unit: 'invalid'. ...
    """
    if unit is None:
        return None

    if unit not in {*_NGFF_TO_OME_LENGTH, *_NGFF_TO_OME_TIME}:
        raise ValueError(
            f"Invalid NGFF unit: {unit!r}. Must be one of the valid NGFF "
            f"unit names. Valid spatial units: {sorted(_NGFF_TO_OME_LENGTH.keys())}. "
            f"Valid temporal units: {sorted(_NGFF_TO_OME_TIME.keys())}."
        )
    return unit


def ngff_to_ome_unit(unit: str) -> str:
    """Convert NGFF unit name to OME-XML symbol for TIFF backend.

    This function is used internally by the TIFF backend to convert NGFF-compliant
    units (e.g., "micrometer") to OME-XML symbols (e.g., "µm").

    Users should always specify units using NGFF-compliant names in their
    Dimension specifications. The conversion to OME-XML format happens
    automatically when writing TIFF files.

    Parameters
    ----------
    unit : str
        NGFF-compliant unit name (e.g., "micrometer", "second")

    Returns
    -------
    str
        OME-XML unit symbol (e.g., "µm", "s")
        Returns input unchanged if no mapping exists.

    Examples
    --------
    >>> ngff_to_ome_unit("micrometer")
    'µm'
    >>> ngff_to_ome_unit("second")
    's'
    >>> ngff_to_ome_unit("nanometer")
    'nm'
    """
    # Combined mapping of all NGFF units → OME symbols
    ngff_to_ome = {**_NGFF_TO_OME_LENGTH, **_NGFF_TO_OME_TIME}
    return ngff_to_ome.get(unit, unit)
