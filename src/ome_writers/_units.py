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

import warnings
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ome_writers._schema import DimensionType


# Mapping from NGFF spatial unit names to OME-XML length symbols
# All 26 NGFF ValidSpaceUnit values → OME UnitsLength
# note, this is not a complete list of OME length units, only those defined by NGFF
# there ARE some units in OME not defined by NGFF (e.g., ly, pt, au, decameter, etc.)
# if a use case arises for these, they can be added later
NGFF_TO_OME_UNITS_LENGTH: dict[str, str] = {
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

# Reverse mapping from OME-XML length symbols to NGFF spatial unit names
ANY_LENGTH_TO_NGFF: dict[str, str] = {v: k for k, v in NGFF_TO_OME_UNITS_LENGTH.items()}
ANY_LENGTH_TO_NGFF.update({k: k for k in NGFF_TO_OME_UNITS_LENGTH})
# special cases we want to handle
ANY_LENGTH_TO_NGFF.update(
    {
        "um": "micrometer",
        "u": "micrometer",
        "micron": "micrometer",
        "A": "angstrom",
    }
)

# Mapping from NGFF temporal unit names to OME-XML time symbols
# All 23 NGFF ValidTimeUnit values → OME UnitsTime
NGFF_TO_OME_UNITS_TIME: dict[str, str] = {
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

# Reverse mapping from OME-XML time symbols to NGFF temporal unit names
ANY_TIME_TO_NGFF: dict[str, str] = {v: k for k, v in NGFF_TO_OME_UNITS_TIME.items()}
ANY_TIME_TO_NGFF.update({k: k for k in NGFF_TO_OME_UNITS_TIME})
# special cases we want to handle
ANY_TIME_TO_NGFF.update(
    {
        "us": "microsecond",
        "usec": "microsecond",
        "µsec": "microsecond",
        "msec": "millisecond",
        "sec": "second",
    }
)


def cast_unit_to_ngff(
    unit: str, dim_type: DimensionType | None
) -> tuple[str, DimensionType | None]:
    """Cast a unit string to its NGFF-compliant equivalent."""
    # all versions of units longer than 2 characters should be lowercased for matching
    unit_lower = str(unit).lower() if len(unit) > 2 else unit

    if dim_type == "space":
        try:
            return ANY_LENGTH_TO_NGFF[unit_lower], dim_type
        except KeyError as e:
            raise ValueError(
                f"Unrecognized unit of length: {unit!r}.\n  "
                f"Recognized units of length include: {list(ANY_LENGTH_TO_NGFF.keys())}"
            ) from e
    elif dim_type == "time":
        try:
            return ANY_TIME_TO_NGFF[unit_lower], dim_type
        except KeyError as e:
            raise ValueError(
                f"Unrecognized unit of time: {unit!r}.\n  "
                f"Recognized units of time include: {list(ANY_TIME_TO_NGFF.keys())}"
            ) from e

    # if the user only provided unit, but not dim_type, try to infer dim_type from unit
    elif dim_type is None:
        if unit_lower in ANY_LENGTH_TO_NGFF:
            return ANY_LENGTH_TO_NGFF[unit_lower], "space"
        elif unit_lower in ANY_TIME_TO_NGFF:
            return ANY_TIME_TO_NGFF[unit_lower], "time"

    # at this point, dim_type is either "channel", "other", or still None, but with
    # an unrecognized unit
    return unit, dim_type


NGFF_TO_OME_UNITS = {**NGFF_TO_OME_UNITS_LENGTH, **NGFF_TO_OME_UNITS_TIME}


def ngff_to_ome_unit(unit: str) -> str | None:
    """Convert an NGFF-compliant unit to its OME-XML equivalent.

    This assumes that the unit was already validated as a valid NGFF unit (above).
    This is the case in _schema.Dimension class.
    """
    try:
        return NGFF_TO_OME_UNITS[unit]
    except KeyError:
        warnings.warn(
            f"Unit {unit!r} not recognized as a valid OME spatial or temporal unit. "
            "Will not be written to OME-XML metadata.",
            stacklevel=2,
        )
    return None
