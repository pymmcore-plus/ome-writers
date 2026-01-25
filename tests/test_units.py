"""Tests for unit conversion and validation."""

from __future__ import annotations

import pytest

from ome_writers import Dimension
from ome_writers._units import (
    ANY_LENGTH_TO_NGFF,
    ANY_TIME_TO_NGFF,
    NGFF_TO_OME_UNITS_LENGTH,
    NGFF_TO_OME_UNITS_TIME,
    ngff_to_ome_unit,
)


@pytest.mark.parametrize(
    ("ngff_unit", "ome_symbol"),
    [
        # Common spatial units
        ("micrometer", "µm"),
        ("nanometer", "nm"),
        ("millimeter", "mm"),
        ("meter", "m"),
        ("angstrom", "Å"),
        # Common temporal units
        ("second", "s"),
        ("millisecond", "ms"),
        ("microsecond", "µs"),
        ("minute", "min"),
        ("hour", "h"),
    ],
)
def test_ngff_to_ome_unit_conversion(ngff_unit: str, ome_symbol: str) -> None:
    """Test conversion from NGFF unit names to OME-XML symbols."""
    assert ngff_to_ome_unit(ngff_unit) == ome_symbol


def test_ngff_to_ome_unit_unknown() -> None:
    """Test that unknown units are returned unchanged."""
    with pytest.warns(UserWarning, match="Will not be written to OME-XML metadata"):
        ngff_to_ome_unit("unknown_unit")


def test_ngff_length_mapping_completeness() -> None:
    """Test that all spatial unit mappings are present and valid."""
    # assert there are no overlapping keys between length and time mappings
    overlap = set(ANY_LENGTH_TO_NGFF.keys()).intersection(set(ANY_TIME_TO_NGFF.keys()))
    if overlap:
        raise AssertionError(
            f"Overlapping unit keys between length and time mappings: {overlap}"
        )
    # assert that all values in the ANY_* mappings are valid NGFF units
    for val in ANY_LENGTH_TO_NGFF.values():
        if val not in NGFF_TO_OME_UNITS_LENGTH:
            raise AssertionError(
                f"Invalid NGFF length unit value in ANY_LENGTH_TO_NGFF mapping: {val!r}"
            )
    for val in ANY_TIME_TO_NGFF.values():
        if val not in NGFF_TO_OME_UNITS_TIME:
            raise AssertionError(
                f"Invalid NGFF time unit value in ANY_TIME_TO_NGFF mapping: {val!r}"
            )


@pytest.mark.parametrize(
    ("unit", "dim_type", "expected_unit", "expected_dim_type"),
    [
        # Standard cases where all values are provided
        ("micrometer", "space", "micrometer", "space"),
        ("nanometer", "space", "nanometer", "space"),
        ("millimeter", "space", "millimeter", "space"),
        ("meter", "space", "meter", "space"),
        ("second", "time", "second", "time"),
        ("millisecond", "time", "millisecond", "time"),
        ("microsecond", "time", "microsecond", "time"),
        ("minute", "time", "minute", "time"),
        # None unit or is allowed
        (None, "space", None, "space"),
        # Abbreviations are cast to full NGFF names
        ("µm", "space", "micrometer", "space"),
        ("um", "space", "micrometer", "space"),
        ("micron", "space", "micrometer", "space"),
        ("nm", "space", "nanometer", "space"),
        ("mm", "space", "millimeter", "space"),
        ("m", "space", "meter", "space"),
        ("s", "time", "second", "time"),
        ("ms", "time", "millisecond", "time"),
        ("µs", "time", "microsecond", "time"),
        ("min", "time", "minute", "time"),
        ("h", "time", "hour", "time"),
        # Dim type inferred from unit
        ("um", None, "micrometer", "space"),
        ("millimeter", None, "millimeter", "space"),
        ("ms", None, "millisecond", "time"),
        # But if no dim-type is provided, unit can still be anything
        ("unrecognized-unit", None, "unrecognized-unit", None),
    ],
)
def test_dimension_valid_ngff_units(
    unit: str | None,
    dim_type: str | None,
    expected_unit: str | None,
    expected_dim_type: str | None,
) -> None:
    """Test that valid NGFF units are accepted in Dimension."""
    dim = Dimension(name="name", type=dim_type, unit=unit)
    assert dim.unit == expected_unit
    assert dim.type == expected_dim_type


@pytest.mark.parametrize(
    ("unit", "dim_type"),
    [
        ("not-a-space", "space"),
        ("not-a-time", "time"),
    ],
)
def test_dimension_invalid_ngff_units(
    unit: str | None,
    dim_type: str | None,
) -> None:
    """Test that valid NGFF units are accepted in Dimension."""
    with pytest.raises(ValueError, match="Unrecognized unit of"):
        Dimension(name="name", type=dim_type, unit=unit)
