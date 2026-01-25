"""Tests for unit conversion and validation."""

from __future__ import annotations

import pytest

from ome_writers._units import (
    _NGFF_TO_OME_LENGTH,
    _NGFF_TO_OME_TIME,
    ngff_to_ome_unit,
    validate_ngff_unit,
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
    unknown = "unknown_unit"
    assert ngff_to_ome_unit(unknown) == unknown


def test_validate_ngff_unit_valid() -> None:
    """Test that valid NGFF units pass validation."""
    # Spatial units
    assert validate_ngff_unit("micrometer") == "micrometer"
    assert validate_ngff_unit("nanometer") == "nanometer"

    # Temporal units
    assert validate_ngff_unit("second") == "second"
    assert validate_ngff_unit("millisecond") == "millisecond"

    # None is valid
    assert validate_ngff_unit(None) is None


def test_validate_ngff_unit_invalid() -> None:
    """Test that invalid NGFF units raise ValueError with helpful message."""
    with pytest.raises(ValueError, match=r"Invalid NGFF unit: 'um'.*Valid spatial"):
        validate_ngff_unit("um")

    with pytest.raises(ValueError, match=r"Invalid NGFF unit: 'ms'.*Valid temporal"):
        validate_ngff_unit("ms")


def test_ngff_length_mapping_completeness() -> None:
    """Test that all spatial unit mappings are present and valid."""
    # Verify we have a reasonable number of spatial units
    assert len(_NGFF_TO_OME_LENGTH) >= 20

    # Verify all values are non-empty strings
    assert all(isinstance(k, str) and k for k in _NGFF_TO_OME_LENGTH.keys())
    assert all(isinstance(v, str) and v for v in _NGFF_TO_OME_LENGTH.values())


def test_ngff_time_mapping_completeness() -> None:
    """Test that all temporal unit mappings are present and valid."""
    # Verify we have a reasonable number of temporal units
    assert len(_NGFF_TO_OME_TIME) >= 20

    # Verify all values are non-empty strings
    assert all(isinstance(k, str) and k for k in _NGFF_TO_OME_TIME.keys())
    assert all(isinstance(v, str) and v for v in _NGFF_TO_OME_TIME.values())
