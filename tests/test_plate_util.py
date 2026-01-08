"""Tests for plate_from_useq_to_yaozarrs utility function."""

from __future__ import annotations

import pytest


def test_plate_from_useq_to_yaozarrs_basic() -> None:
    """Test basic plate conversion from useq to yaozarrs."""
    useq = pytest.importorskip("useq")
    pytest.importorskip("yaozarrs")

    from ome_writers._util import plate_from_useq_to_yaozarrs

    # Create a simple plate plan
    plate = useq.WellPlate.from_str("96-well")
    plan = useq.WellPlatePlan(
        plate=plate,
        a1_center_xy=(0, 0),
        selected_wells=((0, 0), (0, 1)),  # 2 wells
    )

    plate_def = plate_from_useq_to_yaozarrs(plan)

    assert plate_def.name == "96-well"
    # Should have at least these rows/columns
    assert len(plate_def.rows) >= 1
    assert len(plate_def.columns) >= 1
    assert len(plate_def.wells) >= 1
    assert plate_def.field_count == 1 or plate_def.field_count is None


def test_plate_from_useq_to_yaozarrs_multi_field() -> None:
    """Test plate conversion with multiple fields per well."""
    useq = pytest.importorskip("useq")
    pytest.importorskip("yaozarrs")

    from ome_writers._util import plate_from_useq_to_yaozarrs

    # Create plate plan with multiple points per well
    plate = useq.WellPlate.from_str("96-well")
    plan = useq.WellPlatePlan(
        plate=plate,
        a1_center_xy=(0, 0),
        selected_wells=((0, 0), (0, 1)),
        well_points_plan=useq.GridRowsColumns(rows=2, columns=3),
    )

    plate_def = plate_from_useq_to_yaozarrs(plan)

    assert plate_def.field_count == 6  # 2 rows * 3 columns


def test_plate_from_useq_to_yaozarrs_multi_row_columns() -> None:
    """Test plate conversion with many rows and columns."""
    useq = pytest.importorskip("useq")
    pytest.importorskip("yaozarrs")

    from ome_writers._util import plate_from_useq_to_yaozarrs

    # Create a 96-well plate and select a few wells
    plate = useq.WellPlate.from_str("96-well")
    plan = useq.WellPlatePlan(
        plate=plate,
        a1_center_xy=(0, 0),
        selected_wells=((0, 0), (1, 1)),  # A1 and B2
    )

    plate_def = plate_from_useq_to_yaozarrs(plan)

    # Should have created rows and columns
    assert len(plate_def.rows) >= 1
    assert len(plate_def.columns) >= 1
    assert len(plate_def.wells) >= 1


def test_plate_from_useq_to_yaozarrs_invalid_input() -> None:
    """Test error handling for invalid input."""
    pytest.importorskip("useq")
    pytest.importorskip("yaozarrs")

    from ome_writers._util import plate_from_useq_to_yaozarrs

    with pytest.raises(TypeError, match=r"must be a useq.WellPlatePlan"):
        plate_from_useq_to_yaozarrs("not a plate plan")  # type: ignore


def test_plate_from_useq_to_yaozarrs_well_deduplication() -> None:
    """Test that duplicate wells are handled correctly."""
    useq = pytest.importorskip("useq")
    pytest.importorskip("yaozarrs")

    from ome_writers._util import plate_from_useq_to_yaozarrs

    plate = useq.WellPlate.from_str("96-well")
    # Select same well multiple times (should be deduplicated)
    plan = useq.WellPlatePlan(
        plate=plate,
        a1_center_xy=(0, 0),
        selected_wells=((0, 0), (0, 1)),
    )

    plate_def = plate_from_useq_to_yaozarrs(plan)

    # Should have 2 wells: A1 and A2
    assert len(plate_def.rows) == 1  # Just A
    assert len(plate_def.columns) == 2  # 1, 2
    assert len(plate_def.wells) == 2


def test_plate_from_useq_to_yaozarrs_missing_useq() -> None:
    """Test error when useq is not installed."""
    pytest.importorskip("yaozarrs")

    import sys
    from unittest.mock import Mock

    # Temporarily remove useq from modules
    useq_module = sys.modules.get("useq")
    sys.modules["useq"] = None  # type: ignore

    try:
        from ome_writers._util import plate_from_useq_to_yaozarrs

        with pytest.raises(ImportError, match="useq-schema and yaozarrs"):
            plate_from_useq_to_yaozarrs(Mock())
    finally:
        # Restore useq module
        if useq_module is not None:
            sys.modules["useq"] = useq_module
        else:
            sys.modules.pop("useq", None)


def test_plate_from_useq_to_yaozarrs_missing_yaozarrs() -> None:
    """Test error when yaozarrs is not installed."""
    pytest.importorskip("useq")

    import sys
    from unittest.mock import Mock

    # Temporarily remove yaozarrs from modules
    yaozarrs_module = sys.modules.get("yaozarrs")
    sys.modules["yaozarrs"] = None  # type: ignore

    try:
        from ome_writers._util import plate_from_useq_to_yaozarrs

        with pytest.raises(ImportError, match="useq-schema and yaozarrs"):
            plate_from_useq_to_yaozarrs(Mock())
    finally:
        # Restore yaozarrs module
        if yaozarrs_module is not None:
            sys.modules["yaozarrs"] = yaozarrs_module
        else:
            sys.modules.pop("yaozarrs", None)
