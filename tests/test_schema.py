from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.schema import (
    MATERIAL_CATEGORIES,
    HARDNESS_LEVELS,
    MACHINE_TYPES,
    OPERATIONS,
    STAGES,
    TARGET_LABEL,
    get_grade_options,
    validate_material_grade,
)


def test_material_categories_include_key_alloys() -> None:
    assert "Steel" in MATERIAL_CATEGORIES
    assert "Aluminium" in MATERIAL_CATEGORIES
    assert "Titanium" in MATERIAL_CATEGORIES


def test_steel_grades_are_verified_and_available() -> None:
    grades = get_grade_options("Steel")
    assert "EN8" in grades
    assert "AISI 4140" in grades
    assert "Mild Steel" in grades


def test_validate_material_grade_rejects_unknown_grade() -> None:
    assert validate_material_grade("Steel", "EN8") is True
    assert validate_material_grade("Steel", "Unknown Grade") is False


def test_machine_and_operation_metadata_is_present() -> None:
    assert "CNC Lathe" in MACHINE_TYPES
    assert "Turning" in OPERATIONS
    assert "Roughing" in STAGES
    assert TARGET_LABEL == "recommended_tool"


def test_hardness_levels_cover_the_expected_range() -> None:
    assert "Soft" in HARDNESS_LEVELS
    assert "Medium" in HARDNESS_LEVELS
    assert "Hard" in HARDNESS_LEVELS
    assert "Very Hard" in HARDNESS_LEVELS
