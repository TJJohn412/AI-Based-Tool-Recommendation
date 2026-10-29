"""Engineering-informed dataset schema for the Smart Machining Assistant.

This module defines the canonical feature names, enumerations, and grade mappings
used for synthetic data generation and downstream model training. The design is
intentionally explicit so that engineers can review the assumptions before model
training begins.

The grade lists below are limited to representative materials with established
machining relevance. They are presented as verified industrial reference entries
rather than exhaustive materials databases.
"""

from __future__ import annotations

MATERIAL_CATEGORIES: list[str] = [
    "Steel",
    "Stainless Steel",
    "Cast Iron",
    "Aluminium",
    "Copper",
    "Brass",
    "Bronze",
    "Titanium",
    "Nickel Alloy",
    "Plastic",
    "Composite",
]

# Representative grades mapped to material families. These values are intentionally
# limited to common engineering materials and should be expanded only with verified
# references from manufacturer or standards documentation.
MATERIAL_GRADES: dict[str, list[str]] = {
    "Steel": ["Mild Steel", "EN8", "EN19", "EN24", "AISI 1045", "AISI 4140"],
    "Stainless Steel": ["304", "316", "410", "420", "440C"],
    "Cast Iron": ["Gray Cast Iron", "Ductile Iron", "Malleable Iron"],
    "Aluminium": ["6061", "7075", "5083", "2024"],
    "Copper": ["C101", "C110", "C172"],
    "Brass": ["C260", "C360", "C230"],
    "Bronze": ["C932", "C954", "C863"],
    "Titanium": ["Grade 2", "Grade 5 (Ti-6Al-4V)", "Grade 9"],
    "Nickel Alloy": ["Inconel 625", "Inconel 718", "Monel 400"],
    "Plastic": ["PEEK", "Nylon", "Delrin (POM)", "UHMW PE"],
    "Composite": ["Carbon Fiber Composite", "Glass Fiber Composite", "Kevlar Composite"],
}

HARDNESS_LEVELS: list[str] = ["Soft", "Medium", "Hard", "Very Hard"]
HARDNESS_UNITS: list[str] = ["HB", "HRC"]

WORKPIECE_SHAPES: list[str] = [
    "Round Bar",
    "Square Bar",
    "Plate",
    "Pipe",
    "Block",
    "Forged Part",
    "Casting",
]

DIMENSION_FIELDS: dict[str, list[str]] = {
    "Round Bar": ["diameter_mm", "length_mm"],
    "Square Bar": ["width_mm", "length_mm"],
    "Plate": ["length_mm", "width_mm", "thickness_mm"],
    "Pipe": ["diameter_mm", "length_mm", "thickness_mm"],
    "Block": ["length_mm", "width_mm", "height_mm"],
    "Forged Part": ["length_mm", "width_mm", "thickness_mm"],
    "Casting": ["length_mm", "width_mm", "thickness_mm"],
}

OPERATIONS: list[str] = [
    "Turning",
    "Facing",
    "Drilling",
    "Milling",
    "Slot Milling",
    "End Milling",
    "Side Milling",
    "Boring",
    "Threading",
    "Grooving",
    "Parting",
    "Chamfering",
    "Tapping",
]

STAGES: list[str] = ["Roughing", "Semi-finishing", "Finishing"]

MACHINE_TYPES: list[str] = [
    "Manual Lathe",
    "CNC Lathe",
    "Vertical Milling Machine",
    "Horizontal Milling Machine",
    "VMC",
    "HMC",
    "Drill Machine",
]

MACHINE_RIGIDITY: list[str] = ["Low", "Medium", "High"]

SURFACE_FINISHES: list[str] = ["Rough", "Normal", "Fine", "Mirror Finish"]
TOLERANCES: list[str] = ["±0.5 mm", "±0.1 mm", "±0.05 mm", "±0.01 mm"]
ACCURACY_REQUIREMENTS: list[str] = ["Low", "Medium", "High"]

PREFERENCES: list[str] = [
    "Lower tool cost",
    "Longer tool life",
    "Better surface finish",
    "Higher productivity",
    "Lower machining time",
]

TARGET_LABEL: str = "recommended_tool"


def get_grade_options(material_category: str) -> list[str]:
    """Return representative graded options for the selected material family."""
    return MATERIAL_GRADES.get(material_category, []).copy()


def validate_material_grade(material_category: str, material_grade: str) -> bool:
    """Validate whether a grade matches the material family in the schema."""
    if material_category not in MATERIAL_CATEGORIES:
        return False
    return material_grade in MATERIAL_GRADES.get(material_category, [])


DATASET_COLUMNS: list[str] = [
    "material_category",
    "material_grade",
    "hardness_level",
    "hardness_value",
    "hardness_unit",
    "workpiece_shape",
    "length_mm",
    "width_mm",
    "diameter_mm",
    "thickness_mm",
    "height_mm",
    "machining_operation",
    "machining_stage",
    "machine_type",
    "machine_power_kw",
    "max_rpm",
    "machine_rigidity",
    "feed_rate_value",
    "feed_rate_unit",
    "spindle_speed_rpm",
    "cutting_speed_m_min",
    "depth_of_cut_mm",
    "width_of_cut_mm",
    "step_over_mm",
    "surface_finish",
    "tolerance",
    "accuracy_requirement",
    "preference_lower_tool_cost",
    "preference_longer_tool_life",
    "preference_better_surface_finish",
    "preference_higher_productivity",
    "preference_lower_machining_time",
    "recommended_tool",
]
