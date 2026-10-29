"""Synthetic dataset generation for the machining recommendation system.

This module creates a realistic dataset of machining scenarios with a target label
for the recommended tool. The generated records are intentionally interpretable and
narrowly tailored to the engineering schema defined in src.data.schema.
"""

from __future__ import annotations

from random import Random
from typing import Iterable

import pandas as pd

from src.data.schema import (
    MATERIAL_CATEGORIES,
    MATERIAL_GRADES,
    MACHINE_TYPES,
    OPERATIONS,
    STAGES,
    TARGET_LABEL,
    HARDNESS_LEVELS,
    WORKPIECE_SHAPES,
    SURFACE_FINISHES,
    TOLERANCES,
    ACCURACY_REQUIREMENTS,
    PREFERENCES,
    DATASET_COLUMNS,
)


TOOL_LIBRARY: dict[str, list[str]] = {
    "Turning": ["CNMG 120408", "DNMG 150608", "TNMG 160408", "VCMT 160404"],
    "Facing": ["CNMG 120408", "TNMG 160408", "VCMT 160404"],
    "Drilling": ["HSS Twist Drill 10mm", "Carbide Drill 8mm", "Solid Carbide Drill 12mm"],
    "Milling": ["4 Flute End Mill 10mm", "5 Flute End Mill 8mm", "Ball Nose 6mm"],
    "Slot Milling": ["Slitting Cutter 3mm", "End Mill 6mm", "Keyseat Mill 5mm"],
    "End Milling": ["End Mill 10mm", "End Mill 8mm", "Carbide End Mill 12mm"],
    "Side Milling": ["Side Milling Cutter 20mm", "Shell Mill 25mm", "Face Mill 40mm"],
    "Boring": ["Boring Bar 20mm", "Fine Boring Tool 12mm", "Carbide Boring Bar 16mm"],
    "Threading": ["Threading Insert 60°", "Tap M10", "Thread Mill M12"],
    "Grooving": ["Grooving Insert 2mm", "Parting Insert 3mm", "Grooving Tool 4mm"],
    "Parting": ["Parting Tool 3mm", "Parting Insert 4mm", "Cutoff Tool 5mm"],
    "Chamfering": ["Chamfer Cutter 45°", "Deburr Tool 2mm", "Countersink 90°"],
    "Tapping": ["Tap M8", "Tap M10", "M12 Tap"],
}


def _material_family_and_grade(rng: Random) -> tuple[str, str]:
    family = rng.choice(MATERIAL_CATEGORIES)
    grade = rng.choice(MATERIAL_GRADES[family])
    return family, grade


def _hardness_value(hardness_level: str, rng: Random) -> tuple[int, str]:
    mapping = {
        "Soft": (80, 180),
        "Medium": (180, 260),
        "Hard": (260, 380),
        "Very Hard": (380, 520),
    }
    lower, upper = mapping[hardness_level]
    value = rng.randint(lower, upper)
    unit = rng.choice(["HB", "HRC"])
    return value, unit


def _workpiece_dimensions(shape: str, rng: Random) -> dict[str, float]:
    size_map = {
        "Round Bar": {"diameter_mm": rng.randint(25, 160), "length_mm": rng.randint(80, 700)},
        "Square Bar": {"width_mm": rng.randint(20, 150), "length_mm": rng.randint(80, 700)},
        "Plate": {
            "length_mm": rng.randint(100, 600),
            "width_mm": rng.randint(80, 400),
            "thickness_mm": rng.randint(5, 80),
        },
        "Pipe": {
            "diameter_mm": rng.randint(30, 220),
            "length_mm": rng.randint(100, 700),
            "thickness_mm": rng.randint(3, 40),
        },
        "Block": {
            "length_mm": rng.randint(50, 300),
            "width_mm": rng.randint(40, 250),
            "height_mm": rng.randint(30, 200),
        },
        "Forged Part": {
            "length_mm": rng.randint(60, 400),
            "width_mm": rng.randint(50, 240),
            "thickness_mm": rng.randint(10, 100),
        },
        "Casting": {
            "length_mm": rng.randint(80, 500),
            "width_mm": rng.randint(60, 260),
            "thickness_mm": rng.randint(12, 90),
        },
    }
    return size_map[shape]


def _recommended_tool(operation: str, hardness_level: str, material_category: str) -> str:
    options = TOOL_LIBRARY.get(operation, ["General Purpose Tool"])
    preference = ""
    if hardness_level in {"Hard", "Very Hard"}:
        preference = "carbide"
    elif material_category in {"Aluminium", "Plastic"}:
        preference = "high-speed"
    else:
        preference = "general"
    candidates = options[:]
    if preference == "carbide":
        candidates = [tool for tool in candidates if "Carbide" in tool or "Insert" in tool]
    if not candidates:
        candidates = options
    return candidates[0]


def _build_record(row_idx: int, rng: Random) -> dict[str, object]:
    material_category, material_grade = _material_family_and_grade(rng)
    hardness_level = rng.choice(HARDNESS_LEVELS)
    hardness_value, hardness_unit = _hardness_value(hardness_level, rng)
    workpiece_shape = rng.choice(WORKPIECE_SHAPES)
    dimensions = _workpiece_dimensions(workpiece_shape, rng)
    operation = rng.choice(OPERATIONS)
    stage = rng.choice(STAGES)
    machine_type = rng.choice(MACHINE_TYPES)
    surface_finish = rng.choice(SURFACE_FINISHES)
    tolerance = rng.choice(TOLERANCES)
    accuracy_requirement = rng.choice(ACCURACY_REQUIREMENTS)
    preference_flags = {
        "preference_lower_tool_cost": bool(rng.randint(0, 1)),
        "preference_longer_tool_life": bool(rng.randint(0, 1)),
        "preference_better_surface_finish": bool(rng.randint(0, 1)),
        "preference_higher_productivity": bool(rng.randint(0, 1)),
        "preference_lower_machining_time": bool(rng.randint(0, 1)),
    }

    record: dict[str, object] = {
        "material_category": material_category,
        "material_grade": material_grade,
        "hardness_level": hardness_level,
        "hardness_value": hardness_value,
        "hardness_unit": hardness_unit,
        "workpiece_shape": workpiece_shape,
        "machining_operation": operation,
        "machining_stage": stage,
        "machine_type": machine_type,
        "machine_power_kw": round(rng.uniform(3.5, 45.0), 2),
        "max_rpm": rng.randint(1000, 18000),
        "machine_rigidity": rng.choice(["Low", "Medium", "High"]),
        "feed_rate_value": round(rng.uniform(0.08, 3.50), 3),
        "feed_rate_unit": "mm/rev",
        "spindle_speed_rpm": rng.randint(500, 20000),
        "cutting_speed_m_min": round(rng.uniform(20.0, 500.0), 2),
        "depth_of_cut_mm": round(rng.uniform(0.5, 12.0), 2),
        "width_of_cut_mm": round(rng.uniform(1.0, 70.0), 2),
        "step_over_mm": round(rng.uniform(0.5, 12.0), 2),
        "surface_finish": surface_finish,
        "tolerance": tolerance,
        "accuracy_requirement": accuracy_requirement,
        **preference_flags,
        "recommended_tool": _recommended_tool(operation, hardness_level, material_category),
    }

    for key, value in dimensions.items():
        record[key] = value
    return record


def generate_synthetic_dataset(n_rows: int = 1000, seed: int | None = None) -> pd.DataFrame:
    """Return a synthetic machining dataset with engineering-style schema coverage."""
    if n_rows <= 0:
        raise ValueError("n_rows must be a positive integer.")

    rng = Random(seed)
    rows = [_build_record(index, rng) for index in range(n_rows)]
    df = pd.DataFrame(rows)

    for column in DATASET_COLUMNS:
        if column not in df.columns:
            df[column] = None

    return df[DATASET_COLUMNS]


def validate_generated_dataset(df: pd.DataFrame) -> bool:
    """Validate the generated dataset against the canonical schema contract."""
    if df.empty:
        return False

    required = {
        "material_category",
        "material_grade",
        "hardness_level",
        "hardness_value",
        "hardness_unit",
        "workpiece_shape",
        "machining_operation",
        "machining_stage",
        "machine_type",
        "surface_finish",
        "tolerance",
        "accuracy_requirement",
        TARGET_LABEL,
    }
    if not required.issubset(df.columns):
        return False

    if not df["material_category"].isin(MATERIAL_CATEGORIES).all():
        return False

    grade_ok = df.apply(
        lambda row: row["material_grade"] in MATERIAL_GRADES.get(row["material_category"], []),
        axis=1,
    )
    if not grade_ok.all():
        return False

    if not df["hardness_level"].isin(HARDNESS_LEVELS).all():
        return False

    if not df["machining_operation"].isin(OPERATIONS).all():
        return False

    if not df["machining_stage"].isin(STAGES).all():
        return False

    if not df["machine_type"].isin(MACHINE_TYPES).all():
        return False

    if not df[TARGET_LABEL].notna().all():
        return False

    return True
