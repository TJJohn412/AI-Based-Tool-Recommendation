from __future__ import annotations

from typing import Any

import pandas as pd

CATEGORICAL_COLUMNS = [
    "material_category",
    "material_grade",
    "hardness_level",
    "hardness_unit",
    "workpiece_shape",
    "machining_operation",
    "machining_stage",
    "machine_type",
    "machine_rigidity",
    "feed_rate_unit",
    "surface_finish",
    "tolerance",
    "accuracy_requirement",
]

BOOLEAN_COLUMNS = [
    "preference_lower_tool_cost",
    "preference_longer_tool_life",
    "preference_better_surface_finish",
    "preference_higher_productivity",
    "preference_lower_machining_time",
]

NUMERIC_COLUMNS = [
    "hardness_value",
    "length_mm",
    "width_mm",
    "diameter_mm",
    "thickness_mm",
    "height_mm",
    "machine_power_kw",
    "max_rpm",
    "feed_rate_value",
    "spindle_speed_rpm",
    "cutting_speed_m_min",
    "depth_of_cut_mm",
    "width_of_cut_mm",
    "step_over_mm",
]

FEATURE_COLUMNS = CATEGORICAL_COLUMNS + NUMERIC_COLUMNS + BOOLEAN_COLUMNS


def _prepare_row(frame: pd.DataFrame) -> pd.DataFrame:
    row = frame.copy()
    for column in NUMERIC_COLUMNS:
        row[column] = pd.to_numeric(row[column], errors="coerce")
    for column in BOOLEAN_COLUMNS:
        row[column] = row[column].fillna(False).astype(bool)
    for column in CATEGORICAL_COLUMNS:
        row[column] = row[column].fillna("Unknown")
    return row


def prepare_training_data(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series, list[str]]:
    """Return encoded feature matrix, target, and feature names for model training."""
    working = df.copy()
    working = _prepare_row(working)

    target = working["recommended_tool"].copy()
    features = working[FEATURE_COLUMNS].copy()
    encoded = pd.get_dummies(features, columns=CATEGORICAL_COLUMNS, dummy_na=False)
    encoded = encoded.fillna(0)
    encoded = encoded.astype(float)
    return encoded, target, list(encoded.columns)


def payload_to_feature_frame(payload: dict[str, Any], feature_names: list[str] | None = None) -> pd.DataFrame:
    """Convert one payload dict into the feature matrix expected by the trained model."""
    row = {column: payload.get(column) for column in FEATURE_COLUMNS}
    frame = pd.DataFrame([row])
    frame = _prepare_row(frame)
    encoded = pd.get_dummies(frame[FEATURE_COLUMNS], columns=CATEGORICAL_COLUMNS, dummy_na=False)
    encoded = encoded.fillna(0)
    encoded = encoded.astype(float)

    if feature_names is not None:
        missing = [name for name in feature_names if name not in encoded.columns]
        if missing:
            missing_df = pd.DataFrame({name: 0.0 for name in missing}, index=encoded.index)
            encoded = pd.concat([encoded, missing_df], axis=1)
        encoded = encoded[feature_names]
    return encoded
