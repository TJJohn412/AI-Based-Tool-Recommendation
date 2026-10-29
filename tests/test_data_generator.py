from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.generator import generate_synthetic_dataset, validate_generated_dataset
from src.data.schema import (
    MATERIAL_CATEGORIES,
    MACHINE_TYPES,
    OPERATIONS,
    TARGET_LABEL,
)


def test_generate_synthetic_dataset_returns_valid_rows() -> None:
    df = generate_synthetic_dataset(n_rows=250, seed=42)

    assert len(df) == 250
    assert set(df.columns) >= {
        "material_category",
        "material_grade",
        "hardness_level",
        "machining_operation",
        "machine_type",
        "recommended_tool",
    }
    assert df["material_category"].isin(MATERIAL_CATEGORIES).all()
    assert df["machine_type"].isin(MACHINE_TYPES).all()
    assert df["machining_operation"].isin(OPERATIONS).all()
    assert df[TARGET_LABEL].notna().all()


def test_validate_generated_dataset_accepts_good_data() -> None:
    df = generate_synthetic_dataset(n_rows=100, seed=7)
    assert validate_generated_dataset(df) is True


def test_validate_generated_dataset_rejects_invalid_schema() -> None:
    df = generate_synthetic_dataset(n_rows=20, seed=9)
    df.loc[0, "material_category"] = "Unknown Material"
    df.loc[0, "machine_type"] = "Unknown Machine"

    assert validate_generated_dataset(df) is False
