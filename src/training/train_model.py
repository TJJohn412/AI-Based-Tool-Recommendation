from __future__ import annotations

import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split

from src.features.preprocess import prepare_training_data

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATASET_PATH = PROJECT_ROOT / "dataset" / "tool_dataset.csv"
MODEL_PATH = PROJECT_ROOT / "models" / "tool_recommendation_model.joblib"
METADATA_PATH = PROJECT_ROOT / "models" / "tool_recommendation_metadata.json"


def train_and_save_model(data_path: str | Path = DATASET_PATH) -> dict[str, object]:
    """Train the tool recommendation model and save the artifact plus metadata."""
    dataset = pd.read_csv(data_path)
    features, target, feature_names = prepare_training_data(dataset)

    X_train, X_test, y_train, y_test = train_test_split(
        features,
        target,
        test_size=0.2,
        random_state=42,
        stratify=target,
    )

    model = RandomForestClassifier(
        n_estimators=220,
        random_state=42,
        class_weight="balanced_subsample",
        min_samples_leaf=1,
    )
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    report = classification_report(y_test, predictions, output_dict=True, zero_division=0)

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_PATH)

    metadata = {
        "feature_names": feature_names,
        "accuracy": float(accuracy),
        "dataset_path": str(Path(data_path).resolve()),
        "report": report,
    }
    METADATA_PATH.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    return {
        "accuracy": float(accuracy),
        "model_path": str(MODEL_PATH),
        "feature_count": len(feature_names),
    }


if __name__ == "__main__":
    result = train_and_save_model()
    print(f"Model saved to {result['model_path']}")
    print(f"Accuracy: {result['accuracy']:.4f}")
