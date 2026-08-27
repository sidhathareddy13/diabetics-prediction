from functools import lru_cache
from pathlib import Path
import json

import joblib
import pandas as pd

from backend.auth import category_color, risk_category_from_score
from ml.config import (
    FEATURES,
    MODEL_FILENAME,
    MODEL_NAME,
    THRESHOLD_METADATA_FILENAME,
    binary_class_from_probability,
    binary_class_label,
)

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "models" / MODEL_FILENAME
METADATA_PATH = BASE_DIR / "models" / THRESHOLD_METADATA_FILENAME


@lru_cache(maxsize=1)
def get_model():
    """Load the deployed model once per application process."""
    return joblib.load(MODEL_PATH)


@lru_cache(maxsize=1)
def get_model_metadata():
    """Load training metadata, including the OOF-selected threshold."""
    with METADATA_PATH.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def get_classification_threshold() -> float:
    """Return the threshold selected during training by balanced-accuracy optimization."""
    metadata = get_model_metadata()
    threshold = metadata.get("classification_threshold")
    if threshold is None:
        raise RuntimeError("Model metadata does not contain a classification threshold.")
    return float(threshold)


def predict_risk(glucose, blood_pressure, bmi, age, insulin):
    model = get_model()
    threshold = get_classification_threshold()
    df = pd.DataFrame(
        [[glucose, blood_pressure, bmi, age, insulin]],
        columns=FEATURES,
    )
    for col in ["Glucose", "BloodPressure", "BMI", "Insulin"]:
        if df[col].iloc[0] == 0:
            df.at[0, col] = float("nan")

    probability = float(model.predict_proba(df)[0, 1])
    score = probability * 100.0
    predicted_class = binary_class_from_probability(probability, threshold)
    category = risk_category_from_score(score)

    return {
        "probability": probability,
        "score": score,
        "category": category,
        "category_color": category_color(category),
        "predicted_class": predicted_class,
        "classification": binary_class_label(predicted_class),
        "classification_threshold": threshold,
        "threshold_strategy": get_model_metadata().get(
            "threshold_strategy", "balanced_accuracy_on_oof_predictions"
        ),
        "model_name": MODEL_NAME,
    }
