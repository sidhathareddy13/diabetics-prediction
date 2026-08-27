"""Central configuration for the deployed diabetes-risk ML model."""

MODEL_NAME = "XGBoost"
MODEL_FILENAME = "balanced_best_diabetes_pipeline.pkl"
THRESHOLD_METADATA_FILENAME = "model_metadata.json"

# The threshold is NOT hard-coded here. Training selects it from out-of-fold
# predictions by maximizing balanced accuracy and saves the selected value in
# models/model_metadata.json. Inference reads that artifact.

LOW_RISK_MAX = 39.0
MODERATE_RISK_MAX = 69.0

FEATURES = ["Glucose", "BloodPressure", "BMI", "Age", "Insulin"]


def risk_category_from_score(score: float) -> str:
    """Map a 0-100 probability score to the app's display category."""
    score = float(score)
    if score <= LOW_RISK_MAX:
        return "LOW RISK"
    if score <= MODERATE_RISK_MAX:
        return "MODERATE RISK"
    return "HIGH RISK"


def binary_class_from_probability(probability: float, threshold: float) -> int:
    """Convert probability to a binary class using the trained threshold."""
    return int(float(probability) >= float(threshold))


def binary_class_label(predicted_class: int) -> str:
    return "AT RISK" if int(predicted_class) == 1 else "LOWER RISK"
