# Diabetes Risk Predictor — Deployment ML Report

## 1. Deployed model

The deployed artifact is `models/balanced_best_diabetes_pipeline.pkl`. Inspection of the serialized pipeline shows:

- `SimpleImputer(strategy="median")`
- `XGBClassifier`
- Five input features: Glucose, BloodPressure, BMI, Age, Insulin

Therefore the deployed model is **XGBoost**. The older experiment report named Random Forest as the CV-selected model, which did not match the artifact being served. This deployment report corrects that naming inconsistency.

## 2. Model configuration

The XGBoost artifact uses the best XGBoost parameters recorded in the project experiment comparison:

- n_estimators: 100
- max_depth: 3
- learning_rate: 0.05
- subsample: 0.9
- colsample_bytree: 1.0
- min_child_weight: 3
- gamma: 0
- reg_alpha: 0.1
- reg_lambda: 1.0

## 3. Classification threshold

The application does not hard-code a probability such as 0.35. During training, candidate thresholds are evaluated on out-of-fold predictions and the threshold that maximizes **Balanced Accuracy** is selected. That selected value is saved in `models/model_metadata.json` and loaded by the prediction service.

This binary classification is kept separate from the application's display categories.

## 4. Application display categories

The application uses one centralized rule:

- 0–39% → LOW RISK
- 40–69% → MODERATE RISK
- 70–100% → HIGH RISK

These are **application display bands**, not clinically validated diagnostic thresholds.

## 5. Important evaluation note

The old report's headline test metrics (ROC-AUC 0.8605, etc.) belonged to the Random Forest model that it named as the selected model. They must not be presented as XGBoost deployment metrics. The experiment comparison already records XGBoost separately (CV ROC-AUC 0.8242 and Test ROC-AUC 0.8632 in the project comparison table). If formal XGBoost threshold-optimization metrics are needed for a report, they should be regenerated from the exact training environment and locked XGBoost artifact rather than copied from the Random Forest section.

## 6. Dependency consistency

`requirements.txt` now includes XGBoost because the deployed pickle directly depends on `XGBClassifier`. The scikit-learn version remains pinned to the version recorded by the serialized pipeline.

## 7. Safety wording

The application describes the output as educational/risk screening and does not present it as a diagnosis. Recommendations are framed as general wellness information and encourage professional interpretation where appropriate.
