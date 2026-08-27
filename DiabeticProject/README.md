# Diabetes Risk Predictor

A modern healthcare-focused web app built around the existing diabetes risk ML pipeline in this project.

## Project overview

This project preserves the active trained model at `models/balanced_best_diabetes_pipeline.pkl` and wraps it in a full web application with:

- Login and sign-up flows
- Dashboard with latest prediction and risk trend
- Prediction form using the five required inputs
- Personalized recommendations after prediction
- Analytics with real user data
- Reminders management
- History tracking
- About page
- PDF report downloads

## Deployed ML model

The application deploys the **XGBoost** pipeline stored at:

- `models/balanced_best_diabetes_pipeline.pkl`

The serialized pipeline contains median imputation followed by `XGBClassifier`. The deployed model is therefore **XGBoost**, not the Random Forest named in the older experiment report. The old report was inconsistent with the actual artifact and should not be used as the deployment description.

The application keeps two concepts separate:

- **Binary model classification threshold:** selected during training by maximizing balanced accuracy on out-of-fold predictions and stored in `models/model_metadata.json`.
- **Application display bands:** 0–39% Low, 40–69% Moderate, 70–100% High. These are display bands, not clinical diagnostic thresholds.

The model is loaded once per application process and then reused for predictions.

The prediction logic accepts the following five inputs only:

- Glucose
- Blood Pressure
- BMI
- Age
- Insulin

## Run locally

1. Create and activate a virtual environment.
2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Start the app:

```bash
python app.py
```

4. Open the app in a browser:

```text
http://localhost:5000
```

## Default behavior

- User accounts are stored in SQLite.
- Prediction data is stored separately per user.
- Recommendations are generated from actual input values and risk output.
- No fake data is used for analytics or dashboards after the app is running.

## Disclaimer

This tool is for educational and risk-screening purposes only. It does not replace professional medical diagnosis, treatment, or advice.
