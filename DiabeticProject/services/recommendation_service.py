from ml.recommendations import build_recommendations


def get_recommendations(glucose, blood_pressure, bmi, age, insulin, risk_score, risk_category):
    return build_recommendations(glucose, blood_pressure, bmi, age, insulin, risk_score, risk_category)
