from datetime import datetime


def normalize_reminder_status(value):
    if value is None:
        return 'UPCOMING'
    value_str = str(value).strip().upper()
    if value_str in {'UPCOMING', 'COMPLETED', 'OVERDUE'}:
        return value_str
    if value_str == 'PENDING':
        return 'UPCOMING'
    return 'UPCOMING'


def compute_reminder_status(reminder):
    current_status = normalize_reminder_status(reminder.get('status'))
    if current_status == 'COMPLETED':
        return 'COMPLETED'

    try:
        scheduled = datetime.strptime(f"{reminder['date']} {reminder['time']}", '%Y-%m-%d %H:%M')
    except (TypeError, ValueError):
        return current_status or 'UPCOMING'

    if scheduled > datetime.now():
        return 'UPCOMING'
    return 'OVERDUE'


def get_recent_prediction_tips(predictions):
    if not predictions:
        return [
            'Start with a balanced routine: maintain regular meals, hydration, and daily movement.',
            'Use this assessment as a screening check and discuss persistent health concerns with a professional.'
        ]

    latest = predictions[0]
    tips = []

    if latest['glucose'] >= 140:
        tips.append('Your recent glucose readings are elevated. Focus on reducing sugary drinks and refined carbohydrates, and discuss persistent changes with a healthcare professional.')
    elif latest['glucose'] >= 100:
        tips.append('Your recent glucose is moderately elevated. Keep meals balanced and stay consistent with hydration and activity.')
    else:
        tips.append('Your recent glucose is stable. Continue healthy routines and keep monitoring over time.')

    if latest['bmi'] >= 30:
        tips.append('Your recent BMI trend suggests a need for balanced eating and regular physical activity to support long-term wellness.')
    elif latest['bmi'] >= 25:
        tips.append('Your BMI is trending upward. Small, sustainable food and movement changes can help maintain balance.')
    else:
        tips.append('Your BMI is in a healthier range. Maintain routine movement and balanced meals to keep it stable.')

    if latest['risk_score'] >= 70:
        tips.append('Your most recent risk score is high. Focus on consistency in diet, movement, and follow-up with a healthcare professional if needed.')
    elif latest['risk_score'] >= 40:
        tips.append('Your most recent risk score is moderate. Preventive habits and routine check-ins can help improve the trend.')
    else:
        tips.append('Your recent risk score is encouraging. Keep following your healthy routine and continue monitoring progress.')

    if len(predictions) >= 2:
        previous = predictions[1]
        if latest['glucose'] > previous['glucose'] and latest['risk_score'] > previous['risk_score']:
            tips.append('Your recent glucose and risk levels are climbing compared with your previous assessment. Consider a more structured wellness plan and routine follow-up.')
        elif latest['risk_score'] < previous['risk_score']:
            tips.append('Your recent risk trend is improving. Keep the same healthy habits and continue consistent follow-up.')

    deduped = []
    for tip in tips:
        if tip not in deduped:
            deduped.append(tip)
    return deduped[:4]
