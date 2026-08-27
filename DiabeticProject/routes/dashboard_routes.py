from flask import render_template, session

from backend.auth import get_user, login_required
from backend.analytics import generate_risk_trend_svg
from backend.reminders import get_recent_prediction_tips
from ml.data_processing import latest_prediction_for_user, user_predictions


def register_dashboard_routes(app):
    @app.route('/dashboard')
    @login_required
    def dashboard():
        user_id = session['user_id']
        predictions = user_predictions(user_id, limit=10)
        latest = latest_prediction_for_user(user_id)
        trend_svg = generate_risk_trend_svg(predictions[:8])
        tips = get_recent_prediction_tips(predictions)
        return render_template('dashboard.html', user=get_user(user_id), latest=latest, trend_svg=trend_svg, predictions=predictions, quick_tips=tips)
