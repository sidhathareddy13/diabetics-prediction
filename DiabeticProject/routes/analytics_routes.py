from flask import render_template, request, session

from backend.analytics import build_analytics_insights, data_line_chart_svg, date_filter_range, risk_distribution_chart_svg
from backend.auth import get_user, login_required
from ml.data_processing import user_predictions


def register_analytics_routes(app):
    @app.route('/analytics')
    @login_required
    def analytics():
        user_id = session['user_id']
        range_name = request.args.get('range', '30')
        start_date = request.args.get('start')
        end_date = request.args.get('end')
        start_dt, end_dt = date_filter_range(range_name, start_date, end_date)
        predictions = user_predictions(user_id, start_date=start_dt.strftime('%Y-%m-%d %H:%M:%S') if start_dt else None, end_date=end_dt.strftime('%Y-%m-%d %H:%M:%S') if end_dt else None)

        def fmt_date(dt):
            if not dt:
                return ''
            return dt.strftime('%b %d, %Y').replace(' 0', ' ')

        start_value = start_dt.isoformat()[:10] if start_dt else ''
        end_value = end_dt.isoformat()[:10] if end_dt else ''
        date_label = f"{fmt_date(start_dt)} - {fmt_date(end_dt)}" if start_dt and end_dt else 'Custom range'

        if not predictions:
            return render_template(
                'analytics.html',
                user=get_user(user_id),
                range_name=range_name,
                start_value=start_value,
                end_value=end_value,
                date_label=date_label,
                no_data=True,
                insights=['More assessments are needed to identify a meaningful trend.']
            )

        avg_glucose = sum(p['glucose'] for p in predictions) / len(predictions)
        avg_bp = sum(p['blood_pressure'] for p in predictions) / len(predictions)
        avg_bmi = sum(p['bmi'] for p in predictions) / len(predictions)
        avg_age = sum(p['age'] for p in predictions) / len(predictions)
        avg_insulin = sum(p['insulin'] for p in predictions) / len(predictions)
        avg_risk = sum(p['risk_score'] for p in predictions) / len(predictions)

        glucose_series = [p['glucose'] for p in predictions]
        bp_series = [p['blood_pressure'] for p in predictions]
        bmi_series = [p['bmi'] for p in predictions]
        age_series = [p['age'] for p in predictions]
        insulin_series = [p['insulin'] for p in predictions]
        risk_series = [p['risk_score'] for p in predictions]

        summary = {'LOW RISK': 0, 'MODERATE RISK': 0, 'HIGH RISK': 0}
        for p in predictions:
            summary[p['risk_category']] = summary.get(p['risk_category'], 0) + 1

        insights = build_analytics_insights(predictions)
        assessment_count = len(predictions)
        
        return render_template(
            'analytics.html',
            user=get_user(user_id),
            range_name=range_name,
            start_value=start_value,
            end_value=end_value,
            date_label=date_label,
            predictions=predictions,
            assessment_count=assessment_count,
            avg_glucose=avg_glucose,
            avg_bp=avg_bp,
            avg_bmi=avg_bmi,
            avg_age=avg_age,
            avg_insulin=avg_insulin,
            avg_risk=avg_risk,
            glucose_chart=data_line_chart_svg(glucose_series, '#4f46e5', 'Glucose'),
            bp_chart=data_line_chart_svg(bp_series, '#2563eb', 'Blood Pressure'),
            bmi_chart=data_line_chart_svg(bmi_series, '#10b981', 'BMI'),
            age_chart=data_line_chart_svg(age_series, '#f59e0b', 'Age'),
            insulin_chart=data_line_chart_svg(insulin_series, '#ec4899', 'Insulin'),
            risk_chart=data_line_chart_svg(risk_series, '#7c3aed', 'Risk Score'),
            risk_distribution_chart=risk_distribution_chart_svg(summary),
            summary=summary,
            insights=insights,
        )
