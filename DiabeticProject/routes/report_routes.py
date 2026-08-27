from flask import redirect, session

from backend.auth import get_user, login_required
from backend.database import get_db
from backend.reports import build_individual_report_pdf, build_overall_report_pdf, build_pdf_report
from ml.data_processing import latest_prediction_for_user, user_predictions


def register_report_routes(app):
    @app.route('/report/prediction/<int:prediction_id>/pdf')
    @login_required
    def prediction_report_pdf(prediction_id):
        user_id = session['user_id']
        row = get_db().execute('SELECT * FROM predictions WHERE id = ? AND user_id = ?', (prediction_id, user_id)).fetchone()
        if row is None:
            return redirect('/history')
        pred = dict(row)
        user = get_user(user_id)
        pdf_path = build_individual_report_pdf(user, pred)
        return pdf_path.read_bytes(), 200, {'Content-Type': 'application/pdf', 'Content-Disposition': f'attachment; filename=diabetes_report_{prediction_id}.pdf'}

    @app.route('/report/overall-pdf')
    @login_required
    def overall_report_pdf():
        user_id = session['user_id']
        predictions = user_predictions(user_id)
        if not predictions:
            return redirect('/history')
        user = get_user(user_id)
        pdf_path = build_overall_report_pdf(user, predictions)
        return pdf_path.read_bytes(), 200, {'Content-Type': 'application/pdf', 'Content-Disposition': 'attachment; filename=diabetes_overall_report.pdf'}

    @app.route('/report/latest-pdf')
    @login_required
    def latest_report_pdf():
        user_id = session['user_id']
        latest = latest_prediction_for_user(user_id)
        if latest is None:
            return redirect('/dashboard')
        return redirect(f"/report/prediction/{latest['id']}/pdf")

    @app.route('/report/analytics-pdf')
    @login_required
    def analytics_report_pdf():
        user_id = session['user_id']
        predictions = user_predictions(user_id)
        if not predictions:
            return redirect('/analytics')
        summary_lines = []
        for item in predictions[:5]:
            summary_lines.append(f"{item['created_at']} | Risk {item['risk_score']:.1f}% | {item['risk_category']} | Glucose {item['glucose']} | BMI {item['bmi']}")
        avg_risk = sum(p['risk_score'] for p in predictions) / len(predictions)
        title = 'Diabetes Risk Analytics Report'
        sections = [
            ('Summary', f"Total assessments: {len(predictions)} | Average risk score: {avg_risk:.2f}%"),
            ('Recent Records', '\n'.join(summary_lines)),
            ('Risk Summary', f"Low Risk: {sum(1 for p in predictions if p['risk_category'] == 'LOW RISK')} | Moderate Risk: {sum(1 for p in predictions if p['risk_category'] == 'MODERATE RISK')} | High Risk: {sum(1 for p in predictions if p['risk_category'] == 'HIGH RISK')}"),
            ('Disclaimer', 'This tool is for educational and risk-screening purposes only. It does not replace professional medical diagnosis, treatment, or advice.'),
        ]
        pdf_path = build_pdf_report(title, sections)
        return pdf_path.read_bytes(), 200, {'Content-Type': 'application/pdf', 'Content-Disposition': 'attachment; filename=analytics_report.pdf'}
