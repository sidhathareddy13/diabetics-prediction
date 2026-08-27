from flask import render_template, request, redirect, session, url_for

from backend.auth import get_user, login_required
from backend.database import get_db
from ml.data_processing import filtered_user_predictions
from ml.recommendations import build_recommendations, get_risk_category


def register_history_routes(app):
    @app.route('/history')
    @login_required
    def history():
        user_id = session['user_id']
        search = request.args.get('search', '').strip()
        risk = request.args.get('risk', 'All').strip().upper()
        risk = risk if risk in {'All', 'LOW', 'MODERATE', 'HIGH'} else 'All'

        try:
            page = int(request.args.get('page', 1))
        except (TypeError, ValueError):
            page = 1

        if page < 1:
            page = 1

        history_rows, page, total_pages, total_assessments = filtered_user_predictions(
            user_id,
            search=search,
            risk=risk,
            page=page,
            per_page=8,
        )

        return render_template(
            'history.html',
            user=get_user(user_id),
            predictions=history_rows,
            page=page,
            total_pages=total_pages,
            total_assessments=total_assessments,
            search_text=search,
            risk_filter=risk,
        )

    @app.route('/history/<int:prediction_id>/delete', methods=['POST'])
    @login_required
    def delete_prediction(prediction_id):
        user_id = session['user_id']
        row = get_db().execute('SELECT * FROM predictions WHERE id = ? AND user_id = ?', (prediction_id, user_id)).fetchone()
        if row is None:
            return redirect(url_for('history'))
        get_db().execute('DELETE FROM predictions WHERE id = ? AND user_id = ?', (prediction_id, user_id))
        get_db().commit()
        return redirect(url_for('history'))

    @app.route('/history/<int:prediction_id>')
    @login_required
    def history_detail(prediction_id):
        user_id = session['user_id']
        row = get_db().execute('SELECT * FROM predictions WHERE id = ? AND user_id = ?', (prediction_id, user_id)).fetchone()
        if row is None:
            return redirect(url_for('history'))
        pred = dict(row)
        recommendations = build_recommendations(pred['glucose'], pred['blood_pressure'], pred['bmi'], pred['age'], pred['insulin'], pred['risk_score'], pred['risk_category'])
        return render_template('history_detail.html', user=get_user(user_id), prediction=pred, recommendations=recommendations)

    @app.route('/history/<int:prediction_id>/analysis')
    @login_required
    def history_analysis_detail(prediction_id):
        user_id = session['user_id']
        row = get_db().execute('SELECT * FROM predictions WHERE id = ? AND user_id = ?', (prediction_id, user_id)).fetchone()
        if row is None:
            return redirect(url_for('history'))
        pred = dict(row)
        recommendation_data = build_recommendations(pred['glucose'], pred['blood_pressure'], pred['bmi'], pred['age'], pred['insulin'], pred['risk_score'], pred['risk_category'])
        risk_status = get_risk_category(pred['risk_score'])
        if risk_status == 'HIGH RISK':
            interpretation = 'This application score falls in the high display band. Consider discussing the result and your health measurements with a qualified healthcare professional.'
        elif risk_status == 'MODERATE RISK':
            interpretation = 'This application score falls in the moderate display band. Consider maintaining healthy habits and discussing persistent concerns with a qualified healthcare professional.'
        else:
            interpretation = 'This application score falls in the low display band. Continue healthy habits and routine preventive care; the score is not a medical diagnosis.'
        return render_template('history_analysis.html', user=get_user(user_id), prediction=pred, recommendations=recommendation_data, interpretation=interpretation, risk_status=risk_status)
