from datetime import datetime

from flask import render_template, request, session

from backend.auth import get_user, login_required
from backend.database import get_db
from services.prediction_service import predict_risk
from services.recommendation_service import get_recommendations


def register_prediction_routes(app):
    @app.route('/predict', methods=['GET', 'POST'])
    @login_required
    def predict():
        user_id = session['user_id']
        if request.method == 'POST':
            try:
                glucose = float(request.form.get('glucose'))
                blood_pressure = float(request.form.get('blood_pressure'))
                bmi = float(request.form.get('bmi'))
                age = int(request.form.get('age'))
                insulin = float(request.form.get('insulin'))
                remarks = request.form.get('remarks', '').strip() or '—'
            except (TypeError, ValueError):
                return render_template('predict.html', user=get_user(user_id), error='Please enter valid numeric values.', form_data=request.form)

            if glucose <= 0 or blood_pressure <= 0 or bmi <= 0 or age <= 0 or insulin <= 0:
                return render_template('predict.html', user=get_user(user_id), error='All numeric inputs must be greater than zero.', form_data=request.form)

            prediction = predict_risk(glucose, blood_pressure, bmi, age, insulin)
            rec = get_recommendations(glucose, blood_pressure, bmi, age, insulin, prediction['score'], prediction['category'])
            db = get_db()
            cursor = db.execute(
                '''
                INSERT INTO predictions (user_id, created_at, glucose, blood_pressure, bmi, age, insulin, risk_score, risk_category, remarks)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''',
                (
                    user_id,
                    datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                    glucose,
                    blood_pressure,
                    bmi,
                    age,
                    insulin,
                    prediction['score'],
                    prediction['category'],
                    remarks,
                ),
            )
            db.commit()
            prediction['id'] = cursor.lastrowid
            prediction['recommendations'] = rec
            prediction['remarks'] = remarks
            prediction['date_label'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            return render_template('predict.html', user=get_user(user_id), prediction=prediction, show_result=True)

        return render_template('predict.html', user=get_user(user_id), show_result=False)
