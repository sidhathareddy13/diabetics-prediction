from flask import redirect, render_template, request, session, url_for

from backend.auth import get_user, login_required
from backend.database import get_db


def register_profile_routes(app):
    @app.route('/profile', methods=['GET', 'POST'])
    @login_required
    def profile():
        user_id = session['user_id']
        user = get_user(user_id)
        if request.method == 'POST':
            full_name = request.form.get('full_name', '').strip()
            gender = request.form.get('gender', '').strip()
            age = request.form.get('age', '')
            height_cm = request.form.get('height_cm', '')
            weight_kg = request.form.get('weight_kg', '')

            if not full_name:
                return render_template('profile.html', user=user, error='Full name is required.', assessment_count=0, bmi=None)

            get_db().execute(
                'UPDATE users SET full_name = ?, gender = ?, age = ?, height_cm = ?, weight_kg = ? WHERE id = ?',
                (
                    full_name,
                    gender,
                    int(age) if age else None,
                    float(height_cm) if height_cm else None,
                    float(weight_kg) if weight_kg else None,
                    user_id,
                ),
            )
            get_db().commit()
            user = get_user(user_id)
            return redirect(url_for('profile'))

        bmi = None
        if user and user.get('height_cm') and user.get('weight_kg'):
            height_m = float(user['height_cm']) / 100
            if height_m > 0:
                bmi = round(float(user['weight_kg']) / (height_m ** 2), 1)

        assessment_count = get_db().execute('SELECT COUNT(*) FROM predictions WHERE user_id = ?', (user_id,)).fetchone()[0]
        return render_template('profile.html', user=user, assessment_count=assessment_count, bmi=bmi)
