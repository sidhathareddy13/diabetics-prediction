from flask import jsonify, redirect, render_template, request, session, url_for
from werkzeug.security import check_password_hash, generate_password_hash

from backend.auth import get_user, login_required, normalize_theme
from backend.database import get_db


def register_settings_routes(app):
    @app.route('/settings', methods=['GET', 'POST'])
    @login_required
    def settings():
        user_id = session['user_id']
        user = get_user(user_id)
        if request.method == 'POST':
            theme = normalize_theme(request.form.get('theme'))
            notifications_enabled = 1 if request.form.get('notifications_enabled') == 'on' else 0
            reminders_enabled = 1 if request.form.get('reminders_enabled') == 'on' else 0
            get_db().execute(
                'UPDATE users SET theme = ?, notifications_enabled = ?, reminders_enabled = ? WHERE id = ?',
                (theme, notifications_enabled, reminders_enabled, user_id),
            )
            get_db().commit()
            user = get_user(user_id)
            if request.headers.get('X-Requested-With') == 'XMLHttpRequest':
                return jsonify({'success': True, 'theme': theme, 'user_theme': user['theme'] if user else 'light'})
            return redirect(url_for('settings'))
        return render_template('settings.html', user=user)

    @app.route('/settings/password', methods=['POST'])
    @login_required
    def change_password():
        user_id = session['user_id']
        user = get_user(user_id)
        current_password = request.form.get('current_password', '')
        new_password = request.form.get('new_password', '')
        confirm_password = request.form.get('confirm_password', '')

        if not user or not check_password_hash(user['password_hash'], current_password):
            return render_template('settings.html', user=user, error='Current password is incorrect.')
        if len(new_password) < 8:
            return render_template('settings.html', user=user, error='New password must be at least 8 characters long.')
        if new_password != confirm_password:
            return render_template('settings.html', user=user, error='New password and confirmation do not match.')

        get_db().execute(
            'UPDATE users SET password_hash = ? WHERE id = ?',
            (generate_password_hash(new_password), user_id),
        )
        get_db().commit()
        return redirect(url_for('settings'))

    @app.route('/account/delete', methods=['POST'])
    @login_required
    def delete_account():
        user_id = session['user_id']
        db = get_db()
        db.execute('DELETE FROM notifications WHERE user_id = ?', (user_id,))
        db.execute('DELETE FROM reminders WHERE user_id = ?', (user_id,))
        db.execute('DELETE FROM predictions WHERE user_id = ?', (user_id,))
        db.execute('DELETE FROM users WHERE id = ?', (user_id,))
        db.commit()
        session.clear()
        return redirect(url_for('login'))
