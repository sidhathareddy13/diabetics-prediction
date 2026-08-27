import os

from flask import render_template, request, redirect, session, url_for

from backend.auth import login_required, get_user
from backend.database import get_db
from werkzeug.security import check_password_hash, generate_password_hash


def register_auth_routes(app):
    @app.route('/')
    def index():
        if 'user_id' in session:
            return redirect(url_for('dashboard'))
        return redirect(url_for('login'))

    @app.route('/login', methods=['GET', 'POST'])
    def login():
        oauth_configured = bool(os.environ.get('GOOGLE_CLIENT_ID') and os.environ.get('GOOGLE_CLIENT_SECRET'))
        if request.method == 'POST':
            email = request.form.get('email', '').strip().lower()
            password = request.form.get('password', '')
            row = get_db().execute('SELECT * FROM users WHERE email = ?', (email,)).fetchone()
            if row and check_password_hash(row['password_hash'], password):
                session['user_id'] = row['id']
                session['user_name'] = row['full_name']
                return redirect(url_for('dashboard'))
            return render_template('login.html', error='Invalid email or password.', oauth_configured=oauth_configured)
        return render_template('login.html', oauth_configured=oauth_configured)

    @app.route('/signup', methods=['GET', 'POST'])
    def signup():
        if request.method == 'POST':
            full_name = request.form.get('full_name', '').strip()
            email = request.form.get('email', '').strip().lower()
            password = request.form.get('password', '')
            gender = request.form.get('gender', '').strip()

            if not full_name or not email or not password or not gender:
                return render_template('signup.html', error='Please fill in all required fields.')
            if len(password) < 8:
                return render_template('signup.html', error='Password must be at least 8 characters long.')
            if gender not in ('Male', 'Female'):
                return render_template('signup.html', error='Please select a valid gender.')
            if get_db().execute('SELECT id FROM users WHERE email = ?', (email,)).fetchone():
                return render_template('signup.html', error='An account with this email already exists.')

            hashed = generate_password_hash(password)
            cursor = get_db().execute(
                'INSERT INTO users (full_name, email, password_hash, gender) VALUES (?, ?, ?, ?)',
                (full_name, email, hashed, gender),
            )
            get_db().commit()
            session['user_id'] = cursor.lastrowid
            session['user_name'] = full_name
            return redirect(url_for('dashboard'))
        return render_template('signup.html')

    @app.route('/logout')
    def logout():
        session.clear()
        return redirect(url_for('login'))

    @app.route('/about')
    @login_required
    def about():
        return render_template('about.html', user=get_user(session['user_id']))
