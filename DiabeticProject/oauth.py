import os

from flask import redirect, render_template, session, url_for

from backend.database import get_db

try:
    from authlib.integrations.flask_client import OAuth
except ModuleNotFoundError:  # pragma: no cover - optional dependency
    OAuth = None


def register_oauth(app):
    oauth = OAuth(app) if OAuth is not None else None
    client_id = os.environ.get('GOOGLE_CLIENT_ID')
    client_secret = os.environ.get('GOOGLE_CLIENT_SECRET')
    redirect_uri = os.environ.get('GOOGLE_REDIRECT_URI', 'http://localhost:5000/auth/google/callback')

    if oauth and client_id and client_secret:
        oauth.register(
            name='google',
            client_id=client_id,
            client_secret=client_secret,
            server_metadata_url='https://accounts.google.com/.well-known/openid-configuration',
            client_kwargs={'scope': 'openid email profile'},
        )

        @app.route('/auth/google')
        def auth_google():
            return oauth.google.authorize_redirect(redirect_uri)

        @app.route('/auth/google/callback')
        def auth_google_callback():
            token = oauth.google.authorize_access_token()
            user_info = token.get('userinfo') or oauth.google.userinfo()
            if not user_info:
                return redirect(url_for('login'))

            email = (user_info.get('email') or '').strip().lower()
            full_name = user_info.get('name') or user_info.get('given_name') or 'Google User'

            if not email:
                return redirect(url_for('login'))

            row = get_db().execute('SELECT * FROM users WHERE email = ?', (email,)).fetchone()
            if row is None:
                cursor = get_db().execute(
                    'INSERT INTO users (full_name, email, password_hash, gender, created_at) VALUES (?, ?, ?, ?, datetime("now"))',
                    (full_name, email, 'oauth-user', 'Other'),
                )
                get_db().commit()
                user_id = cursor.lastrowid
            else:
                user_id = row['id']

            session['user_id'] = user_id
            session['user_name'] = full_name
            return redirect(url_for('dashboard'))
    else:
        @app.route('/auth/google')
        def auth_google():
            return redirect(url_for('login'))

        @app.route('/auth/google/callback')
        def auth_google_callback():
            return redirect(url_for('login'))

    @app.route('/auth/google/configure')
    def google_oauth_configure():
        return render_template(
            'login.html',
            error='Google OAuth is not configured. Set GOOGLE_CLIENT_ID, GOOGLE_CLIENT_SECRET, and GOOGLE_REDIRECT_URI in your environment.',
            oauth_configured=False,
        )
