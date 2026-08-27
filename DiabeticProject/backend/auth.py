from flask import redirect, session, url_for

from backend.database import get_db
from ml.recommendations import get_risk_category

ALLOWED_THEMES = {'light', 'dark'}


def normalize_theme(value):
    theme = str(value or 'light').strip().lower()
    return theme if theme in ALLOWED_THEMES else 'light'


def login_required(view):
    def wrapped(*args, **kwargs):
        if 'user_id' not in session:
            return redirect(url_for('login'))
        return view(*args, **kwargs)

    wrapped.__name__ = view.__name__
    return wrapped


def risk_category_from_score(score):
    return get_risk_category(float(score))


def category_color(level):
    mapping = {
        'LOW RISK': '#16a34a',
        'MODERATE RISK': '#f59e0b',
        'HIGH RISK': '#dc2626',
    }
    return mapping.get(level, '#7c3aed')


def get_user(user_id):
    row = get_db().execute('SELECT * FROM users WHERE id = ?', (user_id,)).fetchone()
    if not row:
        return None
    user = dict(row)
    user['theme'] = normalize_theme(user.get('theme'))
    return user


def get_active_theme(user=None):
    if user and isinstance(user, dict):
        return normalize_theme(user.get('theme'))
    return 'light'
