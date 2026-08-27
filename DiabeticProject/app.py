from datetime import datetime
from flask import Flask, session

from backend.auth import get_active_theme, get_user
from backend.database import configure_app, get_db, init_db
from oauth import register_oauth
from routes.analytics_routes import register_analytics_routes
from routes.auth_routes import register_auth_routes
from routes.dashboard_routes import register_dashboard_routes
from routes.history_routes import register_history_routes
from routes.prediction_routes import register_prediction_routes
from routes.profile_routes import register_profile_routes
from routes.reminders_routes import register_reminders_routes
from routes.report_routes import register_report_routes
from routes.reports_routes import register_reports_routes
from routes.settings_routes import register_settings_routes

app = Flask(__name__)
configure_app(app)


def _parse_display_datetime(value):
    """Parse common SQLite/ISO datetime strings for consistent UI formatting."""
    if not value:
        return None
    text = str(value).strip()
    candidates = (
        text,
        text.replace('Z', '+00:00'),
        text[:26],
        text[:19],
        text[:10],
    )
    for candidate in candidates:
        for fmt in (
            '%Y-%m-%dT%H:%M:%S.%f',
            '%Y-%m-%dT%H:%M:%S',
            '%Y-%m-%d %H:%M:%S.%f',
            '%Y-%m-%d %H:%M:%S',
            '%Y-%m-%d',
        ):
            try:
                return datetime.strptime(candidate, fmt)
            except Exception:
                continue
        try:
            return datetime.fromisoformat(candidate.replace('Z', '+00:00')).replace(tzinfo=None)
        except Exception:
            continue
    return None


def format_display_date(value):
    """Format application dates consistently as DD-MM-YYYY."""
    parsed = _parse_display_datetime(value)
    return parsed.strftime('%d-%m-%Y') if parsed else str(value or '')[:10]


def format_display_datetime(value):
    """Format application date-times as DD-MM-YYYY HH:MM:SS:milliseconds."""
    parsed = _parse_display_datetime(value)
    if not parsed:
        return str(value or '')
    return parsed.strftime('%d-%m-%Y %H:%M:%S:%f')[:-3]


app.jinja_env.filters['display_date'] = format_display_date
app.jinja_env.filters['display_datetime'] = format_display_datetime


@app.context_processor
def inject_base_context():
    user_id = session.get('user_id')
    user = get_user(user_id) if user_id is not None else None
    user_reminders = []
    reminder_count = 0
    if user_id is not None:
        reminder_rows = get_db().execute(
            'SELECT * FROM reminders WHERE user_id = ? ORDER BY date ASC, time ASC LIMIT 5',
            (user_id,),
        ).fetchall()
        user_reminders = [dict(row) for row in reminder_rows]
        reminder_count = len(user_reminders)
    active_theme = get_active_theme(user)
    return {
        'user': user,
        'current_user': user,
        'active_theme': active_theme,
        'header_reminders': user_reminders,
        'header_reminder_count': reminder_count,
    }


register_auth_routes(app)
register_dashboard_routes(app)
register_prediction_routes(app)
register_analytics_routes(app)
register_profile_routes(app)
register_reminders_routes(app)
register_history_routes(app)
register_report_routes(app)
register_reports_routes(app)
register_settings_routes(app)
register_oauth(app)


if __name__ == '__main__':
    app.run(debug=True)
