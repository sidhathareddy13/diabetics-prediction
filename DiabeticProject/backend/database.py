import os
import sqlite3
from pathlib import Path

from flask import current_app, g

BASE_DIR = Path(__file__).resolve().parent.parent
DB_PATH = BASE_DIR / 'app.db'


def get_db():
    if 'db' not in g:
        conn = sqlite3.connect(current_app.config['DATABASE'])
        conn.row_factory = sqlite3.Row
        conn.execute('PRAGMA foreign_keys = ON')
        g.db = conn
    return g.db


def init_db():
    db = get_db()

    db.execute(
        '''
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            full_name TEXT NOT NULL,
            email TEXT NOT NULL UNIQUE,
            password_hash TEXT NOT NULL,
            gender TEXT,
            age INTEGER,
            height_cm REAL,
            weight_kg REAL,
            role TEXT NOT NULL DEFAULT 'patient',
            theme TEXT NOT NULL DEFAULT 'light',
            notifications_enabled INTEGER NOT NULL DEFAULT 1,
            reminders_enabled INTEGER NOT NULL DEFAULT 1,
            last_login TEXT,
            created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
        )
        '''
    )

    columns = db.execute('PRAGMA table_info(users)').fetchall()
    column_names = [column[1] for column in columns]
    for column_name, column_sql in {
        'gender': 'TEXT',
        'age': 'INTEGER',
        'height_cm': 'REAL',
        'weight_kg': 'REAL',
        'role': "TEXT NOT NULL DEFAULT 'patient'",
        'theme': "TEXT NOT NULL DEFAULT 'light'",
        'notifications_enabled': 'INTEGER NOT NULL DEFAULT 1',
        'reminders_enabled': 'INTEGER NOT NULL DEFAULT 1',
        'last_login': 'TEXT',
    }.items():
        if column_name not in column_names:
            db.execute(f'ALTER TABLE users ADD COLUMN {column_name} {column_sql}')

    db.execute(
        '''
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
            glucose REAL NOT NULL,
            blood_pressure REAL NOT NULL,
            bmi REAL NOT NULL,
            age INTEGER NOT NULL,
            insulin REAL NOT NULL,
            risk_score REAL NOT NULL,
            risk_category TEXT NOT NULL,
            remarks TEXT,
            FOREIGN KEY(user_id) REFERENCES users(id)
        )
        '''
    )

    db.execute(
        '''
        CREATE TABLE IF NOT EXISTS reminders (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            title TEXT NOT NULL,
            description TEXT,
            date TEXT NOT NULL,
            time TEXT NOT NULL,
            repeat TEXT NOT NULL DEFAULT 'Once',
            category TEXT NOT NULL,
            notification_enabled INTEGER NOT NULL DEFAULT 1,
            status TEXT NOT NULL DEFAULT 'Upcoming',
            created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY(user_id) REFERENCES users(id)
        )
        '''
    )

    db.commit()



def close_db(exception):
    db = g.pop('db', None)
    if db is not None:
        db.close()


def configure_app(app):
    app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', 'diabetes-risk-secret-key')
    app.config['DATABASE'] = str(DB_PATH)
    app.teardown_appcontext(close_db)
    with app.app_context():
        init_db()
