"""
Data processing functions for user predictions and analytics.
"""
import re

import pandas as pd

from backend.database import get_db


def _normalize_search_value(value):
    return str(value or '').strip().lower()


def _parse_date_search(value):
    if not value:
        return None

    text = str(value).strip()
    if not text:
        return None

    patterns = [
        '%d/%m/%Y',
        '%d-%m-%Y',
        '%Y-%m-%d',
        '%Y/%m/%d',
        '%d/%m/%y',
        '%d-%m-%y',
    ]
    for fmt in patterns:
        try:
            return __import__('datetime').datetime.strptime(text, fmt).strftime('%Y-%m-%d')
        except ValueError:
            continue
    return None


def filtered_user_predictions(user_id, search=None, risk='All', page=None, per_page=8):
    """Return user predictions filtered by search and risk, with optional pagination."""
    search_text = _normalize_search_value(search)
    risk_filter = str(risk or 'All').strip().upper()

    clauses = ['user_id = ?']
    params = [user_id]

    if risk_filter in {'LOW', 'MODERATE', 'HIGH'}:
        clauses.append('risk_category = ?')
        params.append({'LOW': 'LOW RISK', 'MODERATE': 'MODERATE RISK', 'HIGH': 'HIGH RISK'}[risk_filter])

    if search_text:
        field_query = search_text
        field_name = None
        field_value = None

        if ':' in search_text:
            parts = search_text.split(':', 1)
            field_name = parts[0].strip()
            field_value = parts[1].strip()
            if field_name in {'glucose', 'blood', 'bp', 'bmi', 'age', 'insulin', 'risk', 'date'}:
                field_query = field_value
                normalized_field = {
                    'glucose': 'glucose',
                    'blood': 'blood_pressure',
                    'bp': 'blood_pressure',
                    'bmi': 'bmi',
                    'age': 'age',
                    'insulin': 'insulin',
                    'risk': 'risk_category',
                    'date': 'created_at',
                }.get(field_name)
                if normalized_field == 'risk_category':
                    risk_value = field_query.upper()
                    if risk_value in {'LOW', 'MODERATE', 'HIGH'}:
                        clauses.append('risk_category = ?')
                        params.append({'LOW': 'LOW RISK', 'MODERATE': 'MODERATE RISK', 'HIGH': 'HIGH RISK'}[risk_value])
                        field_query = ''
                elif normalized_field == 'created_at':
                    parsed_date = _parse_date_search(field_query)
                    if parsed_date:
                        clauses.append("DATE(created_at) LIKE ?")
                        params.append(f'%{parsed_date}%')
                        field_query = ''
                elif normalized_field and field_query:
                    clauses.append(f'{normalized_field} = ?')
                    params.append(float(field_query) if re.fullmatch(r'\d+(?:\.\d+)?', field_query) else field_query)
                    field_query = ''

        if field_query:
            numeric = re.fullmatch(r'\d+(?:\.\d+)?', field_query)
            date_value = _parse_date_search(field_query)
            query_terms = []
            query_values = []

            if numeric:
                num = float(field_query)
                query_terms.extend([
                    'glucose = ?',
                    'blood_pressure = ?',
                    'bmi = ?',
                    'age = ?',
                    'insulin = ?',
                    'risk_score = ?',
                ])
                query_values.extend([num, num, num, int(float(num)), num, num])

            if date_value:
                query_terms.append('DATE(created_at) = ?')
                query_values.append(date_value)

            risk_value = field_query.upper()
            if risk_value in {'LOW', 'MODERATE', 'HIGH'}:
                query_terms.append('risk_category = ?')
                query_values.append({'LOW': 'LOW RISK', 'MODERATE': 'MODERATE RISK', 'HIGH': 'HIGH RISK'}[risk_value])

            query_terms.append('LOWER(CAST(glucose AS TEXT)) LIKE ?')
            query_values.append(f'%{field_query}%')
            query_terms.append('LOWER(CAST(blood_pressure AS TEXT)) LIKE ?')
            query_values.append(f'%{field_query}%')
            query_terms.append('LOWER(CAST(bmi AS TEXT)) LIKE ?')
            query_values.append(f'%{field_query}%')
            query_terms.append('LOWER(CAST(age AS TEXT)) LIKE ?')
            query_values.append(f'%{field_query}%')
            query_terms.append('LOWER(CAST(insulin AS TEXT)) LIKE ?')
            query_values.append(f'%{field_query}%')
            query_terms.append('LOWER(CAST(risk_score AS TEXT)) LIKE ?')
            query_values.append(f'%{field_query}%')
            query_terms.append('LOWER(risk_category) LIKE ?')
            query_values.append(f'%{field_query}%')
            query_terms.append('LOWER(CAST(created_at AS TEXT)) LIKE ?')
            query_values.append(f'%{field_query}%')

            if query_terms:
                clauses.append('(' + ' OR '.join(query_terms) + ')')
                params.extend(query_values)

    query = 'SELECT * FROM predictions WHERE ' + ' AND '.join(clauses) + ' ORDER BY created_at DESC'
    db = get_db()
    rows = db.execute(query, params).fetchall()
    rows = [dict(row) for row in rows]

    if page is not None:
        page = int(page)
        if page < 1:
            page = 1
        per_page = int(per_page or 8)
        if per_page < 1:
            per_page = 8
        total = len(rows)
        total_pages = max(1, (total + per_page - 1) // per_page) if total else 1
        if page > total_pages:
            page = total_pages
        start = (page - 1) * per_page
        end = start + per_page
        return rows[start:end], page, total_pages, total

    return rows


def user_predictions(user_id, limit=None, start_date=None, end_date=None):
    """Retrieve user's predictions from database with optional filtering."""
    query = 'SELECT * FROM predictions WHERE user_id = ?'
    params = [user_id]

    if start_date:
        query += ' AND DATE(created_at) >= ?'
        params.append(start_date)

    if end_date:
        query += ' AND DATE(created_at) <= ?'
        params.append(end_date)

    query += ' ORDER BY created_at DESC'

    if limit:
        query += ' LIMIT ?'
        params.append(limit)

    db = get_db()
    rows = db.execute(query, params).fetchall()
    return [dict(row) for row in rows]


def latest_prediction_for_user(user_id):
    """Fetch the most recent prediction for a user."""
    db = get_db()
    row = db.execute(
        'SELECT * FROM predictions WHERE user_id = ? ORDER BY created_at DESC LIMIT 1',
        (user_id,)
    ).fetchone()
    return dict(row) if row else None


def get_summary_table_for_user(user_id):
    """Build a summary dataframe of user predictions for analytics."""
    preds = user_predictions(user_id)
    if not preds:
        return pd.DataFrame()
    
    df = pd.DataFrame(preds)
    df['created_at'] = pd.to_datetime(df['created_at'])
    df = df.sort_values('created_at')
    return df
