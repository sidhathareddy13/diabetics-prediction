from flask import redirect, render_template, request, session, url_for

from backend.auth import get_user, login_required
from backend.database import get_db
from ml.data_processing import latest_prediction_for_user, user_predictions


def register_reports_routes(app):
    @app.route('/reports')
    @login_required
    def reports():
        user_id = session['user_id']
        total_assessments = get_db().execute('SELECT COUNT(*) FROM predictions WHERE user_id = ?', (user_id,)).fetchone()[0]
        return render_template('reports.html', user=get_user(user_id), total_assessments=total_assessments)

    @app.route('/reports/export-csv')
    @login_required
    def export_csv():
        user_id = session['user_id']
        rows = user_predictions(user_id)
        if not rows:
            return redirect(url_for('reports'))
        import csv
        import io
        output = io.StringIO()
        writer = csv.writer(output)
        writer.writerow(['id', 'created_at', 'glucose', 'blood_pressure', 'bmi', 'age', 'insulin', 'risk_score', 'risk_category', 'remarks'])
        for row in rows:
            writer.writerow([row['id'], row['created_at'], row['glucose'], row['blood_pressure'], row['bmi'], row['age'], row['insulin'], row['risk_score'], row['risk_category'], row.get('remarks', '')])
        return output.getvalue(), 200, {'Content-Type': 'text/csv', 'Content-Disposition': 'attachment; filename=diabetes_history.csv'}

    @app.route('/reports/export-excel')
    @login_required
    def export_excel():
        user_id = session['user_id']
        rows = user_predictions(user_id)
        if not rows:
            return redirect(url_for('reports'))
        import io
        from xlsxwriter import Workbook
        output = io.BytesIO()
        workbook = Workbook(output)
        worksheet = workbook.add_worksheet()
        headers = ['id', 'created_at', 'glucose', 'blood_pressure', 'bmi', 'age', 'insulin', 'risk_score', 'risk_category', 'remarks']
        for col, header in enumerate(headers):
            worksheet.write(0, col, header)
        for row_index, row in enumerate(rows, start=1):
            values = [row['id'], row['created_at'], row['glucose'], row['blood_pressure'], row['bmi'], row['age'], row['insulin'], row['risk_score'], row['risk_category'], row.get('remarks', '')]
            for col_index, value in enumerate(values):
                worksheet.write(row_index, col_index, value)
        workbook.close()
        return output.getvalue(), 200, {'Content-Type': 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet', 'Content-Disposition': 'attachment; filename=diabetes_history.xlsx'}
