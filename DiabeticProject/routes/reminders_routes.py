from datetime import datetime

from flask import render_template, request, redirect, session, url_for

from backend.auth import get_user, login_required
from backend.database import get_db
from backend.reminders import compute_reminder_status


def register_reminders_routes(app):
    @app.route('/reminders')
    @login_required
    def reminders():
        user_id = session['user_id']
        rows = get_db().execute('SELECT * FROM reminders WHERE user_id = ? ORDER BY date ASC, time ASC', (user_id,)).fetchall()
        reminders_list = []
        for row in rows:
            reminder = dict(row)
            reminder['status'] = compute_reminder_status(reminder)
            reminders_list.append(reminder)

        total = len(reminders_list)
        today = datetime.now().strftime('%Y-%m-%d')
        completed_today = sum(1 for r in reminders_list if r['status'] == 'COMPLETED' and r['date'] == today)
        upcoming_today = sum(1 for r in reminders_list if r['status'] == 'UPCOMING' and r['date'] == today)
        overdue = sum(1 for r in reminders_list if r['status'] == 'OVERDUE')
        return render_template('reminders.html', user=get_user(user_id), reminders=reminders_list, total=total, completed_today=completed_today, upcoming_today=upcoming_today, overdue=overdue)

    @app.route('/reminders/add', methods=['GET', 'POST'])
    @login_required
    def add_reminder():
        user_id = session['user_id']
        preview = None
        if request.method == 'POST':
            title = request.form.get('title', '').strip()
            description = request.form.get('description', '').strip()
            date = request.form.get('date', '')
            time = request.form.get('time', '')
            repeat = request.form.get('repeat', 'Once')
            category = request.form.get('category', 'Lifestyle')
            notification_enabled = 1 if request.form.get('notification_enabled') == 'on' else 0
            status = compute_reminder_status({'date': date, 'time': time, 'status': 'UPCOMING'})

            if not title or not date or not time:
                return render_template('add_reminder.html', user=get_user(user_id), error='Reminder title, date, and time are required.')
            get_db().execute(
                'INSERT INTO reminders (user_id, title, description, date, time, repeat, category, notification_enabled, status) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)',
                (user_id, title, description or '', date, time, repeat, category, notification_enabled, status),
            )
            get_db().commit()
            return redirect(url_for('reminders'))

        return render_template('add_reminder.html', user=get_user(user_id), preview=preview)

    @app.route('/reminders/<int:reminder_id>/complete', methods=['POST'])
    @login_required
    def complete_reminder(reminder_id):
        user_id = session['user_id']
        get_db().execute('UPDATE reminders SET status = ? WHERE id = ? AND user_id = ?', ('COMPLETED', reminder_id, user_id))
        get_db().commit()
        return redirect(url_for('reminders'))

    @app.route('/reminders/<int:reminder_id>/edit', methods=['GET', 'POST'])
    @login_required
    def edit_reminder(reminder_id):
        user_id = session['user_id']
        row = get_db().execute('SELECT * FROM reminders WHERE id = ? AND user_id = ?', (reminder_id, user_id)).fetchone()
        if row is None:
            return redirect(url_for('reminders'))
        reminder = dict(row)
        if request.method == 'POST':
            title = request.form.get('title', '').strip()
            description = request.form.get('description', '').strip()
            date = request.form.get('date', '')
            time = request.form.get('time', '')
            repeat = request.form.get('repeat', 'Once')
            category = request.form.get('category', 'Lifestyle')
            notification_enabled = 1 if request.form.get('notification_enabled') == 'on' else 0
            status = compute_reminder_status({'date': date, 'time': time, 'status': reminder.get('status', 'UPCOMING')})
            get_db().execute(
                'UPDATE reminders SET title = ?, description = ?, date = ?, time = ?, repeat = ?, category = ?, notification_enabled = ?, status = ? WHERE id = ? AND user_id = ?',
                (title, description, date, time, repeat, category, notification_enabled, status, reminder_id, user_id),
            )
            get_db().commit()
            return redirect(url_for('reminders'))
        return render_template('add_reminder.html', user=get_user(user_id), reminder=reminder, edit_mode=True)

    @app.route('/reminders/<int:reminder_id>/delete', methods=['GET', 'POST'])
    @login_required
    def delete_reminder(reminder_id):
        user_id = session['user_id']
        if request.method == 'POST':
            get_db().execute('DELETE FROM reminders WHERE id = ? AND user_id = ?', (reminder_id, user_id))
            get_db().commit()
            return redirect(url_for('reminders'))
        reminder = get_db().execute('SELECT * FROM reminders WHERE id = ? AND user_id = ?', (reminder_id, user_id)).fetchone()
        if reminder is None:
            return redirect(url_for('reminders'))
        return render_template('delete_reminder.html', user=get_user(user_id), reminder=dict(reminder))
