from datetime import datetime, timedelta


def generate_risk_trend_svg(predictions):
    if not predictions:
        return ''
    max_value = max(item['risk_score'] for item in predictions) if predictions else 100
    min_value = min(item['risk_score'] for item in predictions) if predictions else 0
    points = []
    width = 420
    height = 140
    for index, item in enumerate(predictions):
        x = 20 + (index * (width - 40)) / max(1, len(predictions) - 1)
        y = height - 20 - ((item['risk_score'] - min_value) / max(1, max_value - min_value + 1)) * (height - 40)
        points.append(f'{x:.1f},{y:.1f}')
    polyline = ' '.join(points)
    return f'''
    <svg viewBox="0 0 {width} {height}" width="100%" height="140" xmlns="http://www.w3.org/2000/svg">
      <defs>
        <linearGradient id="trendFill" x1="0" x2="0" y1="0" y2="1">
          <stop offset="0%" stop-color="#7c3aed" stop-opacity="0.25"/>
          <stop offset="100%" stop-color="#7c3aed" stop-opacity="0.03"/>
        </linearGradient>
      </defs>
      <path d="M 20 {height-20} L {points[0]}" fill="none" stroke="transparent"/>
      <polyline points="{polyline}" fill="none" stroke="#7c3aed" stroke-width="3" stroke-linecap="round" stroke-linejoin="round"/>
      <polygon points="20,{height-20} {polyline} {width-20},{height-20}" fill="url(#trendFill)" opacity="0.8"/>
    </svg>
    '''


def data_line_chart_svg(values, color='#7c3aed', label='Value', dates=None):
    if not values:
        return '<div class="empty-inline">No data available</div>'

    width = 420
    height = 170
    left_pad = 26
    right_pad = 78
    top_pad = 16
    bottom_pad = 26

    max_value = max(values) if values else 100
    min_value = min(values) if values else 0
    span = max(1.0, max_value - min_value)

    points = []
    for idx, value in enumerate(values):
        x = left_pad + (idx * (width - left_pad - right_pad)) / max(1, len(values) - 1)
        y = height - bottom_pad - ((value - min_value) / span) * (height - top_pad - bottom_pad)
        points.append((x, y, value))

    def fmt_value(v):
        if abs(v - round(v)) < 1e-9:
            return str(int(round(v)))
        return f'{v:.1f}'

    polyline = ' '.join(f'{x:.1f},{y:.1f}' for x, y, _ in points)
    latest_x, latest_y, latest_value = points[-1]
    latest_text = fmt_value(latest_value)

    grid_lines = []
    for i in range(5):
        y = top_pad + (i * (height - top_pad - bottom_pad) / 4)
        grid_lines.append(f'<line x1="{left_pad}" y1="{y:.1f}" x2="{width - right_pad}" y2="{y:.1f}" stroke="#dfe4ee" stroke-width="1"/>')

    tick_labels = []
    if dates:
        display_dates = [d.strftime('%b %d') if hasattr(d, 'strftime') else str(d) for d in dates]
        for index, date_label in enumerate(display_dates):
            if index == 0 or index == len(display_dates) - 1 or index == len(display_dates) // 2:
                x = left_pad + (index * (width - left_pad - right_pad)) / max(1, len(display_dates) - 1)
                tick_labels.append(f'<text x="{x:.1f}" y="{height - 8}" text-anchor="middle" font-size="10" fill="#6b7280">{date_label}</text>')
    else:
        for idx in range(len(values)):
            if idx == 0 or idx == len(values) - 1 or idx == len(values) // 2:
                x = left_pad + (idx * (width - left_pad - right_pad)) / max(1, len(values) - 1)
                tick_labels.append(f'<text x="{x:.1f}" y="{height - 8}" text-anchor="middle" font-size="10" fill="#6b7280">{idx + 1}</text>')

    area_points = f'{left_pad},{height - bottom_pad} ' + polyline + f' {width - right_pad},{height - bottom_pad}'
    badge_x = latest_x + 10 if latest_x < width - 70 else latest_x - 56
    badge_y = max(latest_y - 14, 18)

    return f'''
    <svg viewBox="0 0 {width} {height}" width="100%" height="170" xmlns="http://www.w3.org/2000/svg" role="img" aria-label="{label} chart">
      <defs>
        <linearGradient id="line-fill-{label.lower().replace(' ', '-')}-{color.replace('#','')}" x1="0" x2="0" y1="0" y2="1">
          <stop offset="0%" stop-color="{color}" stop-opacity="0.18"/>
          <stop offset="100%" stop-color="{color}" stop-opacity="0.02"/>
        </linearGradient>
      </defs>
      {''.join(grid_lines)}
      <path d="M {area_points}" fill="url(#line-fill-{label.lower().replace(' ', '-')}-{color.replace('#','')})" opacity="0.7"/>
      <polyline points="{polyline}" fill="none" stroke="{color}" stroke-width="3" stroke-linecap="round" stroke-linejoin="round"/>
      <circle cx="{latest_x:.1f}" cy="{latest_y:.1f}" r="4" fill="{color}" stroke="white" stroke-width="2"/>
      <g>
        <rect x="{badge_x:.1f}" y="{badge_y:.1f}" width="46" height="22" rx="8" fill="rgba(255,255,255,0.9)" stroke="{color}" stroke-width="1.4"/>
        <text x="{badge_x + 23:.1f}" y="{badge_y + 15:.1f}" text-anchor="middle" font-size="11" font-weight="700" fill="{color}">{latest_text}</text>
      </g>
      {''.join(tick_labels)}
    </svg>
    '''


def date_filter_range(range_name, start_date=None, end_date=None):
    now = datetime.now()
    if range_name == '7':
        start = now - timedelta(days=7)
        end = now
    elif range_name == '30':
        start = now - timedelta(days=30)
        end = now
    elif range_name == '90':
        start = now - timedelta(days=90)
        end = now
    elif range_name == '365':
        start = now - timedelta(days=365)
        end = now
    elif range_name == 'custom' and start_date and end_date:
        start = datetime.fromisoformat(start_date)
        end = datetime.fromisoformat(end_date)
    else:
        start = None
        end = None
    return start, end


def build_analytics_insights(predictions):
    if len(predictions) < 2:
        return ['More assessments are needed to identify a meaningful trend.']
    most_recent = predictions[0]
    previous = predictions[1]
    insights = []
    if most_recent['glucose'] > previous['glucose']:
        insights.append('Your glucose has increased over the selected period.')
    elif most_recent['glucose'] < previous['glucose']:
        insights.append('Your glucose trend has improved compared with the previous assessment.')
    if most_recent['risk_score'] > previous['risk_score']:
        insights.append('Your risk score has increased compared with the previous assessment.')
    elif most_recent['risk_score'] < previous['risk_score']:
        insights.append('Your risk score has decreased compared with the previous assessment.')
    if not insights:
        insights.append('Your recent assessments remain relatively stable across the selected period.')
    return insights


def calculate_trend(values):
    """Calculate trend from a list of values."""
    if len(values) < 2:
        return 'Stable / Insufficient data'

    recent_half = values[:len(values)//2]
    older_half = values[len(values)//2:]

    avg_recent = sum(recent_half) / len(recent_half) if recent_half else 0
    avg_older = sum(older_half) / len(older_half) if older_half else 0

    if avg_recent > avg_older * 1.05:
        return 'Increasing'
    elif avg_recent < avg_older * 0.95:
        return 'Decreasing'
    else:
        return 'Stable'


def risk_distribution_chart_svg(summary):
    """Generate a pie chart showing the distribution of risk categories."""
    if not summary:
        return '<div class="empty-inline">No data available</div>'
    
    total = sum(summary.values())
    if total == 0:
        return '<div class="empty-inline">No data available</div>'
    
    # Calculate percentages and angles
    low_count = summary.get('LOW RISK', 0)
    moderate_count = summary.get('MODERATE RISK', 0)
    high_count = summary.get('HIGH RISK', 0)
    
    low_pct = (low_count / total) * 100
    moderate_pct = (moderate_count / total) * 100
    high_pct = (high_count / total) * 100
    
    low_angle = (low_pct / 100) * 360
    moderate_angle = (moderate_pct / 100) * 360
    high_angle = (high_pct / 100) * 360
    
    # Create pie chart segments
    r = 60
    cx, cy = 100, 100
    
    # Helper function to create arc path
    def create_arc(start_angle, angle, color, label, value):
        start_rad = (start_angle - 90) * 3.14159 / 180
        end_rad = (start_angle + angle - 90) * 3.14159 / 180
        
        x1 = cx + r * (180 * 3.14159 / 180) ** 0 * (3.14159 / 180) * start_rad
        y1 = cy + r * (180 * 3.14159 / 180) ** 0 * (3.14159 / 180) * start_rad
        x2 = cx + r * (180 * 3.14159 / 180) ** 0 * (3.14159 / 180) * end_rad
        y2 = cy + r * (180 * 3.14159 / 180) ** 0 * (3.14159 / 180) * end_rad
        
        x1 = cx + r * ((start_rad) ** 0 if (start_rad) ** 0 else 1)
        y1 = cy + r * ((start_rad) ** 0 if (start_rad) ** 0 else 1)
        
        import math
        x1 = cx + r * math.cos(start_rad)
        y1 = cy + r * math.sin(start_rad)
        x2 = cx + r * math.cos(end_rad)
        y2 = cy + r * math.sin(end_rad)
        
        large_arc = 1 if angle > 180 else 0
        
        return f'<path d="M {cx} {cy} L {x1} {y1} A {r} {r} 0 {large_arc} 1 {x2} {y2} Z" fill="{color}" opacity="0.8"/>'
    
    # Simpler approach: create a basic donut chart
    segments = []
    current_angle = 0
    
    colors = {'LOW RISK': '#16a34a', 'MODERATE RISK': '#f59e0b', 'HIGH RISK': '#dc2626'}
    
    import math
    for risk_level, count in [('LOW RISK', low_count), ('MODERATE RISK', moderate_count), ('HIGH RISK', high_count)]:
        if count == 0:
            continue
        pct = (count / total) * 100
        angle = (pct / 100) * 360
        
        start_rad = math.radians(current_angle - 90)
        end_rad = math.radians(current_angle + angle - 90)
        
        x1 = cx + r * math.cos(start_rad)
        y1 = cy + r * math.sin(start_rad)
        x2 = cx + r * math.cos(end_rad)
        y2 = cy + r * math.sin(end_rad)
        
        large_arc = 1 if angle > 180 else 0
        color = colors[risk_level]
        
        segments.append(f'<path d="M {cx} {cy} L {x1} {y1} A {r} {r} 0 {large_arc} 1 {x2} {y2} Z" fill="{color}" opacity="0.85"/>')
        current_angle += angle
    
    svg_content = '\n'.join(segments)
    
    return f'''
    <svg viewBox="0 0 220 220" width="100%" height="180" xmlns="http://www.w3.org/2000/svg">
      <text x="10" y="20" font-size="14" font-weight="600" fill="#1f2937">Risk Distribution</text>
      {svg_content}
      <circle cx="110" cy="110" r="35" fill="white"/>
      <text x="110" y="105" text-anchor="middle" font-size="24" font-weight="bold" fill="#333">{total}</text>
      <text x="110" y="120" text-anchor="middle" font-size="12" fill="#666">assessments</text>
    </svg>
    '''

