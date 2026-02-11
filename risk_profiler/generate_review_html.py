"""
Generate HTML review pages for feature extraction validation.

Creates interactive HTML files for human-in-the-loop review:
1. Aircraft extraction review
2. Region extraction review
3. Gap analysis (taxonomy but no features)
"""

import sqlite3
import json
from pathlib import Path
from datetime import datetime


def generate_aircraft_review_html(conn, output_path: str):
    """
    Generate HTML page for reviewing aircraft extraction.
    """
    cursor = conn.cursor()

    # Get all reports with extraction results
    reports = cursor.execute("""
        SELECT
            f.report_id,
            r.title,
            f.aircraft_raw,
            f.aircraft_make,
            f.aircraft_model,
            f.aircraft_category,
            f.aircraft_confidence,
            CASE WHEN t.report_id IS NOT NULL THEN 1 ELSE 0 END as has_taxonomy
        FROM report_features f
        JOIN reports r ON f.report_id = r.filename
        LEFT JOIN (
            SELECT DISTINCT report_id FROM report_taxonomy WHERE level='L1'
        ) t ON f.report_id = t.report_id
        ORDER BY
            CASE WHEN f.aircraft_category IS NULL AND t.report_id IS NOT NULL THEN 0
                 WHEN f.aircraft_category IS NULL THEN 1
                 ELSE 2 END,
            f.report_id
    """).fetchall()

    # Count stats
    total = len(reports)
    with_aircraft = sum(1 for r in reports if r[5])
    with_taxonomy = sum(1 for r in reports if r[7])
    gap_count = sum(1 for r in reports if not r[5] and r[7])

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Aircraft Extraction Review - RiskRADAR</title>
    <style>
        * {{ box-sizing: border-box; }}
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            margin: 0; padding: 20px; background: #f5f5f5;
        }}
        .container {{ max-width: 1400px; margin: 0 auto; }}
        h1 {{ color: #333; border-bottom: 2px solid #007bff; padding-bottom: 10px; }}

        .stats {{
            display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 15px; margin-bottom: 20px;
        }}
        .stat-card {{
            background: white; padding: 15px; border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        .stat-card h3 {{ margin: 0 0 5px 0; color: #666; font-size: 14px; }}
        .stat-card .value {{ font-size: 28px; font-weight: bold; color: #333; }}
        .stat-card.critical .value {{ color: #dc3545; }}
        .stat-card.success .value {{ color: #28a745; }}

        .filters {{
            background: white; padding: 15px; border-radius: 8px;
            margin-bottom: 20px; display: flex; gap: 15px; flex-wrap: wrap;
            align-items: center;
        }}
        .filters select, .filters input {{
            padding: 8px 12px; border: 1px solid #ddd; border-radius: 4px;
        }}

        table {{
            width: 100%; background: white; border-collapse: collapse;
            border-radius: 8px; overflow: hidden;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        th {{
            background: #007bff; color: white; padding: 12px;
            text-align: left; position: sticky; top: 0;
        }}
        td {{ padding: 10px 12px; border-bottom: 1px solid #eee; }}
        tr:hover {{ background: #f8f9fa; }}

        .badge {{
            display: inline-block; padding: 3px 8px; border-radius: 12px;
            font-size: 12px; font-weight: 500;
        }}
        .badge-success {{ background: #d4edda; color: #155724; }}
        .badge-warning {{ background: #fff3cd; color: #856404; }}
        .badge-danger {{ background: #f8d7da; color: #721c24; }}
        .badge-info {{ background: #d1ecf1; color: #0c5460; }}
        .badge-secondary {{ background: #e2e3e5; color: #383d41; }}

        .gap-row {{ background: #fff3cd !important; }}
        .gap-row:hover {{ background: #ffe69c !important; }}

        .title-cell {{ max-width: 400px; }}
        .title-text {{
            display: -webkit-box; -webkit-line-clamp: 2;
            -webkit-box-orient: vertical; overflow: hidden;
        }}

        .review-select {{
            padding: 4px 8px; border: 1px solid #ddd; border-radius: 4px;
            font-size: 12px;
        }}

        .export-btn {{
            background: #28a745; color: white; padding: 10px 20px;
            border: none; border-radius: 4px; cursor: pointer;
            font-size: 14px;
        }}
        .export-btn:hover {{ background: #218838; }}

        .search-box {{
            padding: 8px 12px; width: 250px;
            border: 1px solid #ddd; border-radius: 4px;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>Aircraft Extraction Review</h1>
        <p>Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}</p>

        <div class="stats">
            <div class="stat-card">
                <h3>Total Reports</h3>
                <div class="value">{total}</div>
            </div>
            <div class="stat-card success">
                <h3>With Aircraft</h3>
                <div class="value">{with_aircraft} ({with_aircraft/total*100:.1f}%)</div>
            </div>
            <div class="stat-card">
                <h3>With Taxonomy</h3>
                <div class="value">{with_taxonomy}</div>
            </div>
            <div class="stat-card critical">
                <h3>CRITICAL GAP</h3>
                <div class="value">{gap_count}</div>
                <small>Has taxonomy but no aircraft</small>
            </div>
        </div>

        <div class="filters">
            <label>Filter:
                <select id="filterSelect" onchange="filterTable()">
                    <option value="all">All Reports</option>
                    <option value="gap">Gap (needs aircraft)</option>
                    <option value="extracted">Has Aircraft</option>
                    <option value="no-taxonomy">No Taxonomy</option>
                </select>
            </label>
            <label>Category:
                <select id="categoryFilter" onchange="filterTable()">
                    <option value="">All Categories</option>
                    <option value="jet-wide">jet-wide</option>
                    <option value="jet-narrow">jet-narrow</option>
                    <option value="jet-regional">jet-regional</option>
                    <option value="turboprop">turboprop</option>
                    <option value="multi-piston">multi-piston</option>
                    <option value="single-piston">single-piston</option>
                    <option value="helicopter">helicopter</option>
                    <option value="other">other</option>
                </select>
            </label>
            <input type="text" class="search-box" id="searchBox"
                   placeholder="Search titles..." onkeyup="filterTable()">
            <button class="export-btn" onclick="exportReviews()">Export Reviews</button>
        </div>

        <table id="reviewTable">
            <thead>
                <tr>
                    <th>Report ID</th>
                    <th>Title</th>
                    <th>Extracted Make</th>
                    <th>Extracted Model</th>
                    <th>Category</th>
                    <th>Confidence</th>
                    <th>Taxonomy</th>
                    <th>Review</th>
                </tr>
            </thead>
            <tbody>
"""

    for report in reports:
        report_id, title, raw, make, model, category, confidence, has_tax = report

        # Determine row class
        row_class = ""
        if not category and has_tax:
            row_class = "gap-row"

        # Category badge
        if category:
            cat_badge = f'<span class="badge badge-info">{category}</span>'
        else:
            cat_badge = '<span class="badge badge-danger">MISSING</span>'

        # Confidence badge
        if confidence == "high":
            conf_badge = '<span class="badge badge-success">high</span>'
        elif confidence == "medium":
            conf_badge = '<span class="badge badge-warning">medium</span>'
        elif confidence:
            conf_badge = f'<span class="badge badge-secondary">{confidence}</span>'
        else:
            conf_badge = ''

        # Taxonomy badge
        tax_badge = '<span class="badge badge-success">Yes</span>' if has_tax else '<span class="badge badge-secondary">No</span>'

        # Escape title for HTML
        title_escaped = (title or "").replace('"', '&quot;').replace('<', '&lt;').replace('>', '&gt;')

        html += f"""
                <tr class="{row_class}" data-category="{category or ''}" data-has-aircraft="{1 if category else 0}" data-has-taxonomy="{has_tax}">
                    <td><code>{report_id}</code></td>
                    <td class="title-cell"><div class="title-text" title="{title_escaped}">{title_escaped[:100]}{'...' if len(title_escaped) > 100 else ''}</div></td>
                    <td>{make or '-'}</td>
                    <td>{model or '-'}</td>
                    <td>{cat_badge}</td>
                    <td>{conf_badge}</td>
                    <td>{tax_badge}</td>
                    <td>
                        <select class="review-select" data-report="{report_id}">
                            <option value="">-</option>
                            <option value="correct">Correct</option>
                            <option value="wrong-category">Wrong Category</option>
                            <option value="needs-manual">Needs Manual</option>
                            <option value="not-aircraft">Not Aircraft</option>
                        </select>
                    </td>
                </tr>
"""

    html += """
            </tbody>
        </table>
    </div>

    <script>
        function filterTable() {
            const filter = document.getElementById('filterSelect').value;
            const category = document.getElementById('categoryFilter').value;
            const search = document.getElementById('searchBox').value.toLowerCase();
            const rows = document.querySelectorAll('#reviewTable tbody tr');

            rows.forEach(row => {
                let show = true;

                // Filter type
                if (filter === 'gap') {
                    show = row.dataset.hasAircraft === '0' && row.dataset.hasTaxonomy === '1';
                } else if (filter === 'extracted') {
                    show = row.dataset.hasAircraft === '1';
                } else if (filter === 'no-taxonomy') {
                    show = row.dataset.hasTaxonomy === '0';
                }

                // Category filter
                if (category && row.dataset.category !== category) {
                    show = false;
                }

                // Search filter
                if (search && !row.textContent.toLowerCase().includes(search)) {
                    show = false;
                }

                row.style.display = show ? '' : 'none';
            });
        }

        function exportReviews() {
            const reviews = [];
            document.querySelectorAll('.review-select').forEach(select => {
                if (select.value) {
                    reviews.push({
                        report_id: select.dataset.report,
                        review: select.value
                    });
                }
            });

            const blob = new Blob([JSON.stringify(reviews, null, 2)], {type: 'application/json'});
            const url = URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = url;
            a.download = 'aircraft_reviews_' + new Date().toISOString().split('T')[0] + '.json';
            a.click();
        }
    </script>
</body>
</html>
"""

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html)

    print(f"Generated: {output_path}")
    return output_path


def generate_gap_analysis_html(conn, output_path: str):
    """
    Generate HTML page specifically for gap analysis - reports needing aircraft extraction.
    """
    cursor = conn.cursor()

    # Get gap reports with full details
    gap_reports = cursor.execute("""
        SELECT
            f.report_id,
            r.title,
            r.location,
            r.accident_date,
            GROUP_CONCAT(t.category_code, ', ') as categories
        FROM report_features f
        JOIN reports r ON f.report_id = r.filename
        JOIN report_taxonomy t ON f.report_id = t.report_id AND t.level='L1'
        WHERE f.aircraft_category IS NULL OR f.aircraft_category = ''
        GROUP BY f.report_id
        ORDER BY f.report_id
    """).fetchall()

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Gap Analysis - Reports Needing Aircraft - RiskRADAR</title>
    <style>
        * {{ box-sizing: border-box; }}
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            margin: 0; padding: 20px; background: #f5f5f5;
        }}
        .container {{ max-width: 1200px; margin: 0 auto; }}
        h1 {{ color: #dc3545; }}

        .summary {{
            background: #fff3cd; padding: 20px; border-radius: 8px;
            border-left: 4px solid #ffc107; margin-bottom: 20px;
        }}

        .card {{
            background: white; margin-bottom: 15px; border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1); overflow: hidden;
        }}
        .card-header {{
            background: #f8f9fa; padding: 15px;
            border-bottom: 1px solid #eee;
            display: flex; justify-content: space-between; align-items: center;
        }}
        .card-body {{ padding: 15px; }}

        .report-id {{ font-family: monospace; font-weight: bold; }}
        .title {{ font-size: 16px; margin-bottom: 10px; }}
        .meta {{ color: #666; font-size: 14px; }}
        .categories {{ margin-top: 10px; }}
        .badge {{
            display: inline-block; padding: 3px 8px; border-radius: 12px;
            font-size: 12px; margin-right: 5px; margin-bottom: 5px;
            background: #e2e3e5; color: #383d41;
        }}

        .input-row {{
            display: grid; grid-template-columns: 200px 150px 1fr;
            gap: 10px; margin-top: 10px; align-items: center;
        }}
        .input-row label {{ font-weight: 500; }}
        .input-row select, .input-row input {{
            padding: 6px 10px; border: 1px solid #ddd; border-radius: 4px;
        }}

        .export-btn {{
            background: #28a745; color: white; padding: 12px 24px;
            border: none; border-radius: 4px; cursor: pointer;
            font-size: 16px; margin-top: 20px;
        }}
        .export-btn:hover {{ background: #218838; }}
    </style>
</head>
<body>
    <div class="container">
        <h1>Gap Analysis: Reports Needing Aircraft Extraction</h1>

        <div class="summary">
            <strong>{len(gap_reports)} reports</strong> have taxonomy classification but no aircraft category extracted.
            These need manual review or pattern additions.
        </div>

        <button class="export-btn" onclick="exportManualEntries()">Export Manual Entries</button>

        <div style="margin-top: 20px;">
"""

    for report_id, title, location, accident_date, categories in gap_reports:
        title_escaped = (title or "").replace('"', '&quot;').replace('<', '&lt;').replace('>', '&gt;')

        # Format categories as badges
        cat_badges = ""
        if categories:
            for cat in categories.split(', '):
                cat_badges += f'<span class="badge">{cat}</span>'

        html += f"""
            <div class="card">
                <div class="card-header">
                    <span class="report-id">{report_id}</span>
                    <span class="meta">{accident_date or 'No date'} | {location or 'No location'}</span>
                </div>
                <div class="card-body">
                    <div class="title">{title_escaped}</div>
                    <div class="categories">Categories: {cat_badges}</div>
                    <div class="input-row">
                        <label>Aircraft Category:</label>
                        <select data-report="{report_id}" data-field="category">
                            <option value="">-- Select --</option>
                            <option value="jet-wide">jet-wide</option>
                            <option value="jet-narrow">jet-narrow</option>
                            <option value="jet-regional">jet-regional</option>
                            <option value="turboprop">turboprop</option>
                            <option value="multi-piston">multi-piston</option>
                            <option value="single-piston">single-piston</option>
                            <option value="helicopter">helicopter</option>
                            <option value="balloon">balloon</option>
                            <option value="other">other</option>
                            <option value="not-applicable">not-applicable</option>
                        </select>
                        <input type="text" data-report="{report_id}" data-field="notes"
                               placeholder="Notes (e.g., aircraft make/model)">
                    </div>
                </div>
            </div>
"""

    html += """
        </div>
    </div>

    <script>
        function exportManualEntries() {
            const entries = [];
            const cards = document.querySelectorAll('.card');

            cards.forEach(card => {
                const reportId = card.querySelector('[data-field="category"]').dataset.report;
                const category = card.querySelector('[data-field="category"]').value;
                const notes = card.querySelector('[data-field="notes"]').value;

                if (category) {
                    entries.push({
                        report_id: reportId,
                        aircraft_category: category,
                        notes: notes,
                        source: 'manual_review'
                    });
                }
            });

            if (entries.length === 0) {
                alert('No entries to export. Please fill in some categories first.');
                return;
            }

            const blob = new Blob([JSON.stringify(entries, null, 2)], {type: 'application/json'});
            const url = URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = url;
            a.download = 'manual_aircraft_entries_' + new Date().toISOString().split('T')[0] + '.json';
            a.click();

            alert('Exported ' + entries.length + ' entries');
        }
    </script>
</body>
</html>
"""

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html)

    print(f"Generated: {output_path}")
    return output_path


def generate_all_reviews(db_path: str = "sqlite/riskradar.db"):
    """Generate all review HTML files."""
    conn = sqlite3.connect(db_path)

    output_dir = Path("risk_profiler/review")
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Generating review HTML files...")
    print("=" * 50)

    # Aircraft review
    generate_aircraft_review_html(
        conn,
        str(output_dir / "aircraft_review.html")
    )

    # Gap analysis
    generate_gap_analysis_html(
        conn,
        str(output_dir / "gap_analysis.html")
    )

    print("=" * 50)
    print(f"\nOpen in browser:")
    print(f"  - {output_dir / 'aircraft_review.html'}")
    print(f"  - {output_dir / 'gap_analysis.html'}")

    conn.close()


if __name__ == "__main__":
    generate_all_reviews()
