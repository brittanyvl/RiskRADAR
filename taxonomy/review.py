"""
taxonomy/review.py
------------------
Export CICTT mappings for human review with full context.

Generates HTML reports with:
- Full chunk text (not truncated)
- Report metadata (ID, title, date)
- Category information and similarity scores
- Collapsible sections for easy navigation
"""

import html
import json
import logging
from datetime import datetime
from pathlib import Path

import pandas as pd

from .cictt import CICTT_BY_CODE, get_primary_categories
from .config import (
    TAXONOMY_DATA_DIR,
    TAXONOMY_REVIEW_DIR,
    CHUNKS_JSONL_PATH,
    REVIEW_CONFIG,
    PROJECT_ROOT,
)


def escape_for_html(text: str) -> str:
    """Escape text for safe HTML insertion."""
    if not text:
        return ""
    # First escape HTML entities, then escape curly braces for .format()
    escaped = html.escape(str(text))
    return escaped.replace("{", "{{").replace("}", "}}")

logger = logging.getLogger(__name__)


HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>RiskRADAR - CICTT Taxonomy Review</title>
    <style>
        * {{ box-sizing: border-box; }}
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            line-height: 1.6;
            max-width: 1400px;
            margin: 0 auto;
            padding: 20px;
            background: #f5f5f5;
        }}
        h1 {{ color: #1a1a2e; border-bottom: 3px solid #16213e; padding-bottom: 10px; }}
        h2 {{ color: #16213e; margin-top: 30px; }}
        h3 {{ color: #0f3460; }}

        .stats-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 15px;
            margin: 20px 0;
        }}
        .stat-card {{
            background: white;
            padding: 20px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            text-align: center;
        }}
        .stat-value {{ font-size: 2em; font-weight: bold; color: #16213e; }}
        .stat-label {{ color: #666; font-size: 0.9em; }}

        .category-section {{
            background: white;
            margin: 20px 0;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            overflow: hidden;
        }}
        .category-header {{
            background: #16213e;
            color: white;
            padding: 15px 20px;
            cursor: pointer;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }}
        .category-header:hover {{ background: #1a1a2e; }}
        .category-code {{ font-weight: bold; font-size: 1.2em; }}
        .category-name {{ color: #a0a0a0; }}
        .category-count {{
            background: #e94560;
            padding: 5px 15px;
            border-radius: 20px;
            font-weight: bold;
        }}
        .category-content {{
            padding: 20px;
            display: none;
        }}
        .category-content.expanded {{ display: block; }}
        .category-description {{
            background: #f8f9fa;
            padding: 15px;
            border-radius: 5px;
            margin-bottom: 20px;
            border-left: 4px solid #16213e;
        }}

        .report-card {{
            border: 1px solid #ddd;
            border-radius: 8px;
            margin: 15px 0;
            overflow: hidden;
        }}
        .report-header {{
            background: #f8f9fa;
            padding: 15px;
            border-bottom: 1px solid #ddd;
            display: flex;
            justify-content: space-between;
            flex-wrap: wrap;
            gap: 10px;
        }}
        .report-id {{
            font-weight: bold;
            color: #16213e;
            font-family: monospace;
            font-size: 1.1em;
        }}
        .report-meta {{
            color: #666;
            font-size: 0.9em;
        }}
        .report-title {{
            width: 100%;
            color: #333;
            font-style: italic;
        }}
        .similarity-badge {{
            background: #28a745;
            color: white;
            padding: 3px 10px;
            border-radius: 15px;
            font-size: 0.85em;
        }}
        .similarity-badge.medium {{ background: #ffc107; color: #333; }}
        .similarity-badge.low {{ background: #dc3545; }}

        .chunk-container {{
            padding: 15px;
        }}
        .chunk {{
            background: #fff;
            border: 1px solid #e0e0e0;
            border-radius: 5px;
            margin: 10px 0;
            padding: 15px;
        }}
        .chunk-header {{
            display: flex;
            justify-content: space-between;
            margin-bottom: 10px;
            padding-bottom: 10px;
            border-bottom: 1px solid #eee;
            font-size: 0.85em;
            color: #666;
        }}
        .chunk-section {{ font-weight: bold; color: #0f3460; }}
        .chunk-text {{
            white-space: pre-wrap;
            font-family: Georgia, serif;
            line-height: 1.7;
        }}

        .review-form {{
            background: #fff3cd;
            padding: 15px;
            border-radius: 5px;
            margin-top: 15px;
        }}
        .review-form label {{ font-weight: bold; display: block; margin-bottom: 5px; }}
        .review-form select, .review-form textarea {{
            width: 100%;
            padding: 8px;
            margin-bottom: 10px;
            border: 1px solid #ddd;
            border-radius: 4px;
        }}

        .toc {{
            background: white;
            padding: 20px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            margin-bottom: 30px;
        }}
        .toc h2 {{ margin-top: 0; }}
        .toc-list {{
            column-count: 3;
            column-gap: 20px;
        }}
        .toc-item {{
            break-inside: avoid;
            margin-bottom: 8px;
        }}
        .toc-item a {{
            text-decoration: none;
            color: #16213e;
        }}
        .toc-item a:hover {{ text-decoration: underline; }}

        @media (max-width: 768px) {{
            .toc-list {{ column-count: 1; }}
            .stats-grid {{ grid-template-columns: 1fr 1fr; }}
        }}
    </style>
</head>
<body>
    <h1>RiskRADAR - CICTT Taxonomy Review</h1>
    <p>Generated: {timestamp}</p>

    <div class="stats-grid">
        <div class="stat-card">
            <div class="stat-value">{n_reports}</div>
            <div class="stat-label">Reports Mapped</div>
        </div>
        <div class="stat-card">
            <div class="stat-value">{n_categories}</div>
            <div class="stat-label">Categories Used</div>
        </div>
        <div class="stat-card">
            <div class="stat-value">{n_assignments}</div>
            <div class="stat-label">Category Assignments</div>
        </div>
        <div class="stat-card">
            <div class="stat-value">{n_chunks}</div>
            <div class="stat-label">Chunks Analyzed</div>
        </div>
    </div>

    <div class="toc">
        <h2>Categories</h2>
        <div class="toc-list">
            {toc_items}
        </div>
    </div>

    {category_sections}

    <script>
        document.querySelectorAll('.category-header').forEach(header => {{
            header.addEventListener('click', () => {{
                const content = header.nextElementSibling;
                content.classList.toggle('expanded');
            }});
        }});
    </script>
</body>
</html>
"""


def get_similarity_class(score: float) -> str:
    """Return CSS class based on similarity score."""
    if score >= 0.6:
        return ""
    elif score >= 0.5:
        return "medium"
    else:
        return "low"


def load_chunks_dict() -> dict:
    """Load chunks as dictionary for quick lookup."""
    chunks = {}
    with open(CHUNKS_JSONL_PATH, "r", encoding="utf-8") as f:
        for line in f:
            chunk = json.loads(line)
            chunks[chunk["chunk_id"]] = chunk
    return chunks


def export_review_html(run_id: int = 1, max_reports_per_cat: int = 10) -> Path:
    """
    Export CICTT mappings as HTML for human review.

    Args:
        run_id: Mapping run identifier
        max_reports_per_cat: Maximum reports to show per category

    Returns:
        Path to generated HTML file
    """
    logger.info(f"Exporting review HTML for run {run_id}")

    # Load mapping results
    report_cats_path = TAXONOMY_DATA_DIR / f"report_categories_run{run_id}.parquet"
    chunk_assigns_path = TAXONOMY_DATA_DIR / f"chunk_assignments_run{run_id}.parquet"
    stats_path = TAXONOMY_DATA_DIR / f"mapping_stats_run{run_id}.json"

    if not report_cats_path.exists():
        raise FileNotFoundError(
            f"No mapping results found. Run `python -m taxonomy.cli map` first."
        )

    report_categories = pd.read_parquet(report_cats_path)
    chunk_assignments = pd.read_parquet(chunk_assigns_path)

    with open(stats_path) as f:
        stats = json.load(f)

    # Load full chunk data
    chunks_dict = load_chunks_dict()

    # Load report metadata
    from riskradar.config import DB_PATH
    import sqlite3

    conn = sqlite3.connect(DB_PATH)
    reports_df = pd.read_sql_query("""
        SELECT filename as report_id, title, location, accident_date, report_date
        FROM reports
    """, conn)
    conn.close()
    reports_dict = reports_df.set_index("report_id").to_dict("index")

    # Build category sections
    category_sections = []
    toc_items = []

    # Group by category
    for cat_code in report_categories["category_code"].unique():
        cat = CICTT_BY_CODE.get(cat_code)
        if not cat:
            continue

        cat_reports = report_categories[
            report_categories["category_code"] == cat_code
        ].sort_values("score", ascending=False)

        n_reports = len(cat_reports)

        # TOC entry
        toc_items.append(
            f'<div class="toc-item">'
            f'<a href="#{cat_code}">{cat_code} - {cat.name} ({n_reports})</a>'
            f'</div>'
        )

        # Build report cards
        report_cards = []
        for _, report_row in cat_reports.head(max_reports_per_cat).iterrows():
            report_id = report_row["report_id"]
            report_meta = reports_dict.get(report_id, {})

            # Get chunks for this report-category combination
            report_chunks = chunk_assignments[
                (chunk_assignments["report_id"] == report_id) &
                (chunk_assignments["category_code"] == cat_code)
            ].sort_values("similarity", ascending=False)

            # Build chunk HTML
            chunk_html = []
            for _, chunk_row in report_chunks.iterrows():
                chunk_id = chunk_row["chunk_id"]
                chunk_data = chunks_dict.get(chunk_id, {})

                # Escape all user content to prevent HTML injection and format() issues
                section_name = escape_for_html(chunk_data.get("section_name", "Unknown"))
                chunk_text = escape_for_html(chunk_data.get("chunk_text", "[Chunk not found]"))
                page_start = chunk_data.get("page_start", "?")
                page_end = chunk_data.get("page_end", "?")
                token_count = chunk_data.get("token_count", "?")
                similarity = chunk_row["similarity"]

                chunk_html.append(f'''
                <div class="chunk">
                    <div class="chunk-header">
                        <span class="chunk-section">{section_name}</span>
                        <span>Pages {page_start}-{page_end} |
                              {token_count} tokens |
                              Similarity: {similarity:.3f}</span>
                    </div>
                    <div class="chunk-text">{chunk_text}</div>
                </div>
                ''')

            sim_class = get_similarity_class(report_row["score"])
            # Escape report metadata
            accident_date = escape_for_html(report_meta.get("accident_date", "Date unknown"))
            location = escape_for_html(report_meta.get("location", "Location unknown"))
            title = escape_for_html(report_meta.get("title", "Title unknown"))
            score = report_row["score"]
            pct = report_row["pct_contribution"]

            report_cards.append(f'''
            <div class="report-card">
                <div class="report-header">
                    <span class="report-id">{report_id}</span>
                    <span class="similarity-badge {sim_class}">
                        Score: {score:.3f} ({pct:.1f}%)
                    </span>
                    <span class="report-meta">
                        {accident_date} |
                        {location}
                    </span>
                    <span class="report-title">{title}</span>
                </div>
                <div class="chunk-container">
                    {"".join(chunk_html)}
                </div>
            </div>
            ''')

        # Category section - escape category info
        cat_name = escape_for_html(cat.name)
        cat_desc = escape_for_html(cat.description)
        showing_text = f'<p><em>Showing {min(max_reports_per_cat, n_reports)} of {n_reports} reports</em></p>' if n_reports > max_reports_per_cat else ''

        section_html = f'''
        <div class="category-section" id="{cat_code}">
            <div class="category-header">
                <div>
                    <span class="category-code">{cat_code}</span>
                    <span class="category-name"> - {cat_name}</span>
                </div>
                <span class="category-count">{n_reports} reports</span>
            </div>
            <div class="category-content">
                <div class="category-description">
                    <strong>Definition:</strong> {cat_desc}
                </div>
                {"".join(report_cards)}
                {showing_text}
            </div>
        </div>
        '''
        category_sections.append(section_html)

    # Generate final HTML
    html = HTML_TEMPLATE.format(
        timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        n_reports=stats["n_reports"],
        n_categories=stats["categories_used"],
        n_assignments=stats["n_report_assignments"],
        n_chunks=stats["n_chunks_processed"],
        toc_items="\n".join(toc_items),
        category_sections="\n".join(category_sections),
    )

    # Save
    output_path = TAXONOMY_REVIEW_DIR / f"cictt_review_run{run_id}.html"
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html)

    logger.info(f"Exported review HTML to {output_path}")
    return output_path


def export_review_csv(run_id: int = 1) -> Path:
    """
    Export CICTT mappings as CSV for spreadsheet review.
    """
    logger.info(f"Exporting review CSV for run {run_id}")

    report_cats_path = TAXONOMY_DATA_DIR / f"report_categories_run{run_id}.parquet"

    if not report_cats_path.exists():
        raise FileNotFoundError(
            f"No mapping results found. Run `python -m taxonomy.cli map` first."
        )

    report_categories = pd.read_parquet(report_cats_path)

    # Load report metadata
    from riskradar.config import DB_PATH
    import sqlite3

    conn = sqlite3.connect(DB_PATH)
    reports_df = pd.read_sql_query("""
        SELECT filename as report_id, title, location, accident_date, report_date
        FROM reports
    """, conn)
    conn.close()

    # Merge
    review_df = report_categories.merge(reports_df, on="report_id", how="left")

    # Add category names
    review_df["category_name"] = review_df["category_code"].map(
        lambda c: CICTT_BY_CODE[c].name if c in CICTT_BY_CODE else c
    )

    # Reorder columns
    review_df = review_df[[
        "report_id", "title", "accident_date", "location",
        "category_code", "category_name", "score", "pct_contribution",
        "avg_similarity", "max_similarity", "n_chunks", "rank"
    ]]

    # Add review columns
    review_df["your_decision"] = ""  # APPROVE / REJECT / UNCERTAIN
    review_df["notes"] = ""

    output_path = TAXONOMY_REVIEW_DIR / f"cictt_review_run{run_id}.csv"
    review_df.to_csv(output_path, index=False)

    logger.info(f"Exported review CSV to {output_path}")
    return output_path
