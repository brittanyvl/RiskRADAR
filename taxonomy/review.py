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


# =============================================================================
# Hierarchical (L1 + L2) Review Functions
# =============================================================================

def export_hierarchical_review_csv(
    run_id: int = 1,
    sample_size: int = 50,
    stratified: bool = True,
) -> Path:
    """
    Export hierarchical classification results for human review.

    Creates a CSV with L1 and L2 assignments for stratified sample of reports.

    Args:
        run_id: Classification run identifier
        sample_size: Number of reports to include in review
        stratified: If True, sample proportionally from each L1 category

    Returns:
        Path to generated CSV file
    """
    from .subcategories import SUBCATEGORY_BY_CODE

    logger.info(f"Exporting hierarchical review CSV for run {run_id}")

    # Load L1 results
    l1_path = TAXONOMY_DATA_DIR / f"report_l1_run{run_id}.parquet"
    if not l1_path.exists():
        raise FileNotFoundError(
            f"No L1 results found. Run hierarchical classification first."
        )
    l1_reports = pd.read_parquet(l1_path)

    # Load L2 results (may not exist if L2 disabled)
    l2_path = TAXONOMY_DATA_DIR / f"report_l2_run{run_id}.parquet"
    l2_reports = pd.read_parquet(l2_path) if l2_path.exists() else pd.DataFrame()

    # Load report metadata
    from riskradar.config import DB_PATH
    import sqlite3

    conn = sqlite3.connect(DB_PATH)
    reports_df = pd.read_sql_query("""
        SELECT filename as report_id, title, location, accident_date, report_date
        FROM reports
    """, conn)
    conn.close()

    # Select sample reports
    all_report_ids = l1_reports["report_id"].unique()

    if stratified and len(all_report_ids) > sample_size:
        # Stratified sampling by top L1 category
        top_l1 = l1_reports[l1_reports["rank"] == 1][["report_id", "category_code"]]
        samples_per_cat = max(1, sample_size // top_l1["category_code"].nunique())

        sampled_ids = []
        for cat in top_l1["category_code"].unique():
            cat_reports = top_l1[top_l1["category_code"] == cat]["report_id"].tolist()
            n_sample = min(samples_per_cat, len(cat_reports))
            sampled_ids.extend(pd.Series(cat_reports).sample(n=n_sample).tolist())

        # Top up to sample_size if needed
        if len(sampled_ids) < sample_size:
            remaining = [r for r in all_report_ids if r not in sampled_ids]
            additional = min(sample_size - len(sampled_ids), len(remaining))
            sampled_ids.extend(pd.Series(remaining).sample(n=additional).tolist())

        sample_reports = sampled_ids[:sample_size]
    else:
        sample_reports = list(all_report_ids)[:sample_size]

    logger.info(f"Selected {len(sample_reports)} reports for review")

    # Build review rows
    review_rows = []

    for report_id in sample_reports:
        report_meta = reports_df[reports_df["report_id"] == report_id].iloc[0] if len(
            reports_df[reports_df["report_id"] == report_id]
        ) > 0 else {}

        # Get L1 assignments for this report
        report_l1 = l1_reports[l1_reports["report_id"] == report_id].sort_values("rank")

        for _, l1_row in report_l1.iterrows():
            row = {
                "report_id": report_id,
                "title": report_meta.get("title", "") if isinstance(report_meta, dict) else report_meta["title"] if "title" in report_meta else "",
                "accident_date": report_meta.get("accident_date", "") if isinstance(report_meta, dict) else report_meta["accident_date"] if "accident_date" in report_meta else "",
                "location": report_meta.get("location", "") if isinstance(report_meta, dict) else report_meta["location"] if "location" in report_meta else "",
                "level": "L1",
                "category_code": l1_row["category_code"],
                "category_name": l1_row["category_name"],
                "parent_code": "",
                "score": round(l1_row["score"], 3),
                "pct_contribution": round(l1_row["pct_contribution"], 1),
                "confidence": round(l1_row["avg_similarity"], 3),
                "n_chunks": l1_row["n_chunks"],
                "rank": l1_row["rank"],
                "your_decision": "",  # APPROVE / REJECT / CHANGE
                "correct_code": "",   # If CHANGE, what it should be
                "notes": "",
            }
            review_rows.append(row)

            # Get L2 subcategories for this L1 category
            if not l2_reports.empty:
                report_l2 = l2_reports[
                    (l2_reports["report_id"] == report_id) &
                    (l2_reports["parent_code"] == l1_row["category_code"])
                ].sort_values("rank")

                for _, l2_row in report_l2.iterrows():
                    subcat_name = SUBCATEGORY_BY_CODE.get(
                        l2_row["subcategory_code"],
                        type("", (), {"name": l2_row["subcategory_code"]})()
                    ).name

                    row = {
                        "report_id": report_id,
                        "title": "",  # Don't repeat
                        "accident_date": "",
                        "location": "",
                        "level": "L2",
                        "category_code": l2_row["subcategory_code"],
                        "category_name": subcat_name,
                        "parent_code": l2_row["parent_code"],
                        "score": round(l2_row["score"], 3),
                        "pct_contribution": round(l2_row["pct_of_parent"], 1),
                        "confidence": round(l2_row["combined_confidence"], 3),
                        "n_chunks": l2_row["n_chunks"],
                        "rank": l2_row["rank"],
                        "your_decision": "",
                        "correct_code": "",
                        "notes": "",
                    }
                    review_rows.append(row)

    review_df = pd.DataFrame(review_rows)

    # Save
    output_path = TAXONOMY_REVIEW_DIR / f"hierarchical_review_run{run_id}.csv"
    review_df.to_csv(output_path, index=False)

    logger.info(f"Exported hierarchical review CSV to {output_path}")
    logger.info(f"  - {len(sample_reports)} reports")
    logger.info(f"  - {len(review_df[review_df['level'] == 'L1'])} L1 assignments")
    logger.info(f"  - {len(review_df[review_df['level'] == 'L2'])} L2 assignments")

    return output_path


def import_hierarchical_review(
    review_file: Path,
    run_id: int = 1,
) -> dict:
    """
    Import reviewed hierarchical classifications and compute metrics.

    Args:
        review_file: Path to reviewed CSV file
        run_id: Original classification run identifier

    Returns:
        Dict with review statistics and metrics
    """
    logger.info(f"Importing reviewed classifications from {review_file}")

    review_df = pd.read_csv(review_file)

    # Filter to rows with decisions
    reviewed = review_df[review_df["your_decision"].notna() & (review_df["your_decision"] != "")]

    if reviewed.empty:
        return {"error": "No reviewed rows found"}

    # Compute L1 metrics
    l1_reviewed = reviewed[reviewed["level"] == "L1"]
    l1_stats = {
        "total": len(l1_reviewed),
        "approved": len(l1_reviewed[l1_reviewed["your_decision"] == "APPROVE"]),
        "rejected": len(l1_reviewed[l1_reviewed["your_decision"] == "REJECT"]),
        "changed": len(l1_reviewed[l1_reviewed["your_decision"] == "CHANGE"]),
        "uncertain": len(l1_reviewed[l1_reviewed["your_decision"] == "UNCERTAIN"]),
    }
    l1_stats["precision"] = (
        l1_stats["approved"] / l1_stats["total"] * 100
        if l1_stats["total"] > 0 else 0
    )

    # Compute L2 metrics
    l2_reviewed = reviewed[reviewed["level"] == "L2"]
    l2_stats = {
        "total": len(l2_reviewed),
        "approved": len(l2_reviewed[l2_reviewed["your_decision"] == "APPROVE"]),
        "rejected": len(l2_reviewed[l2_reviewed["your_decision"] == "REJECT"]),
        "changed": len(l2_reviewed[l2_reviewed["your_decision"] == "CHANGE"]),
        "uncertain": len(l2_reviewed[l2_reviewed["your_decision"] == "UNCERTAIN"]),
    }
    l2_stats["precision"] = (
        l2_stats["approved"] / l2_stats["total"] * 100
        if l2_stats["total"] > 0 else 0
    )

    # Overall metrics
    overall_stats = {
        "total_reviewed": len(reviewed),
        "reports_reviewed": reviewed["report_id"].nunique(),
        "l1_precision": round(l1_stats["precision"], 1),
        "l2_precision": round(l2_stats["precision"], 1),
    }

    # Save review summary
    summary = {
        "run_id": run_id,
        "review_file": str(review_file),
        "overall": overall_stats,
        "l1": l1_stats,
        "l2": l2_stats,
    }

    summary_path = TAXONOMY_REVIEW_DIR / f"review_summary_run{run_id}.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    # Store reviews in database if desired
    # (Implementation depends on schema - can add to taxonomy_reviews table)

    logger.info(f"Review Summary:")
    logger.info(f"  L1 Precision: {l1_stats['precision']:.1f}%")
    logger.info(f"  L2 Precision: {l2_stats['precision']:.1f}%")
    logger.info(f"  Total reviewed: {len(reviewed)}")

    return summary


def get_review_statistics(run_id: int = 1) -> dict:
    """
    Get statistics from a completed review.

    Args:
        run_id: Classification run identifier

    Returns:
        Dict with review statistics
    """
    summary_path = TAXONOMY_REVIEW_DIR / f"review_summary_run{run_id}.json"

    if not summary_path.exists():
        return {"error": f"No review summary found for run {run_id}"}

    with open(summary_path) as f:
        return json.load(f)


# =============================================================================
# L2 Subcategory Review (HTML like L1)
# =============================================================================

L2_HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>RiskRADAR - L2 Subcategory Review</title>
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

        .stats-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
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

        /* L1 Category (outer section) */
        .l1-section {{
            background: white;
            margin: 20px 0;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            overflow: hidden;
        }}
        .l1-header {{
            background: #16213e;
            color: white;
            padding: 15px 20px;
            cursor: pointer;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }}
        .l1-header:hover {{ background: #1a1a2e; }}
        .l1-code {{ font-weight: bold; font-size: 1.2em; }}
        .l1-name {{ color: #a0a0a0; margin-left: 10px; }}
        .l1-badge {{
            background: #e94560;
            padding: 5px 15px;
            border-radius: 20px;
            font-weight: bold;
        }}
        .l1-content {{
            padding: 0;
            display: none;
        }}
        .l1-content.expanded {{ display: block; }}

        /* L2 Subcategory (inner section) */
        .l2-section {{
            border-bottom: 1px solid #eee;
        }}
        .l2-section:last-child {{ border-bottom: none; }}
        .l2-header {{
            background: #f8f9fa;
            padding: 12px 20px;
            cursor: pointer;
            display: flex;
            justify-content: space-between;
            align-items: center;
            border-left: 4px solid #0f3460;
        }}
        .l2-header:hover {{ background: #e9ecef; }}
        .l2-code {{ font-weight: bold; color: #0f3460; }}
        .l2-name {{ color: #666; margin-left: 10px; }}
        .l2-badge {{
            background: #28a745;
            color: white;
            padding: 3px 12px;
            border-radius: 15px;
            font-size: 0.85em;
        }}
        .l2-content {{
            padding: 0 20px;
            display: none;
        }}
        .l2-content.expanded {{ display: block; padding: 15px 20px; }}

        /* Report cards */
        .report-card {{
            border: 1px solid #ddd;
            border-radius: 8px;
            margin: 15px 0;
            overflow: hidden;
        }}
        .report-header {{
            background: #fff;
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
        .report-meta {{ color: #666; font-size: 0.9em; }}
        .report-title {{ width: 100%; color: #333; font-style: italic; }}
        .similarity-badge {{
            background: #28a745;
            color: white;
            padding: 3px 10px;
            border-radius: 15px;
            font-size: 0.85em;
        }}
        .similarity-badge.medium {{ background: #ffc107; color: #333; }}
        .similarity-badge.low {{ background: #dc3545; }}

        .chunk-container {{ padding: 15px; }}
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

        .toc {{
            background: white;
            padding: 20px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            margin-bottom: 30px;
        }}
        .toc h2 {{ margin-top: 0; }}
        .toc-list {{ column-count: 2; column-gap: 20px; }}
        .toc-item {{ break-inside: avoid; margin-bottom: 8px; }}
        .toc-item a {{ text-decoration: none; color: #16213e; }}
        .toc-item a:hover {{ text-decoration: underline; }}
        .toc-sub {{ margin-left: 20px; font-size: 0.9em; color: #666; }}

        @media (max-width: 768px) {{
            .toc-list {{ column-count: 1; }}
            .stats-grid {{ grid-template-columns: 1fr 1fr; }}
        }}
    </style>
</head>
<body>
    <h1>RiskRADAR - L2 Subcategory Review</h1>
    <p>Generated: {timestamp} | L1 Run: {l1_run_id} | L2 Run: {l2_run_id}</p>

    <div class="stats-grid">
        <div class="stat-card">
            <div class="stat-value">{n_l1_categories}</div>
            <div class="stat-label">L1 Categories with L2</div>
        </div>
        <div class="stat-card">
            <div class="stat-value">{n_subcategories}</div>
            <div class="stat-label">L2 Subcategories Used</div>
        </div>
        <div class="stat-card">
            <div class="stat-value">{n_reports_with_l2}</div>
            <div class="stat-label">Reports with L2</div>
        </div>
        <div class="stat-card">
            <div class="stat-value">{n_l2_assignments}</div>
            <div class="stat-label">L2 Assignments</div>
        </div>
    </div>

    <div class="toc">
        <h2>L1 Categories with Subcategories</h2>
        <div class="toc-list">
            {toc_items}
        </div>
    </div>

    {l1_sections}

    <script>
        document.querySelectorAll('.l1-header').forEach(header => {{
            header.addEventListener('click', () => {{
                const content = header.nextElementSibling;
                content.classList.toggle('expanded');
            }});
        }});
        document.querySelectorAll('.l2-header').forEach(header => {{
            header.addEventListener('click', (e) => {{
                e.stopPropagation();
                const content = header.nextElementSibling;
                content.classList.toggle('expanded');
            }});
        }});
    </script>
</body>
</html>
"""


def export_l2_review_html(
    l1_run_id: int = 1,
    l2_run_id: int = 1,
    max_reports_per_subcat: int = 10,
) -> Path:
    """
    Export L2 subcategory results as HTML for human review.

    Organized by L1 category > L2 subcategory > reports.

    Args:
        l1_run_id: L1 classification run identifier
        l2_run_id: L2 classification run identifier
        max_reports_per_subcat: Maximum reports to show per subcategory

    Returns:
        Path to generated HTML file
    """
    from .subcategories import SUBCATEGORY_BY_CODE

    logger.info(f"Exporting L2 review HTML for L1 run {l1_run_id}, L2 run {l2_run_id}")

    # Load L2 results
    l2_chunk_path = TAXONOMY_DATA_DIR / f"chunk_l2_run{l2_run_id}.parquet"
    l2_report_path = TAXONOMY_DATA_DIR / f"report_l2_run{l2_run_id}.parquet"
    l2_stats_path = TAXONOMY_DATA_DIR / f"l2_stats_run{l2_run_id}.json"

    if not l2_report_path.exists():
        raise FileNotFoundError(
            f"No L2 results found. Run `python -m taxonomy.cli classify-l2` first."
        )

    l2_report_assignments = pd.read_parquet(l2_report_path)
    l2_chunk_assignments = pd.read_parquet(l2_chunk_path) if l2_chunk_path.exists() else pd.DataFrame()

    with open(l2_stats_path) as f:
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

    # Build L1 sections
    l1_sections = []
    toc_items = []

    # Group by parent L1 category
    for l1_code in sorted(l2_report_assignments["parent_code"].unique()):
        l1_cat = CICTT_BY_CODE.get(l1_code)
        l1_name = l1_cat.name if l1_cat else l1_code

        l1_subcats = l2_report_assignments[
            l2_report_assignments["parent_code"] == l1_code
        ]
        n_subcats = l1_subcats["subcategory_code"].nunique()
        n_reports = l1_subcats["report_id"].nunique()

        # TOC entry for L1
        subcat_toc = []
        for subcat_code in l1_subcats["subcategory_code"].unique():
            subcat = SUBCATEGORY_BY_CODE.get(subcat_code)
            subcat_name = subcat.name if subcat else subcat_code
            n_sub_reports = len(l1_subcats[l1_subcats["subcategory_code"] == subcat_code])
            subcat_toc.append(
                f'<div class="toc-sub">{subcat_code}: {subcat_name} ({n_sub_reports})</div>'
            )

        toc_items.append(
            f'<div class="toc-item">'
            f'<a href="#{l1_code}"><strong>{l1_code}</strong> - {l1_name} ({n_subcats} subcats, {n_reports} reports)</a>'
            f'{"".join(subcat_toc)}'
            f'</div>'
        )

        # Build L2 subsections
        l2_subsections = []
        for subcat_code in sorted(l1_subcats["subcategory_code"].unique()):
            subcat = SUBCATEGORY_BY_CODE.get(subcat_code)
            subcat_name = subcat.name if subcat else subcat_code

            subcat_reports = l1_subcats[
                l1_subcats["subcategory_code"] == subcat_code
            ].sort_values("score", ascending=False)

            n_sub_reports = len(subcat_reports)

            # Build report cards for this subcategory
            report_cards = []
            for _, report_row in subcat_reports.head(max_reports_per_subcat).iterrows():
                report_id = report_row["report_id"]
                report_meta = reports_dict.get(report_id, {})

                # Get chunks for this report-subcategory combination
                if not l2_chunk_assignments.empty:
                    report_chunks = l2_chunk_assignments[
                        (l2_chunk_assignments["report_id"] == report_id) &
                        (l2_chunk_assignments["subcategory_code"] == subcat_code)
                    ].sort_values("l2_similarity", ascending=False)
                else:
                    report_chunks = pd.DataFrame()

                # Build chunk HTML
                chunk_html = []
                for _, chunk_row in report_chunks.head(3).iterrows():  # Top 3 chunks
                    chunk_id = chunk_row["chunk_id"]
                    chunk_data = chunks_dict.get(chunk_id, {})

                    section_name = escape_for_html(chunk_data.get("section_name", "Unknown"))
                    chunk_text = escape_for_html(chunk_data.get("chunk_text", "[Chunk not found]"))
                    page_start = chunk_data.get("page_start", "?")
                    page_end = chunk_data.get("page_end", "?")
                    token_count = chunk_data.get("token_count", "?")
                    l2_sim = chunk_row.get("l2_similarity", 0)
                    combined = chunk_row.get("combined_confidence", 0)

                    chunk_html.append(f'''
                    <div class="chunk">
                        <div class="chunk-header">
                            <span class="chunk-section">{section_name}</span>
                            <span>Pages {page_start}-{page_end} |
                                  {token_count} tokens |
                                  L2 sim: {l2_sim:.3f} |
                                  Combined: {combined:.3f}</span>
                        </div>
                        <div class="chunk-text">{chunk_text}</div>
                    </div>
                    ''')

                sim_class = get_similarity_class(report_row.get("score", 0))
                accident_date = escape_for_html(report_meta.get("accident_date", "Date unknown"))
                location = escape_for_html(report_meta.get("location", "Location unknown"))
                title = escape_for_html(report_meta.get("title", "Title unknown"))
                score = report_row.get("score", 0)
                pct = report_row.get("pct_of_parent", 0)

                report_cards.append(f'''
                <div class="report-card">
                    <div class="report-header">
                        <span class="report-id">{report_id}</span>
                        <span class="similarity-badge {sim_class}">
                            Score: {score:.3f} ({pct:.1f}% of {l1_code})
                        </span>
                        <span class="report-meta">
                            {accident_date} |
                            {location}
                        </span>
                        <span class="report-title">{title}</span>
                    </div>
                    <div class="chunk-container">
                        {"".join(chunk_html) if chunk_html else "<p><em>No chunk evidence available</em></p>"}
                    </div>
                </div>
                ''')

            # Showing text
            showing_text = f'<p><em>Showing {min(max_reports_per_subcat, n_sub_reports)} of {n_sub_reports} reports</em></p>' if n_sub_reports > max_reports_per_subcat else ''

            l2_subsections.append(f'''
            <div class="l2-section">
                <div class="l2-header">
                    <div>
                        <span class="l2-code">{subcat_code}</span>
                        <span class="l2-name"> - {escape_for_html(subcat_name)}</span>
                    </div>
                    <span class="l2-badge">{n_sub_reports} reports</span>
                </div>
                <div class="l2-content">
                    {"".join(report_cards)}
                    {showing_text}
                </div>
            </div>
            ''')

        # L1 section
        l1_sections.append(f'''
        <div class="l1-section" id="{l1_code}">
            <div class="l1-header">
                <div>
                    <span class="l1-code">{l1_code}</span>
                    <span class="l1-name"> - {escape_for_html(l1_name)}</span>
                </div>
                <span class="l1-badge">{n_subcats} subcategories | {n_reports} reports</span>
            </div>
            <div class="l1-content">
                {"".join(l2_subsections)}
            </div>
        </div>
        ''')

    # Generate final HTML
    html = L2_HTML_TEMPLATE.format(
        timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        l1_run_id=l1_run_id,
        l2_run_id=l2_run_id,
        n_l1_categories=l2_report_assignments["parent_code"].nunique(),
        n_subcategories=stats["l2"]["subcategories_used"],
        n_reports_with_l2=stats["l2"]["reports_with_l2"],
        n_l2_assignments=stats["l2"]["report_assignments"],
        toc_items="\n".join(toc_items),
        l1_sections="\n".join(l1_sections),
    )

    # Save
    output_path = TAXONOMY_REVIEW_DIR / f"l2_review_l1run{l1_run_id}_l2run{l2_run_id}.html"
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html)

    logger.info(f"Exported L2 review HTML to {output_path}")
    return output_path


def export_l2_review_csv(
    l1_run_id: int = 1,
    l2_run_id: int = 1,
) -> Path:
    """
    Export L2 subcategory results as CSV for spreadsheet review.

    Args:
        l1_run_id: L1 classification run identifier
        l2_run_id: L2 classification run identifier

    Returns:
        Path to generated CSV file
    """
    from .subcategories import SUBCATEGORY_BY_CODE

    logger.info(f"Exporting L2 review CSV for L1 run {l1_run_id}, L2 run {l2_run_id}")

    # Load L2 results
    l2_report_path = TAXONOMY_DATA_DIR / f"report_l2_run{l2_run_id}.parquet"

    if not l2_report_path.exists():
        raise FileNotFoundError(
            f"No L2 results found. Run `python -m taxonomy.cli classify-l2` first."
        )

    l2_reports = pd.read_parquet(l2_report_path)

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
    review_df = l2_reports.merge(reports_df, on="report_id", how="left")

    # Add subcategory names
    review_df["subcategory_name"] = review_df["subcategory_code"].map(
        lambda c: SUBCATEGORY_BY_CODE[c].name if c in SUBCATEGORY_BY_CODE else c
    )

    # Add L1 category names
    review_df["parent_name"] = review_df["parent_code"].map(
        lambda c: CICTT_BY_CODE[c].name if c in CICTT_BY_CODE else c
    )

    # Reorder columns
    review_df = review_df[[
        "report_id", "title", "accident_date", "location",
        "parent_code", "parent_name",
        "subcategory_code", "subcategory_name",
        "score", "pct_of_parent", "combined_confidence",
        "n_chunks", "rank"
    ]]

    # Add review columns
    review_df["your_decision"] = ""  # APPROVE / REJECT / UNCERTAIN
    review_df["notes"] = ""

    output_path = TAXONOMY_REVIEW_DIR / f"l2_review_l1run{l1_run_id}_l2run{l2_run_id}.csv"
    review_df.to_csv(output_path, index=False)

    logger.info(f"Exported L2 review CSV to {output_path}")
    return output_path
