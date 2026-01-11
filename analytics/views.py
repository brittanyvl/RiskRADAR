"""
Pre-built analytical views for RiskRADAR DuckDB analytics.

These views provide common aggregations and joins for quick analysis.
"""

import duckdb

# View definitions
VIEWS = {
    "data_summary": """
        CREATE OR REPLACE VIEW data_summary AS
        SELECT
            (SELECT COUNT(*) FROM pages) as total_pages,
            (SELECT COUNT(*) FROM documents) as total_documents,
            (SELECT COUNT(*) FROM chunks) as total_chunks,
            (SELECT SUM(token_count) FROM chunks) as total_tokens,
            (SELECT ROUND(AVG(token_count), 1) FROM chunks) as avg_tokens_per_chunk
    """,

    "extraction_quality": """
        CREATE OR REPLACE VIEW extraction_quality AS
        SELECT
            source,
            COUNT(*) as page_count,
            ROUND(AVG(char_count), 0)::INTEGER as avg_chars,
            ROUND(AVG(alphabetic_ratio), 3) as avg_alpha_ratio,
            ROUND(AVG(ocr_confidence), 1) as avg_ocr_conf
        FROM pages
        GROUP BY source
    """,

    "chunks_by_section": """
        CREATE OR REPLACE VIEW chunks_by_section AS
        SELECT
            COALESCE(section_name, 'NO_SECTION') as section_name,
            COUNT(*) as chunk_count,
            ROUND(AVG(token_count), 0)::INTEGER as avg_tokens,
            SUM(token_count) as total_tokens
        FROM chunks
        GROUP BY section_name
        ORDER BY chunk_count DESC
    """,

    "token_distribution": """
        CREATE OR REPLACE VIEW token_distribution AS
        SELECT
            CASE
                WHEN token_count < 500 THEN 'under_500'
                WHEN token_count BETWEEN 500 AND 700 THEN '500-700'
                ELSE 'over_700'
            END as bucket,
            COUNT(*) as count,
            ROUND(100.0 * COUNT(*) / SUM(COUNT(*)) OVER (), 1) as pct
        FROM chunks
        GROUP BY bucket
        ORDER BY bucket
    """,

    "chunks_per_report": """
        CREATE OR REPLACE VIEW chunks_per_report AS
        SELECT
            report_id,
            COUNT(*) as chunk_count,
            SUM(token_count) as total_tokens,
            MIN(token_count) as min_tokens,
            MAX(token_count) as max_tokens
        FROM chunks
        GROUP BY report_id
        ORDER BY chunk_count DESC
    """,

    "section_detection_stats": """
        CREATE OR REPLACE VIEW section_detection_stats AS
        SELECT
            section_detection_method,
            COUNT(*) as chunk_count,
            ROUND(100.0 * COUNT(*) / SUM(COUNT(*)) OVER (), 1) as pct
        FROM chunks
        GROUP BY section_detection_method
        ORDER BY chunk_count DESC
    """,

    "text_source_distribution": """
        CREATE OR REPLACE VIEW text_source_distribution AS
        SELECT
            text_source,
            COUNT(*) as chunk_count,
            ROUND(100.0 * COUNT(*) / SUM(COUNT(*)) OVER (), 1) as pct,
            SUM(token_count) as total_tokens
        FROM chunks
        GROUP BY text_source
        ORDER BY chunk_count DESC
    """,
}

# Views that require SQLite attachment (report metadata joins)
SQLITE_VIEWS = {
    "chunks_enriched": """
        CREATE OR REPLACE VIEW chunks_enriched AS
        SELECT
            c.*,
            r.title,
            r.accident_date,
            r.report_date,
            r.location,
            r.report_number
        FROM chunks c
        JOIN sqlite.reports r ON c.report_id = r.filename
    """,

    "timeline_by_decade": """
        CREATE OR REPLACE VIEW timeline_by_decade AS
        SELECT
            (EXTRACT(YEAR FROM CAST(r.accident_date AS DATE))::INTEGER // 10) * 10 as decade,
            COUNT(DISTINCT c.report_id) as reports,
            COUNT(*) as chunks,
            SUM(c.token_count) as total_tokens
        FROM chunks c
        JOIN sqlite.reports r ON c.report_id = r.filename
        WHERE r.accident_date IS NOT NULL AND r.accident_date != ''
        GROUP BY decade
        ORDER BY decade
    """,

    "timeline_by_year": """
        CREATE OR REPLACE VIEW timeline_by_year AS
        SELECT
            EXTRACT(YEAR FROM CAST(r.accident_date AS DATE))::INTEGER as year,
            COUNT(DISTINCT c.report_id) as reports,
            COUNT(*) as chunks,
            SUM(c.token_count) as total_tokens
        FROM chunks c
        JOIN sqlite.reports r ON c.report_id = r.filename
        WHERE r.accident_date IS NOT NULL AND r.accident_date != ''
        GROUP BY year
        ORDER BY year
    """,
}


def register_views(conn: duckdb.DuckDBPyConnection, include_sqlite: bool = True) -> list[str]:
    """
    Register all pre-built views in the DuckDB connection.

    Args:
        conn: DuckDB connection with tables loaded
        include_sqlite: Whether to register views that join with SQLite

    Returns:
        List of registered view names
    """
    registered = []

    # Register base views
    for name, sql in VIEWS.items():
        try:
            conn.execute(sql)
            registered.append(name)
        except Exception as e:
            print(f"Warning: Could not create view '{name}': {e}")

    # Register SQLite-dependent views if requested
    if include_sqlite:
        for name, sql in SQLITE_VIEWS.items():
            try:
                conn.execute(sql)
                registered.append(name)
            except Exception as e:
                print(f"Warning: Could not create view '{name}': {e}")

    return registered


def list_views() -> list[str]:
    """Return list of all available view names."""
    return list(VIEWS.keys()) + list(SQLITE_VIEWS.keys())
