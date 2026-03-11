"""
analytics.queries.search_filters — Filter options and post-search filtering for the search page.
"""

from .shared import get_connection


def get_filter_options() -> dict:
    """
    Return all dropdown options for the search filter panel.

    Returns dict with keys:
        l1_categories: list of (code, name) tuples
        aircraft_categories: list of distinct aircraft_category values
        date_range: (min_date, max_date) from reports
    """
    conn = get_connection()

    # L1 categories
    l1_rows = conn.execute("""
        SELECT DISTINCT category_code, category_name
        FROM report_taxonomy
        WHERE level = 'L1'
        ORDER BY category_code
    """).fetchall()

    # Aircraft categories
    ac_rows = conn.execute("""
        SELECT DISTINCT aircraft_category
        FROM report_features
        WHERE aircraft_category IS NOT NULL AND aircraft_category != ''
        ORDER BY aircraft_category
    """).fetchall()

    # Date range
    date_row = conn.execute("""
        SELECT MIN(accident_date), MAX(accident_date)
        FROM reports
        WHERE accident_date IS NOT NULL
    """).fetchone()

    conn.close()

    return {
        "l1_categories": [(r[0], r[1]) for r in l1_rows],
        "aircraft_categories": [r[0] for r in ac_rows],
        "date_range": (date_row[0], date_row[1]) if date_row else (None, None),
    }


def filter_reports_by_aircraft(
    report_ids: list[str],
    aircraft_categories: list[str],
) -> set[str]:
    """
    Return subset of report_ids matching any of the given aircraft_categories.

    Used for post-search filtering when aircraft type is not available
    in the search backend (Qdrant has no aircraft_category field).
    """
    if not report_ids or not aircraft_categories:
        return set(report_ids)

    conn = get_connection()
    rid_ph = ",".join("?" * len(report_ids))
    ac_ph = ",".join("?" * len(aircraft_categories))

    rows = conn.execute(f"""
        SELECT DISTINCT report_id
        FROM report_features
        WHERE report_id IN ({rid_ph})
          AND aircraft_category IN ({ac_ph})
    """, report_ids + aircraft_categories).fetchall()
    conn.close()

    return {r[0] for r in rows}
