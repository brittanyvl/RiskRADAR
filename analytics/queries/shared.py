"""
analytics.queries.shared — Core analytics functions used by all stakeholder reports.

All functions return pandas DataFrames or dicts and query SQLite directly.
The 431-row accident population is small enough that SQLite is efficient.
"""

import sqlite3
import pandas as pd
from pathlib import Path

# Feature columns that are safe to interpolate into SQL
_VALID_FEATURE_COLS = frozenset({
    "aircraft_category", "aircraft_make", "season", "region",
    "weather_category", "time_of_day", "decade",
})

# Douglas / McDonnell Douglas merge fragment
_DOUGLAS_MERGE = """
CASE WHEN aircraft_make IN ('Douglas', 'McDonnell Douglas', 'McDonnell-Douglas')
     THEN 'Douglas / McDonnell Douglas' ELSE aircraft_make END
"""


def get_connection() -> sqlite3.Connection:
    """Return a read-only SQLite connection using the project config path."""
    from riskradar.config import DB_PATH
    return sqlite3.connect(str(DB_PATH))


def get_analytics_population() -> pd.DataFrame:
    """
    Return the 431-row accident-only population with features + taxonomy.

    Columns: report_id, aircraft_make (Douglas-merged), aircraft_category,
    region, season, weather_category, time_of_day, decade, l1_categories (comma-sep).
    """
    conn = get_connection()
    df = pd.read_sql_query(f"""
        SELECT
            f.report_id,
            {_DOUGLAS_MERGE} AS aircraft_make,
            f.aircraft_category,
            f.region,
            f.season,
            f.weather_category,
            f.time_of_day,
            f.decade,
            GROUP_CONCAT(DISTINCT t.category_code) AS l1_categories
        FROM report_features f
        JOIN report_types rt ON f.report_id = rt.report_id
        LEFT JOIN report_taxonomy t
            ON f.report_id = t.report_id AND t.level = 'L1'
        WHERE rt.report_type = 'accident'
        GROUP BY f.report_id
    """, conn)
    conn.close()
    return df


def category_counts() -> pd.DataFrame:
    """
    Return 27 L1 categories with report_count and pct_of_reports.

    Only counts accident reports.
    """
    conn = get_connection()
    total = conn.execute("""
        SELECT COUNT(DISTINCT t.report_id)
        FROM report_taxonomy t
        JOIN report_types rt ON t.report_id = rt.report_id
        WHERE t.level = 'L1' AND rt.report_type = 'accident'
    """).fetchone()[0]

    df = pd.read_sql_query("""
        SELECT t.category_code, t.category_name,
               COUNT(DISTINCT t.report_id) AS report_count
        FROM report_taxonomy t
        JOIN report_types rt ON t.report_id = rt.report_id
        WHERE t.level = 'L1' AND rt.report_type = 'accident'
        GROUP BY t.category_code, t.category_name
        ORDER BY report_count DESC
    """, conn)
    conn.close()

    df["pct_of_reports"] = (df["report_count"] / total * 100).round(1)
    return df


def subcategory_counts(parent_code: str) -> pd.DataFrame:
    """
    Return L2 subcategory breakdown for a given L1 parent_code.
    """
    conn = get_connection()
    df = pd.read_sql_query("""
        SELECT t.category_code, t.category_name,
               COUNT(DISTINCT t.report_id) AS report_count
        FROM report_taxonomy t
        JOIN report_types rt ON t.report_id = rt.report_id
        WHERE t.level = 'L2' AND t.parent_code = ?
              AND rt.report_type = 'accident'
        GROUP BY t.category_code, t.category_name
        ORDER BY report_count DESC
    """, conn, params=(parent_code,))
    conn.close()
    return df


def cooccurrence_matrix() -> pd.DataFrame:
    """
    Return L1 x L1 symmetric co-occurrence matrix (accident reports only).

    Returns a pivot table where rows and columns are category codes
    and values are the number of reports sharing both categories.
    """
    conn = get_connection()
    df = pd.read_sql_query("""
        SELECT t1.category_code AS cat1, t2.category_code AS cat2,
               COUNT(DISTINCT t1.report_id) AS co_count
        FROM report_taxonomy t1
        JOIN report_taxonomy t2 ON t1.report_id = t2.report_id
            AND t1.category_code <= t2.category_code
        JOIN report_types rt ON t1.report_id = rt.report_id
        WHERE t1.level = 'L1' AND t2.level = 'L1'
              AND rt.report_type = 'accident'
        GROUP BY t1.category_code, t2.category_code
    """, conn)
    conn.close()

    # Build symmetric matrix
    # Combine both directions
    mirror = df.rename(columns={"cat1": "cat2", "cat2": "cat1"})
    full = pd.concat([df, mirror[mirror["cat1"] != mirror["cat2"]]])
    matrix = full.pivot_table(
        index="cat1", columns="cat2", values="co_count", fill_value=0
    )
    # Sort both axes alphabetically for consistency
    codes = sorted(matrix.index)
    matrix = matrix.reindex(index=codes, columns=codes, fill_value=0)
    return matrix


def category_by_feature(feature_col: str, categories: list[str] | None = None) -> pd.DataFrame:
    """
    Return pivot table: feature_value (rows) x L1 categories (columns) with report counts.

    Args:
        feature_col: One of the whitelisted feature columns.
        categories: Optional list of L1 codes to filter to.

    Raises:
        ValueError: If feature_col is not in the whitelist.
    """
    if feature_col not in _VALID_FEATURE_COLS:
        raise ValueError(
            f"Invalid feature_col '{feature_col}'. "
            f"Allowed: {sorted(_VALID_FEATURE_COLS)}"
        )

    # Apply Douglas merge when querying aircraft_make
    select_col = _DOUGLAS_MERGE if feature_col == "aircraft_make" else f"f.{feature_col}"

    conn = get_connection()
    query = f"""
        SELECT {select_col} AS feature_value,
               t.category_code, t.category_name,
               COUNT(DISTINCT t.report_id) AS report_count
        FROM report_taxonomy t
        JOIN report_features f ON t.report_id = f.report_id
        JOIN report_types rt ON t.report_id = rt.report_id
        WHERE t.level = 'L1' AND rt.report_type = 'accident'
              AND {select_col} IS NOT NULL
        GROUP BY feature_value, t.category_code, t.category_name
    """
    df = pd.read_sql_query(query, conn)
    conn.close()

    if categories:
        df = df[df["category_code"].isin(categories)]

    if df.empty:
        return df

    pivot = df.pivot_table(
        index="feature_value",
        columns="category_code",
        values="report_count",
        fill_value=0,
    )
    return pivot


def category_prevalence_by_decade(categories: list[str] | None = None) -> pd.DataFrame:
    """
    Return decade time series of category prevalence.

    Columns: decade, category_code, report_count, total_in_decade, prevalence_pct.
    """
    conn = get_connection()
    df = pd.read_sql_query("""
        SELECT f.decade, t.category_code,
               COUNT(DISTINCT t.report_id) AS report_count
        FROM report_taxonomy t
        JOIN report_features f ON t.report_id = f.report_id
        JOIN report_types rt ON t.report_id = rt.report_id
        WHERE t.level = 'L1' AND rt.report_type = 'accident'
              AND f.decade IS NOT NULL
        GROUP BY f.decade, t.category_code
        ORDER BY f.decade, report_count DESC
    """, conn)

    # Get total reports per decade
    totals = pd.read_sql_query("""
        SELECT f.decade, COUNT(DISTINCT f.report_id) AS total_in_decade
        FROM report_features f
        JOIN report_types rt ON f.report_id = rt.report_id
        WHERE rt.report_type = 'accident' AND f.decade IS NOT NULL
        GROUP BY f.decade
    """, conn)
    conn.close()

    df = df.merge(totals, on="decade")
    df["prevalence_pct"] = (df["report_count"] / df["total_in_decade"] * 100).round(1)

    if categories:
        df = df[df["category_code"].isin(categories)]

    return df


def dataset_summary() -> dict:
    """
    Return headline metrics for the dataset.

    Keys: total_reports, accident_reports, l1_categories, l2_subcategories,
    unique_manufacturers, date_range, weather_coverage_pct, time_coverage_pct.
    """
    conn = get_connection()
    c = conn.cursor()

    total = c.execute("SELECT COUNT(*) FROM reports").fetchone()[0]
    accidents = c.execute(
        "SELECT COUNT(*) FROM report_types WHERE report_type = 'accident'"
    ).fetchone()[0]
    l1_count = c.execute(
        "SELECT COUNT(DISTINCT category_code) FROM report_taxonomy WHERE level = 'L1'"
    ).fetchone()[0]
    l2_count = c.execute(
        "SELECT COUNT(DISTINCT category_code) FROM report_taxonomy WHERE level = 'L2'"
    ).fetchone()[0]
    mfr_count = c.execute(f"""
        SELECT COUNT(DISTINCT {_DOUGLAS_MERGE})
        FROM report_features f
        JOIN report_types rt ON f.report_id = rt.report_id
        WHERE rt.report_type = 'accident' AND f.aircraft_make IS NOT NULL
    """).fetchone()[0]

    weather_cov = c.execute("""
        SELECT COUNT(*) FROM report_features f
        JOIN report_types rt ON f.report_id = rt.report_id
        WHERE rt.report_type = 'accident' AND f.weather_category IS NOT NULL
    """).fetchone()[0]
    time_cov = c.execute("""
        SELECT COUNT(*) FROM report_features f
        JOIN report_types rt ON f.report_id = rt.report_id
        WHERE rt.report_type = 'accident' AND f.time_of_day IS NOT NULL
    """).fetchone()[0]

    # Date range
    date_range = c.execute("""
        SELECT MIN(r.accident_date), MAX(r.accident_date)
        FROM reports r
        JOIN report_types rt ON r.filename = rt.report_id
        WHERE rt.report_type = 'accident'
    """).fetchone()

    # Categories per report stats
    cats_per = c.execute("""
        SELECT AVG(cnt) FROM (
            SELECT COUNT(DISTINCT t.category_code) AS cnt
            FROM report_taxonomy t
            JOIN report_types rt ON t.report_id = rt.report_id
            WHERE t.level = 'L1' AND rt.report_type = 'accident'
            GROUP BY t.report_id
        )
    """).fetchone()[0]

    # High-lethality categories: LOC-I + CFIT involvement rate
    loci_cfit = c.execute("""
        SELECT COUNT(DISTINCT t.report_id)
        FROM report_taxonomy t
        JOIN report_types rt ON t.report_id = rt.report_id
        WHERE t.level = 'L1' AND rt.report_type = 'accident'
              AND t.category_code IN ('LOC-I', 'CFIT')
    """).fetchone()[0]

    # Component failure rate: SCF-PP or SCF-NP involvement
    scf_rate = c.execute("""
        SELECT COUNT(DISTINCT t.report_id)
        FROM report_taxonomy t
        JOIN report_types rt ON t.report_id = rt.report_id
        WHERE t.level = 'L1' AND rt.report_type = 'accident'
              AND t.category_code IN ('SCF-PP', 'SCF-NP')
    """).fetchone()[0]

    # IMC involvement rate (weather = IMC)
    imc_count = c.execute("""
        SELECT COUNT(DISTINCT f.report_id)
        FROM report_features f
        JOIN report_types rt ON f.report_id = rt.report_id
        WHERE rt.report_type = 'accident' AND f.weather_category = 'IMC'
    """).fetchone()[0]

    conn.close()

    return {
        "total_reports": total,
        "accident_reports": accidents,
        "l1_categories": l1_count,
        "l2_subcategories": l2_count,
        "unique_manufacturers": mfr_count,
        "date_range_start": date_range[0] if date_range else None,
        "date_range_end": date_range[1] if date_range else None,
        "weather_coverage_pct": round(weather_cov / accidents * 100, 1) if accidents else 0,
        "time_coverage_pct": round(time_cov / accidents * 100, 1) if accidents else 0,
        "avg_categories_per_report": round(cats_per, 1) if cats_per else 0,
        "loci_cfit_pct": round(loci_cfit / accidents * 100, 1) if accidents else 0,
        "component_failure_pct": round(scf_rate / accidents * 100, 1) if accidents else 0,
        "imc_pct": round(imc_count / accidents * 100, 1) if accidents else 0,
    }
