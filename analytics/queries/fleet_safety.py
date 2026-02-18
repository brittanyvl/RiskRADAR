"""
analytics.queries.fleet_safety — Fleet Safety report queries.

Aircraft type + manufacturer + component failure analysis.
"""

import pandas as pd
from .shared import get_connection, category_by_feature, subcategory_counts

# Douglas merge SQL fragment (duplicated here to keep queries self-contained)
_DOUGLAS_MERGE = """
CASE WHEN aircraft_make IN ('Douglas', 'McDonnell Douglas', 'McDonnell-Douglas')
     THEN 'Douglas / McDonnell Douglas' ELSE aircraft_make END
"""


def risk_by_aircraft_category() -> pd.DataFrame:
    """
    Top L1 categories per aircraft_category with prevalence %.

    Returns: DataFrame with aircraft_category, category_code, category_name,
             report_count, total_in_category, prevalence_pct.
    """
    conn = get_connection()
    df = pd.read_sql_query("""
        SELECT f.aircraft_category, t.category_code, t.category_name,
               COUNT(DISTINCT t.report_id) AS report_count
        FROM report_taxonomy t
        JOIN report_features f ON t.report_id = f.report_id
        JOIN report_types rt ON t.report_id = rt.report_id
        WHERE t.level = 'L1' AND rt.report_type = 'accident'
              AND f.aircraft_category IS NOT NULL
        GROUP BY f.aircraft_category, t.category_code, t.category_name
        ORDER BY f.aircraft_category, report_count DESC
    """, conn)

    totals = pd.read_sql_query("""
        SELECT f.aircraft_category,
               COUNT(DISTINCT f.report_id) AS total_in_category
        FROM report_features f
        JOIN report_types rt ON f.report_id = rt.report_id
        WHERE rt.report_type = 'accident' AND f.aircraft_category IS NOT NULL
        GROUP BY f.aircraft_category
    """, conn)
    conn.close()

    df = df.merge(totals, on="aircraft_category")
    df["prevalence_pct"] = (df["report_count"] / df["total_in_category"] * 100).round(1)
    return df


def risk_by_manufacturer(top_n: int = 15) -> pd.DataFrame:
    """
    Top manufacturers + their dominant L1 categories (Douglas merged).

    Returns: DataFrame with aircraft_make, category_code, category_name,
             report_count, total_by_make, prevalence_pct.
    """
    conn = get_connection()
    df = pd.read_sql_query(f"""
        SELECT {_DOUGLAS_MERGE} AS aircraft_make,
               t.category_code, t.category_name,
               COUNT(DISTINCT t.report_id) AS report_count
        FROM report_features f
        JOIN report_taxonomy t ON f.report_id = t.report_id
        JOIN report_types rt ON f.report_id = rt.report_id
        WHERE t.level = 'L1' AND rt.report_type = 'accident'
              AND f.aircraft_make IS NOT NULL
        GROUP BY aircraft_make, t.category_code, t.category_name
    """, conn)

    totals = pd.read_sql_query(f"""
        SELECT {_DOUGLAS_MERGE} AS aircraft_make,
               COUNT(DISTINCT f.report_id) AS total_by_make
        FROM report_features f
        JOIN report_types rt ON f.report_id = rt.report_id
        WHERE rt.report_type = 'accident' AND f.aircraft_make IS NOT NULL
        GROUP BY aircraft_make
        ORDER BY total_by_make DESC
    """, conn)
    conn.close()

    # Filter to top_n manufacturers by total reports
    top_makes = totals.head(top_n)["aircraft_make"].tolist()
    df = df[df["aircraft_make"].isin(top_makes)]
    df = df.merge(totals, on="aircraft_make")
    df["prevalence_pct"] = (df["report_count"] / df["total_by_make"] * 100).round(1)
    return df


def scf_pp_breakdown() -> pd.DataFrame:
    """SCF-PP L2 distribution (ENG, FUEL, PROP, FIRE)."""
    return subcategory_counts("SCF-PP")


def scf_np_breakdown() -> pd.DataFrame:
    """SCF-NP L2 distribution (FLT, HYD, ELEC, STRUCT, GEAR)."""
    return subcategory_counts("SCF-NP")


def failure_trends_by_decade() -> pd.DataFrame:
    """
    SCF-PP + SCF-NP prevalence by decade.

    Returns: DataFrame with decade, category_code, report_count,
             total_in_decade, prevalence_pct.
    """
    conn = get_connection()
    df = pd.read_sql_query("""
        SELECT f.decade, t.category_code,
               COUNT(DISTINCT t.report_id) AS report_count
        FROM report_taxonomy t
        JOIN report_features f ON t.report_id = f.report_id
        JOIN report_types rt ON t.report_id = rt.report_id
        WHERE t.level = 'L1' AND rt.report_type = 'accident'
              AND t.category_code IN ('SCF-PP', 'SCF-NP')
              AND f.decade IS NOT NULL
        GROUP BY f.decade, t.category_code
        ORDER BY f.decade
    """, conn)

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
    return df


def manufacturer_category_heatmap(top_n: int = 10) -> pd.DataFrame:
    """
    Pivot table for heatmap: top manufacturers (rows) x L1 categories (cols).

    Values are report counts. Douglas merged.
    """
    conn = get_connection()

    # Get top manufacturers
    top_makes = pd.read_sql_query(f"""
        SELECT {_DOUGLAS_MERGE} AS aircraft_make,
               COUNT(DISTINCT f.report_id) AS total
        FROM report_features f
        JOIN report_types rt ON f.report_id = rt.report_id
        WHERE rt.report_type = 'accident' AND f.aircraft_make IS NOT NULL
        GROUP BY aircraft_make
        ORDER BY total DESC
        LIMIT {top_n}
    """, conn)["aircraft_make"].tolist()

    df = pd.read_sql_query(f"""
        SELECT {_DOUGLAS_MERGE} AS aircraft_make,
               t.category_code,
               COUNT(DISTINCT t.report_id) AS report_count
        FROM report_features f
        JOIN report_taxonomy t ON f.report_id = t.report_id
        JOIN report_types rt ON f.report_id = rt.report_id
        WHERE t.level = 'L1' AND rt.report_type = 'accident'
              AND f.aircraft_make IS NOT NULL
        GROUP BY aircraft_make, t.category_code
    """, conn)
    conn.close()

    df = df[df["aircraft_make"].isin(top_makes)]
    pivot = df.pivot_table(
        index="aircraft_make", columns="category_code",
        values="report_count", fill_value=0
    )
    # Reindex to preserve ranking order
    pivot = pivot.reindex(top_makes)
    return pivot
