"""
analytics.queries.operational_risk — Operational Risk report queries.

LOC-I/CFIT deep dives, weather x time matrix, seasonal patterns,
regional risk concentrations, night accident share.
"""

import pandas as pd
from .shared import get_connection, subcategory_counts, category_by_feature, category_counts


def loc_i_breakdown() -> pd.DataFrame:
    """LOC-I L2 subcategories (STALL, UPSET, SD, ENV, SYS, LOAD)."""
    return subcategory_counts("LOC-I")


def cfit_breakdown() -> pd.DataFrame:
    """CFIT L2 subcategories (NAV, PROC, SA, VIS, TAWS)."""
    return subcategory_counts("CFIT")


def weather_time_matrix() -> pd.DataFrame:
    """
    Pivot: weather_category (rows) x time_of_day (cols) accident counts.
    """
    conn = get_connection()
    df = pd.read_sql_query("""
        SELECT f.weather_category, f.time_of_day,
               COUNT(DISTINCT f.report_id) AS report_count
        FROM report_features f
        JOIN report_types rt ON f.report_id = rt.report_id
        WHERE rt.report_type = 'accident'
              AND f.weather_category IN ('VMC', 'IMC')
              AND f.time_of_day IS NOT NULL
        GROUP BY f.weather_category, f.time_of_day
    """, conn)
    conn.close()

    pivot = df.pivot_table(
        index="weather_category", columns="time_of_day",
        values="report_count", fill_value=0
    )
    time_order = ["Morning", "Afternoon", "Evening", "Night"]
    pivot = pivot.reindex(columns=[t for t in time_order if t in pivot.columns])
    return pivot


def seasonal_patterns(categories: list[str] | None = None) -> pd.DataFrame:
    """
    Category prevalence by season.

    Returns: DataFrame with season, category_code, report_count,
             total_in_season, prevalence_pct.
    """
    conn = get_connection()

    cat_filter = ""
    if categories:
        placeholders = ",".join("?" * len(categories))
        cat_filter = f"AND t.category_code IN ({placeholders})"

    query = f"""
        SELECT f.season, t.category_code,
               COUNT(DISTINCT t.report_id) AS report_count
        FROM report_taxonomy t
        JOIN report_features f ON t.report_id = f.report_id
        JOIN report_types rt ON t.report_id = rt.report_id
        WHERE t.level = 'L1' AND rt.report_type = 'accident'
              AND f.season IS NOT NULL
              {cat_filter}
        GROUP BY f.season, t.category_code
        ORDER BY f.season, report_count DESC
    """
    params = categories if categories else []
    df = pd.read_sql_query(query, conn, params=params)

    totals = pd.read_sql_query("""
        SELECT f.season, COUNT(DISTINCT f.report_id) AS total_in_season
        FROM report_features f
        JOIN report_types rt ON f.report_id = rt.report_id
        WHERE rt.report_type = 'accident' AND f.season IS NOT NULL
        GROUP BY f.season
    """, conn)
    conn.close()

    df = df.merge(totals, on="season")
    df["prevalence_pct"] = (df["report_count"] / df["total_in_season"] * 100).round(1)
    return df


def aircraft_type_risk_signatures() -> pd.DataFrame:
    """
    Top 5 L1 categories per aircraft_category.

    Returns: DataFrame with aircraft_category, rank, category_code,
             category_name, report_count, prevalence_pct.
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

    # Keep top 5 per aircraft category
    df = df.sort_values(["aircraft_category", "report_count"], ascending=[True, False])
    df["rank"] = df.groupby("aircraft_category").cumcount() + 1
    df = df[df["rank"] <= 5]
    return df


def critical_phase_categories() -> pd.DataFrame:
    """
    Categories with highest IMC+Night concentration.

    Returns: DataFrame with category_code, imc_night_count, total_count, risk_ratio.
    """
    conn = get_connection()

    # Total per category
    total_df = pd.read_sql_query("""
        SELECT t.category_code,
               COUNT(DISTINCT t.report_id) AS total_count
        FROM report_taxonomy t
        JOIN report_types rt ON t.report_id = rt.report_id
        WHERE t.level = 'L1' AND rt.report_type = 'accident'
        GROUP BY t.category_code
    """, conn)

    # IMC + Night per category
    imc_night_df = pd.read_sql_query("""
        SELECT t.category_code,
               COUNT(DISTINCT t.report_id) AS imc_night_count
        FROM report_taxonomy t
        JOIN report_features f ON t.report_id = f.report_id
        JOIN report_types rt ON t.report_id = rt.report_id
        WHERE t.level = 'L1' AND rt.report_type = 'accident'
              AND f.weather_category = 'IMC' AND f.time_of_day = 'Night'
        GROUP BY t.category_code
    """, conn)

    # Overall IMC+Night rate
    overall = pd.read_sql_query("""
        SELECT COUNT(DISTINCT f.report_id) AS imc_night_total,
               (SELECT COUNT(DISTINCT f2.report_id)
                FROM report_features f2
                JOIN report_types rt2 ON f2.report_id = rt2.report_id
                WHERE rt2.report_type = 'accident') AS grand_total
        FROM report_features f
        JOIN report_types rt ON f.report_id = rt.report_id
        WHERE rt.report_type = 'accident'
              AND f.weather_category = 'IMC' AND f.time_of_day = 'Night'
    """, conn)
    conn.close()

    df = total_df.merge(imc_night_df, on="category_code", how="left")
    df["imc_night_count"] = df["imc_night_count"].fillna(0).astype(int)

    # Risk ratio: observed IMC+Night rate for this category vs overall rate
    overall_rate = (overall["imc_night_total"].iloc[0] /
                    overall["grand_total"].iloc[0]) if overall["grand_total"].iloc[0] else 0
    df["category_imc_night_rate"] = df["imc_night_count"] / df["total_count"]
    df["risk_ratio"] = (df["category_imc_night_rate"] / overall_rate).round(2) if overall_rate else 0

    df = df.sort_values("risk_ratio", ascending=False)
    return df


def night_accident_share() -> dict:
    """
    Percentage of accidents (with known time_of_day) that occurred at night.

    Returns: dict with night_count, total_with_time, night_pct.
    """
    conn = get_connection()
    row = conn.execute("""
        SELECT
            SUM(CASE WHEN f.time_of_day = 'Night' THEN 1 ELSE 0 END) AS night_count,
            COUNT(DISTINCT f.report_id) AS total_with_time
        FROM report_features f
        JOIN report_types rt ON f.report_id = rt.report_id
        WHERE rt.report_type = 'accident' AND f.time_of_day IS NOT NULL
    """).fetchone()
    conn.close()

    night_count = row[0] or 0
    total_with_time = row[1] or 1
    return {
        "night_count": night_count,
        "total_with_time": total_with_time,
        "night_pct": round(night_count / total_with_time * 100, 1),
    }


def region_category_matrix(top_n_categories: int = 5) -> pd.DataFrame:
    """
    Region (rows) x top-N L1 categories (columns) report count matrix.

    Categories are selected as the top N by overall report count.
    Returns: pivot DataFrame with region index, category_code columns, int values.
    """
    top_cats = category_counts().head(top_n_categories)["category_code"].tolist()
    matrix = category_by_feature("region", categories=top_cats)
    if matrix.empty:
        return matrix
    # Sort rows by total report count descending
    matrix["_total"] = matrix.sum(axis=1)
    matrix = matrix.sort_values("_total", ascending=False).drop(columns=["_total"])
    return matrix
