"""
analytics.queries.underwriting — Underwriting Risk report queries.

Region/season/weather/time risk matrices and Bayesian profile comparisons.
"""

import pandas as pd
from .shared import get_connection


def region_season_matrix() -> pd.DataFrame:
    """
    Pivot: region (rows) x season (cols) accident counts.
    """
    conn = get_connection()
    df = pd.read_sql_query("""
        SELECT f.region, f.season,
               COUNT(DISTINCT f.report_id) AS report_count
        FROM report_features f
        JOIN report_types rt ON f.report_id = rt.report_id
        WHERE rt.report_type = 'accident'
              AND f.region IS NOT NULL AND f.season IS NOT NULL
        GROUP BY f.region, f.season
    """, conn)
    conn.close()

    pivot = df.pivot_table(
        index="region", columns="season",
        values="report_count", fill_value=0
    )
    # Reorder seasons
    season_order = ["Spring", "Summer", "Fall", "Winter"]
    pivot = pivot.reindex(columns=[s for s in season_order if s in pivot.columns])
    return pivot


def vmc_imc_category_distribution() -> pd.DataFrame:
    """
    L1 categories split by VMC/IMC.

    Returns: DataFrame with category_code, category_name, weather_category,
             report_count, total_weather, prevalence_pct.
    """
    conn = get_connection()
    df = pd.read_sql_query("""
        SELECT t.category_code, t.category_name, f.weather_category,
               COUNT(DISTINCT t.report_id) AS report_count
        FROM report_taxonomy t
        JOIN report_features f ON t.report_id = f.report_id
        JOIN report_types rt ON t.report_id = rt.report_id
        WHERE t.level = 'L1' AND rt.report_type = 'accident'
              AND f.weather_category IN ('VMC', 'IMC')
        GROUP BY t.category_code, t.category_name, f.weather_category
        ORDER BY report_count DESC
    """, conn)

    totals = pd.read_sql_query("""
        SELECT f.weather_category,
               COUNT(DISTINCT f.report_id) AS total_weather
        FROM report_features f
        JOIN report_types rt ON f.report_id = rt.report_id
        WHERE rt.report_type = 'accident'
              AND f.weather_category IN ('VMC', 'IMC')
        GROUP BY f.weather_category
    """, conn)
    conn.close()

    df = df.merge(totals, on="weather_category")
    df["prevalence_pct"] = (df["report_count"] / df["total_weather"] * 100).round(1)
    return df


def time_of_day_distribution() -> pd.DataFrame:
    """
    L1 categories split by Morning/Afternoon/Evening/Night.

    Returns: DataFrame with category_code, category_name, time_of_day,
             report_count, total_time, prevalence_pct.
    """
    conn = get_connection()
    df = pd.read_sql_query("""
        SELECT t.category_code, t.category_name, f.time_of_day,
               COUNT(DISTINCT t.report_id) AS report_count
        FROM report_taxonomy t
        JOIN report_features f ON t.report_id = f.report_id
        JOIN report_types rt ON t.report_id = rt.report_id
        WHERE t.level = 'L1' AND rt.report_type = 'accident'
              AND f.time_of_day IS NOT NULL
        GROUP BY t.category_code, t.category_name, f.time_of_day
        ORDER BY report_count DESC
    """, conn)

    totals = pd.read_sql_query("""
        SELECT f.time_of_day,
               COUNT(DISTINCT f.report_id) AS total_time
        FROM report_features f
        JOIN report_types rt ON f.report_id = rt.report_id
        WHERE rt.report_type = 'accident' AND f.time_of_day IS NOT NULL
        GROUP BY f.time_of_day
    """, conn)
    conn.close()

    df = df.merge(totals, on="time_of_day")
    df["prevalence_pct"] = (df["report_count"] / df["total_time"] * 100).round(1)
    return df


def multi_label_complexity() -> pd.DataFrame:
    """
    Distribution of categories-per-report (1, 2, 3, 4+).

    Returns: DataFrame with n_categories, report_count, pct.
    """
    conn = get_connection()
    df = pd.read_sql_query("""
        SELECT cnt AS n_categories, COUNT(*) AS report_count
        FROM (
            SELECT t.report_id, COUNT(DISTINCT t.category_code) AS cnt
            FROM report_taxonomy t
            JOIN report_types rt ON t.report_id = rt.report_id
            WHERE t.level = 'L1' AND rt.report_type = 'accident'
            GROUP BY t.report_id
        )
        GROUP BY cnt
        ORDER BY cnt
    """, conn)
    conn.close()

    total = df["report_count"].sum()
    df["pct"] = (df["report_count"] / total * 100).round(1)

    # Bucket 4+ together
    if not df.empty and df["n_categories"].max() > 4:
        below = df[df["n_categories"] <= 3]
        above = df[df["n_categories"] >= 4]
        bucket = pd.DataFrame([{
            "n_categories": "4+",
            "report_count": above["report_count"].sum(),
            "pct": above["pct"].sum(),
        }])
        df = pd.concat([below, bucket], ignore_index=True)
        df["n_categories"] = df["n_categories"].astype(str)

    return df


def bayesian_profile_comparison(profiles: list[dict]) -> pd.DataFrame:
    """
    Run the Bayesian model for each profile dict, return side-by-side comparison.

    Each dict in profiles should have keys matching model features:
    aircraft_category, season, region, weather_category, time_of_day.
    Plus a 'label' key for display.

    Returns: DataFrame with category rows and one column per profile label.
    """
    from risk_profiler.bayesian_model import load_model
    from riskradar.config import DB_PATH
    model = load_model(db_path=str(DB_PATH))

    results = {}
    for profile in profiles:
        label = profile.pop("label", str(profile))
        preds = model.predict(top_k=27, **profile)
        results[label] = {p["category_code"]: p["probability"] for p in preds}

    df = pd.DataFrame(results)
    df.index.name = "category_code"
    # Sort by max probability across profiles
    df["_max"] = df.max(axis=1)
    df = df.sort_values("_max", ascending=False).drop(columns=["_max"])
    return df


def category_by_aircraft_and_weather() -> pd.DataFrame:
    """
    Three-way cross-tab: aircraft_category x weather_category x top L1 categories.

    Returns long-form DataFrame with aircraft_category, weather_category,
    category_code, report_count.
    """
    conn = get_connection()
    df = pd.read_sql_query("""
        SELECT f.aircraft_category, f.weather_category, t.category_code,
               COUNT(DISTINCT t.report_id) AS report_count
        FROM report_taxonomy t
        JOIN report_features f ON t.report_id = f.report_id
        JOIN report_types rt ON t.report_id = rt.report_id
        WHERE t.level = 'L1' AND rt.report_type = 'accident'
              AND f.aircraft_category IS NOT NULL
              AND f.weather_category IN ('VMC', 'IMC')
        GROUP BY f.aircraft_category, f.weather_category, t.category_code
        ORDER BY f.aircraft_category, f.weather_category, report_count DESC
    """, conn)
    conn.close()
    return df
