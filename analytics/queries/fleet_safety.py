"""
analytics.queries.fleet_safety — Fleet Safety report queries.

Aircraft type + manufacturer + component failure analysis +
human factors + decade trends + weather-conditional risk.
"""

import pandas as pd
import numpy as np
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

    Values are prevalence percentages (% of that manufacturer's reports
    matching each category), so high-volume manufacturers don't overwhelm
    low-volume ones.  Douglas merged.
    """
    conn = get_connection()

    # Get top manufacturers with their totals (explicit GROUP BY on CASE)
    totals_df = pd.read_sql_query(f"""
        SELECT merged_make AS aircraft_make, SUM(cnt) AS total
        FROM (
            SELECT {_DOUGLAS_MERGE} AS merged_make,
                   COUNT(DISTINCT f.report_id) AS cnt
            FROM report_features f
            JOIN report_types rt ON f.report_id = rt.report_id
            WHERE rt.report_type = 'accident' AND f.aircraft_make IS NOT NULL
            GROUP BY f.aircraft_make
        )
        GROUP BY merged_make
        ORDER BY total DESC
        LIMIT {top_n}
    """, conn)
    top_makes = totals_df["aircraft_make"].tolist()
    totals_map = dict(zip(totals_df["aircraft_make"], totals_df["total"]))

    df = pd.read_sql_query(f"""
        SELECT merged_make AS aircraft_make, category_code,
               SUM(report_count) AS report_count
        FROM (
            SELECT {_DOUGLAS_MERGE} AS merged_make,
                   t.category_code,
                   COUNT(DISTINCT t.report_id) AS report_count
            FROM report_features f
            JOIN report_taxonomy t ON f.report_id = t.report_id
            JOIN report_types rt ON f.report_id = rt.report_id
            WHERE t.level = 'L1' AND rt.report_type = 'accident'
                  AND f.aircraft_make IS NOT NULL
            GROUP BY f.aircraft_make, t.category_code
        )
        GROUP BY merged_make, category_code
    """, conn)
    conn.close()

    df = df[df["aircraft_make"].isin(top_makes)]
    # Convert to prevalence percentage per manufacturer
    df["prevalence_pct"] = df.apply(
        lambda r: round(r["report_count"] / totals_map.get(r["aircraft_make"], 1) * 100, 1),
        axis=1,
    )
    pivot = df.pivot_table(
        index="aircraft_make", columns="category_code",
        values="prevalence_pct", fill_value=0,
    )
    # Reindex to preserve ranking order
    pivot = pivot.reindex(top_makes)
    return pivot


# ── New analytics: trends, human factors, weather risk ────────────────────


def key_risk_trends_by_decade(categories: list[str] | None = None) -> pd.DataFrame:
    """
    Decade-by-decade prevalence trends for key fleet safety categories.

    Default categories: LOC-I, CFIT, SCF-PP, SCF-NP, ICE, FUEL.
    Returns: decade, category_code, report_count, total_in_decade, prevalence_pct.
    """
    if categories is None:
        categories = ["LOC-I", "CFIT", "SCF-PP", "SCF-NP", "ICE", "FUEL"]
    placeholders = ",".join("?" for _ in categories)

    conn = get_connection()
    df = pd.read_sql_query(f"""
        SELECT f.decade, t.category_code,
               COUNT(DISTINCT t.report_id) AS report_count
        FROM report_taxonomy t
        JOIN report_features f ON t.report_id = f.report_id
        JOIN report_types rt ON t.report_id = rt.report_id
        WHERE t.level = 'L1' AND rt.report_type = 'accident'
              AND t.category_code IN ({placeholders})
              AND f.decade IS NOT NULL
        GROUP BY f.decade, t.category_code
        ORDER BY f.decade
    """, conn, params=categories)

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


def human_factors_breakdown() -> pd.DataFrame:
    """
    Aggregate human factors L2 subcategories across all parent categories.

    Combines HF-DECISION, HF-PERCEPTUAL, HF-SKILL, HF-VIOLATION counts
    regardless of which parent they were classified under.
    Returns: category_code, category_name, report_count, parent_breakdown (dict).
    """
    conn = get_connection()
    df = pd.read_sql_query("""
        SELECT t.category_code, t.category_name, t.parent_code,
               COUNT(DISTINCT t.report_id) AS report_count
        FROM report_taxonomy t
        JOIN report_types rt ON t.report_id = rt.report_id
        WHERE t.level = 'L2' AND rt.report_type = 'accident'
              AND t.category_code LIKE 'HF-%'
        GROUP BY t.category_code, t.category_name, t.parent_code
        ORDER BY t.category_code, report_count DESC
    """, conn)
    conn.close()

    # Aggregate across parents — a report may have HF-VIOLATION under both LOC-I and CFIT
    agg = df.groupby(["category_code", "category_name"])["report_count"].sum().reset_index()
    agg = agg.sort_values("report_count", ascending=False)
    return agg


def loci_subtypes_by_aircraft() -> pd.DataFrame:
    """
    LOC-I L2 subcategories broken down by aircraft category.

    Returns: aircraft_category, category_code, category_name, report_count,
             total_loci_in_type, pct_of_loci.
    """
    conn = get_connection()
    df = pd.read_sql_query("""
        SELECT f.aircraft_category, t.category_code, t.category_name,
               COUNT(DISTINCT t.report_id) AS report_count
        FROM report_taxonomy t
        JOIN report_features f ON t.report_id = f.report_id
        JOIN report_types rt ON t.report_id = rt.report_id
        WHERE t.level = 'L2' AND t.parent_code = 'LOC-I'
              AND rt.report_type = 'accident'
              AND f.aircraft_category IS NOT NULL
              AND t.category_code LIKE 'LOC-I-%'
        GROUP BY f.aircraft_category, t.category_code, t.category_name
    """, conn)

    totals = pd.read_sql_query("""
        SELECT f.aircraft_category,
               COUNT(DISTINCT t.report_id) AS total_loci_in_type
        FROM report_taxonomy t
        JOIN report_features f ON t.report_id = f.report_id
        JOIN report_types rt ON t.report_id = rt.report_id
        WHERE t.level = 'L1' AND t.category_code = 'LOC-I'
              AND rt.report_type = 'accident'
              AND f.aircraft_category IS NOT NULL
        GROUP BY f.aircraft_category
    """, conn)
    conn.close()

    df = df.merge(totals, on="aircraft_category", how="left")
    df["pct_of_loci"] = (df["report_count"] / df["total_loci_in_type"] * 100).round(1)
    return df


def weather_risk_ratios() -> pd.DataFrame:
    """
    Risk ratios for L1 categories: IMC vs VMC.

    risk_ratio = (category prevalence in IMC) / (category prevalence in VMC).
    A ratio > 1.0 means the category is overrepresented in IMC conditions.
    Returns: category_code, category_name, imc_count, vmc_count,
             imc_prevalence, vmc_prevalence, risk_ratio.
    """
    conn = get_connection()

    # Category counts by weather
    df = pd.read_sql_query("""
        SELECT f.weather_category, t.category_code, t.category_name,
               COUNT(DISTINCT t.report_id) AS report_count
        FROM report_taxonomy t
        JOIN report_features f ON t.report_id = f.report_id
        JOIN report_types rt ON t.report_id = rt.report_id
        WHERE t.level = 'L1' AND rt.report_type = 'accident'
              AND f.weather_category IN ('VMC', 'IMC')
        GROUP BY f.weather_category, t.category_code, t.category_name
    """, conn)

    # Total reports per weather condition
    totals = pd.read_sql_query("""
        SELECT f.weather_category,
               COUNT(DISTINCT f.report_id) AS total_reports
        FROM report_features f
        JOIN report_types rt ON f.report_id = rt.report_id
        WHERE rt.report_type = 'accident'
              AND f.weather_category IN ('VMC', 'IMC')
        GROUP BY f.weather_category
    """, conn)
    conn.close()

    totals_map = dict(zip(totals["weather_category"], totals["total_reports"]))
    imc_total = totals_map.get("IMC", 1)
    vmc_total = totals_map.get("VMC", 1)

    # Pivot to get IMC and VMC counts side by side
    imc = df[df["weather_category"] == "IMC"][["category_code", "category_name", "report_count"]].rename(
        columns={"report_count": "imc_count"}
    )
    vmc = df[df["weather_category"] == "VMC"][["category_code", "report_count"]].rename(
        columns={"report_count": "vmc_count"}
    )

    result = imc.merge(vmc, on="category_code", how="outer").fillna(0)
    result["imc_count"] = result["imc_count"].astype(int)
    result["vmc_count"] = result["vmc_count"].astype(int)
    result["imc_prevalence"] = (result["imc_count"] / imc_total * 100).round(1)
    result["vmc_prevalence"] = (result["vmc_count"] / vmc_total * 100).round(1)
    result["risk_ratio"] = np.where(
        result["vmc_prevalence"] > 0,
        (result["imc_prevalence"] / result["vmc_prevalence"]).round(2),
        np.nan,
    )
    result = result.sort_values("risk_ratio", ascending=False, na_position="last")
    return result


def human_factors_by_category() -> pd.DataFrame:
    """
    Cross-tabulation: HF subcategories × L1 parent categories.

    Shows how human factors distribute across risk categories.
    Returns: hf_code, hf_name, parent_code, report_count.
    """
    conn = get_connection()
    df = pd.read_sql_query("""
        SELECT t.category_code AS hf_code, t.category_name AS hf_name,
               t.parent_code,
               COUNT(DISTINCT t.report_id) AS report_count
        FROM report_taxonomy t
        JOIN report_types rt ON t.report_id = rt.report_id
        WHERE t.level = 'L2' AND rt.report_type = 'accident'
              AND t.category_code LIKE 'HF-%'
        GROUP BY t.category_code, t.category_name, t.parent_code
        ORDER BY report_count DESC
    """, conn)
    conn.close()
    return df


def component_failures_by_aircraft() -> pd.DataFrame:
    """
    SCF-PP and SCF-NP L2 subcategories by aircraft category.

    Returns: aircraft_category, parent_code, category_code, category_name,
             report_count.
    """
    conn = get_connection()
    df = pd.read_sql_query("""
        SELECT f.aircraft_category, t.parent_code, t.category_code,
               t.category_name,
               COUNT(DISTINCT t.report_id) AS report_count
        FROM report_taxonomy t
        JOIN report_features f ON t.report_id = f.report_id
        JOIN report_types rt ON t.report_id = rt.report_id
        WHERE t.level = 'L2' AND rt.report_type = 'accident'
              AND t.parent_code IN ('SCF-PP', 'SCF-NP')
              AND f.aircraft_category IS NOT NULL
        GROUP BY f.aircraft_category, t.parent_code, t.category_code, t.category_name
        ORDER BY f.aircraft_category, report_count DESC
    """, conn)
    conn.close()
    return df
