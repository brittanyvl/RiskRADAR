"""
Underwriting Risk Report — Aviation specialty insurance risk analysis.

Persona: Aviation underwriting analyst at a specialty insurer.
Core question: "What risk segments should I price differently?"
"""

import streamlit as st
import pandas as pd
import numpy as np
from app.components import data_loader as dl
from app.components.charts import (
    horizontal_bar, vertical_bar, grouped_bar, heatmap,
)
from app.components.report_layout import (
    page_header, kpi_row, section_divider, insight, coverage_note,
    sample_note, methodology_section, chart_with_insight, abbr,
)
from app.components.theme import (
    STEEL, CORAL, AMBER, TEAL, NAVY, CHART_PALETTE,
    TIME_COLORS, TIME_WINDOWS,
)


def render():
    """Render the full Underwriting Risk Report."""

    # ── Load all data up front ────────────────────────────────────────────
    summary = dl.dataset_summary()
    n_accidents = summary["accident_reports"]

    page_header(
        "Underwriting Risk Report",
        "Risk segmentation analysis for aviation specialty insurance portfolios",
    )

    # ── Section 1: KPI cards ──────────────────────────────────────────────
    _render_kpis(summary, n_accidents)
    section_divider()

    # ── Section 2: Category Co-occurrence ─────────────────────────────────
    _render_cooccurrence(n_accidents)
    section_divider()

    # ── Section 3: Geographic & Seasonal Risk ─────────────────────────────
    _render_region_season(n_accidents)
    section_divider()

    # ── Section 4: Weather Impact ─────────────────────────────────────────
    _render_weather(summary, n_accidents)
    section_divider()

    # ── Section 5: Time-of-Day Risk ───────────────────────────────────────
    _render_time_of_day(summary, n_accidents)
    section_divider()

    # ── Section 6: Multi-Label Complexity ─────────────────────────────────
    _render_complexity(n_accidents)
    section_divider()

    # ── Section 7: Bayesian Profile Comparisons ───────────────────────────
    _render_bayesian_profiles()
    section_divider()

    # ── Section 8: Methodology ────────────────────────────────────────────
    _render_methodology(summary, n_accidents)


# ══════════════════════════════════════════════════════════════════════════
# Section renderers
# ══════════════════════════════════════════════════════════════════════════


def _render_kpis(summary: dict, n_accidents: int):
    """Section 1: Headline KPI cards."""

    # Find highest-risk region-season combination
    rs = dl.region_season_matrix()
    max_val = rs.max().max()
    max_loc = rs.stack().idxmax()
    top_region_season = f"{max_loc[0]}, {max_loc[1]}"

    # Weather split
    weather_df = dl.vmc_imc_category_distribution()
    imc_share = 0
    vmc_share = 0
    if not weather_df.empty:
        imc_total = weather_df[weather_df["weather_category"] == "IMC"]["report_count"].sum()
        vmc_total = weather_df[weather_df["weather_category"] == "VMC"]["report_count"].sum()
        wx_total = imc_total + vmc_total
        if wx_total > 0:
            imc_share = imc_total / wx_total * 100
            vmc_share = vmc_total / wx_total * 100

    # High-complexity: reports with above-average category assignments
    complexity = dl.multi_label_complexity()
    four_plus = complexity[complexity["n_categories"] == "4+"]
    high_complexity_pct = four_plus["pct"].values[0] if not four_plus.empty else 0

    kpi_row([
        {
            "label": "High Complexity Reports",
            "value": f"{high_complexity_pct:.0f}%",
            "detail": "4+ categories assigned",
        },
        {
            "label": "Highest-Risk Segment",
            "value": top_region_season,
            "detail": f"{int(max_val)} reports",
        },
        {
            "label": "IMC Accident Share",
            "value": f"{imc_share:.0f}%",
            "detail": "Instrument conditions",
        },
        {
            "label": "VMC Accident Share",
            "value": f"{vmc_share:.0f}%",
            "detail": "Visual conditions",
        },
    ])


def _render_cooccurrence(n_accidents: int):
    """Section 2: Category co-occurrence heatmap (lower triangle only)."""

    st.markdown("### Risk Clustering: Which Hazards Travel Together?")
    st.markdown(
        "When two occurrence categories frequently appear in the same accident report, "
        "it signals a compounding risk pattern. Underwriters should consider these "
        "pairings when assessing aggregate exposure."
    )

    matrix = dl.cooccurrence_matrix()

    # Filter to top 12 categories by diagonal (self-count = total reports)
    diag = pd.Series(dict(zip(matrix.index, matrix.values.diagonal())), name="count")
    top12 = diag.nlargest(12).index.tolist()
    matrix_top = matrix.loc[top12, top12]

    fig = heatmap(
        matrix_top,
        title="Category Co-occurrence (Top 12 Categories)",
        mask_diagonal=True,
        lower_triangle_only=True,
        annotation_threshold=10,
        height=500,
    )

    # Find top co-occurring pairs (off-diagonal, lower triangle)
    vals = matrix_top.values.astype(float).copy()
    np.fill_diagonal(vals, 0)
    # Zero out upper triangle to only search lower
    vals[np.triu_indices_from(vals, k=1)] = 0
    flat_idx = np.argmax(vals)
    r, c = divmod(flat_idx, vals.shape[1])
    pair1, pair2 = matrix_top.index[r], matrix_top.columns[c]
    pair_count = int(vals[r, c])

    chart_with_insight(
        fig,
        f"The strongest co-occurrence is between <b>{pair1}</b> and <b>{pair2}</b>, "
        f"appearing together in {pair_count} reports. High co-occurrence pairs represent "
        f"compound risk: a policy covering both hazard types faces correlated loss potential. "
        f"The upper triangle is removed since the matrix is symmetric — each pair "
        f"appears once in the lower half.",
        chart_key="cooccurrence_heatmap",
    )
    sample_note(n_accidents)


def _render_region_season(n_accidents: int):
    """Section 3: Geographic and seasonal risk heatmap."""

    st.markdown("### Geographic and Seasonal Exposure")
    st.markdown(
        "Accident frequency varies significantly by region and season. "
        "These patterns help identify where and when portfolio exposure concentrates."
    )

    rs = dl.region_season_matrix()

    fig = heatmap(
        rs,
        title="Accident Reports by Region and Season",
        annotation_threshold=5,
        height=400,
    )

    # Find peak cell
    max_val = rs.max().max()
    max_loc = rs.stack().idxmax()

    # Find quietest
    min_val = rs.min().min()
    min_loc = rs.stack().idxmin()

    chart_with_insight(
        fig,
        f"The <b>{max_loc[0]}</b> region in <b>{max_loc[1]}</b> has the highest "
        f"concentration with {int(max_val)} accident reports, while "
        f"<b>{min_loc[0]}</b> in <b>{min_loc[1]}</b> is the lowest at {int(min_val)}. "
        f"Seasonal pricing adjustments should reflect these geographic patterns — "
        f"policies in the peak region/season may warrant a higher rate loading.",
        chart_key="region_season_heatmap",
    )
    sample_note(n_accidents)


def _render_weather(summary: dict, n_accidents: int):
    """Section 4: VMC vs IMC weather impact analysis."""

    st.markdown("### Weather Impact on Accident Types")
    st.markdown(
        "Visual Meteorological Conditions (VMC) and Instrument Meteorological Conditions "
        "(IMC) create fundamentally different risk profiles. Understanding which categories "
        "are weather-sensitive allows for more precise pricing of IFR-equipped vs VFR-only "
        "operations."
    )

    coverage_note("Weather", summary["weather_coverage_pct"], n_accidents)

    weather_df = dl.vmc_imc_category_distribution()

    if weather_df.empty:
        st.warning("Weather distribution data is not available.")
        return

    # Calculate IMC share for key underwriting categories — show as KPI cards
    imc_kpis = []
    for cat_code, cat_label in [
        ("CFIT", "Controlled Flight Into Terrain"),
        ("ICE", "Icing"),
        ("UIMC", "Unintended Flight in IMC"),
        ("LOC-I", "Loss of Control — In Flight"),
    ]:
        cat_data = weather_df[weather_df["category_code"] == cat_code]
        if not cat_data.empty:
            imc_row = cat_data[cat_data["weather_category"] == "IMC"]
            total_cat = cat_data["report_count"].sum()
            imc_count = imc_row["report_count"].values[0] if not imc_row.empty else 0
            imc_pct = imc_count / total_cat * 100 if total_cat > 0 else 0
            imc_kpis.append({
                "label": f"{cat_code} in IMC",
                "value": f"{imc_pct:.0f}%",
                "detail": cat_label,
            })

    if imc_kpis:
        kpi_row(imc_kpis)
        st.markdown("")

    # Top 8 categories by total reports across both conditions
    totals_by_cat = (
        weather_df.groupby("category_code")["report_count"]
        .sum()
        .nlargest(8)
        .index.tolist()
    )
    filtered = weather_df[weather_df["category_code"].isin(totals_by_cat)].copy()

    # Sort categories by total count for consistent ordering
    cat_order = (
        filtered.groupby("category_code")["report_count"]
        .sum()
        .sort_values(ascending=False)
        .index.tolist()
    )
    filtered["category_code"] = pd.Categorical(
        filtered["category_code"], categories=cat_order, ordered=True,
    )
    filtered = filtered.sort_values("category_code")

    fig = grouped_bar(
        filtered,
        x="category_code",
        y="report_count",
        group="weather_category",
        title="VMC vs IMC: Top 8 Accident Categories",
        colors={"VMC": STEEL, "IMC": AMBER},
        height=420,
    )

    chart_with_insight(
        fig,
        "Controlled Flight Into Terrain (CFIT) and Unintended Flight in Instrument "
        "Meteorological Conditions (UIMC) are heavily concentrated in IMC, confirming "
        "that instrument conditions are a primary driver for these categories. Policies "
        "covering Instrument Flight Rules (IFR) operations should price these risks "
        "accordingly. Conversely, Loss of Control — In Flight (LOC-I) occurs substantially "
        "in VMC, suggesting that pilot skill factors — not just weather — drive "
        "loss-of-control events.",
        chart_key="weather_grouped_bar",
    )


def _render_time_of_day(summary: dict, n_accidents: int):
    """Section 5: Time-of-day risk patterns."""

    st.markdown("### Time-of-Day Risk Patterns")
    st.markdown(
        "Accident risk shifts with the time of day. Night operations, in particular, "
        "carry elevated risk for certain categories. These patterns inform surcharges "
        "for nighttime flight operations."
    )
    # Show time window definitions
    windows_text = " · ".join(f"**{k}** {v}" for k, v in TIME_WINDOWS.items())
    st.caption(windows_text)

    coverage_note("Time-of-day", summary["time_coverage_pct"], n_accidents)

    time_df = dl.time_of_day_distribution()
    if time_df.empty:
        st.warning("Time-of-day distribution data is not available.")
        return

    # Top 6 categories by total time-classified reports
    totals_by_cat = (
        time_df.groupby("category_code")["report_count"]
        .sum()
        .nlargest(6)
        .index.tolist()
    )
    filtered = time_df[time_df["category_code"].isin(totals_by_cat)].copy()

    # Enforce time ordering
    time_order = ["Morning", "Afternoon", "Evening", "Night"]
    filtered["time_of_day"] = pd.Categorical(
        filtered["time_of_day"], categories=time_order, ordered=True,
    )
    filtered = filtered.sort_values(["category_code", "time_of_day"])

    # Sort categories by total count
    cat_order = (
        filtered.groupby("category_code")["report_count"]
        .sum()
        .sort_values(ascending=False)
        .index.tolist()
    )
    filtered["category_code"] = pd.Categorical(
        filtered["category_code"], categories=cat_order, ordered=True,
    )

    fig = grouped_bar(
        filtered,
        x="category_code",
        y="report_count",
        group="time_of_day",
        title="Accident Categories by Time of Day (Top 6)",
        colors=TIME_COLORS,
        height=420,
    )

    # Calculate night share for high-risk categories
    night_shares = {}
    for cat in totals_by_cat[:4]:
        cat_data = filtered[filtered["category_code"] == cat]
        total = cat_data["report_count"].sum()
        night = cat_data[cat_data["time_of_day"] == "Night"]["report_count"].sum()
        if total > 0:
            night_shares[cat] = night / total * 100

    night_insight_parts = []
    for cat, pct in sorted(night_shares.items(), key=lambda x: -x[1]):
        night_insight_parts.append(f"{cat} ({pct:.0f}%)")

    chart_with_insight(
        fig,
        f"Night operations (21:00–04:59 local) show elevated risk across multiple "
        f"categories. Night share of accidents by category: "
        f"{', '.join(night_insight_parts)}. "
        f"Afternoon flights account for the highest volume of accidents overall, "
        f"consistent with higher flight activity during daytime hours. "
        f"Underwriters should consider a night operations surcharge, especially "
        f"for categories with disproportionate nighttime occurrence.",
        chart_key="time_of_day_grouped",
    )


def _render_complexity(n_accidents: int):
    """Section 6: Multi-label complexity as severity proxy."""

    st.markdown("### Accident Complexity Distribution")
    st.markdown(
        "Each accident report is classified with one or more occurrence categories. "
        "Reports with many categories indicate complex, multi-factor events — often "
        "the most severe and costly losses. This distribution is a useful proxy for "
        "claim severity."
    )

    complexity = dl.multi_label_complexity()
    if complexity.empty:
        st.warning("Complexity data is not available.")
        return

    # Add descriptive labels
    complexity["label"] = complexity["n_categories"].astype(str).apply(
        lambda x: f"{x} category" if x == "1" else f"{x} categories"
    )

    fig = vertical_bar(
        complexity,
        x="label",
        y="report_count",
        title="Categories per Accident Report",
        color=STEEL,
        height=350,
    )

    # Calculate key stats
    single = complexity[complexity["n_categories"] == "1"]
    multi = complexity[complexity["n_categories"] != "1"]
    single_pct = single["pct"].values[0] if not single.empty else 0
    multi_pct = multi["pct"].sum() if not multi.empty else 0

    four_plus = complexity[complexity["n_categories"] == "4+"]
    four_plus_pct = four_plus["pct"].values[0] if not four_plus.empty else 0

    chart_with_insight(
        fig,
        f"About {multi_pct:.0f}% of accident reports involve multiple occurrence "
        f"categories, indicating complex chain-of-events scenarios. "
        f"Reports with 4 or more categories ({four_plus_pct:.0f}% of cases) "
        f"represent the highest-complexity events, which typically correlate with "
        f"hull loss or fatal outcomes. For pricing purposes, multi-category accidents "
        f"should be weighted more heavily in loss models.",
        chart_key="complexity_bar",
    )
    sample_note(n_accidents)


def _render_bayesian_profiles():
    """Section 7: Bayesian model profile comparisons."""

    st.markdown("### Risk Profile Comparisons")
    st.markdown(
        "The Bayesian risk model estimates the probability of each occurrence category "
        "given a set of operational characteristics. Below are four representative "
        "underwriting profiles — comparing their predicted risk distributions helps "
        "identify which operations merit differential pricing."
    )

    profiles = [
        {
            "label": "SE Piston, Summer, South, VMC, Afternoon",
            "aircraft_category": "single-piston",
            "season": "Summer",
            "region": "South",
            "weather_category": "VMC",
            "time_of_day": "Afternoon",
        },
        {
            "label": "Turboprop, Winter, Northeast, IMC, Night",
            "aircraft_category": "turboprop",
            "season": "Winter",
            "region": "Northeast",
            "weather_category": "IMC",
            "time_of_day": "Night",
        },
        {
            "label": "Helicopter, Summer, West, VMC, Morning",
            "aircraft_category": "helicopter",
            "season": "Summer",
            "region": "West",
            "weather_category": "VMC",
            "time_of_day": "Morning",
        },
        {
            "label": "SE Piston, Winter, Midwest, IMC, Night",
            "aircraft_category": "single-piston",
            "season": "Winter",
            "region": "Midwest",
            "weather_category": "IMC",
            "time_of_day": "Night",
        },
    ]

    try:
        comparison = dl.bayesian_profile_comparison(profiles)
    except Exception as e:
        st.warning(f"Could not load Bayesian model: {e}")
        return

    if comparison.empty:
        st.warning("No Bayesian model results available.")
        return

    # Show top 10 categories (most probable across any profile)
    top10 = comparison.head(10).copy()

    # Format as percentages for display
    styled = top10.copy()
    for col in styled.columns:
        styled[col] = styled[col].apply(lambda x: f"{x * 100:.1f}%")

    styled.index.name = "Category"

    st.markdown("#### Predicted Risk by Profile (Top 10 Categories)")
    st.markdown(
        "*Values show the estimated probability of each category appearing in an "
        "accident report for the given operational profile.*"
    )

    # Style the dataframe with highlighting
    def _highlight_high(val_str):
        """Highlight cells above 50% probability."""
        try:
            val = float(val_str.strip("%"))
            if val >= 50:
                return "background-color: #fdf0f0; font-weight: bold; color: #C44E52;"
            elif val >= 30:
                return "background-color: #fff8f0; font-weight: bold;"
            return ""
        except (ValueError, AttributeError):
            return ""

    st.dataframe(
        styled.style.map(_highlight_high),
        use_container_width=True,
        height=400,
    )

    # Build narrative from the data
    col_names = top10.columns.tolist()
    narratives = []
    for col in col_names:
        top_cat = top10[col].idxmax()
        top_prob = top10[col].max() * 100
        narratives.append(f"<b>{col}</b>: highest risk is {top_cat} ({top_prob:.0f}%)")

    insight(
        "Profile comparison highlights: " + ". ".join(narratives) + ". "
        "The winter / Instrument Meteorological Conditions (IMC) / night profiles show "
        "elevated Controlled Flight Into Terrain (CFIT) and Unintended Flight in IMC "
        "(UIMC) probabilities, while Visual Meteorological Conditions (VMC) daytime "
        "profiles skew toward Loss of Control (LOC-I) and mechanical failures. "
        "These differences quantify the basis for differential pricing across "
        "operational profiles.",
    )

    st.markdown(
        '<div class="coverage-note">'
        "Probabilities are generated by a Binary Relevance Naive Bayes model trained on "
        "431 accident reports with 5 features. See Methodology for calibration details."
        "</div>",
        unsafe_allow_html=True,
    )


def _render_methodology(summary: dict, n_accidents: int):
    """Section 8: Methodology and data notes."""

    methodology_section(f"""
**Data Source:** {summary['total_reports']} National Transportation Safety Board (NTSB)
aviation accident and incident reports (1966–present), of which {n_accidents} are
classified as accident reports and form the analysis population.

**Taxonomy:** Reports are classified using the CAST/ICAO Common Taxonomy Team (CICTT)
occurrence categories. Each report may be assigned multiple categories (multi-label
classification), reflecting the complex, multi-factor nature of aviation accidents.
The taxonomy includes 27 Level 1 categories and 32 Level 2 subcategories.

**Features:** Five operational features are extracted from each report:
- **Aircraft category** (single-piston, multi-piston, turboprop, helicopter, etc.)
- **Season** (Spring, Summer, Fall, Winter)
- **Region** (Northeast, South, Midwest, West — US Census regions)
- **Weather conditions** (VMC or IMC) — available for {summary['weather_coverage_pct']:.0f}% of reports
- **Time of day** (Morning, Afternoon, Evening, Night) — available for {summary['time_coverage_pct']:.0f}% of reports

**Bayesian Model:** Binary Relevance Naive Bayes classifier trained on {n_accidents}
accident reports. Each of the 27 categories has an independent binary classifier.
The model achieves an Expected Calibration Error (ECE) of 0.021, indicating well-calibrated
probability estimates. Laplace smoothing is applied to handle unseen feature combinations.

**Limitations:**
- Sample size of {n_accidents} reports limits statistical power for rare categories and
  multi-way cross-tabulations.
- Weather and time-of-day features have incomplete coverage; analyses using these features
  are based on the available subset.
- The Naive Bayes assumption of feature independence is a simplification — in practice,
  weather and time of day are correlated (e.g., IMC is more common at night).
- Historical data may not reflect current fleet composition or operational practices.
""")
