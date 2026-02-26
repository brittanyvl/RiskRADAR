"""
Underwriting Risk Report — Aviation specialty insurance risk analysis.

Persona: Aviation underwriting analyst at a specialty insurer.
Core question: "What risk segments should I price differently?"
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from app.components import data_loader as dl
from app.components.charts import (
    horizontal_bar, vertical_bar, grouped_bar, heatmap, line_chart,
    diverging_bar, _apply_layout,
)
from app.components.report_layout import (
    page_header, kpi_row, section_divider, insight, coverage_note,
    sample_note, methodology_section, chart_with_insight, ABBREVIATIONS,
)
from app.components.theme import (
    STEEL, CORAL, AMBER, TEAL, NAVY, CHART_PALETTE,
    TIME_COLORS, TIME_WINDOWS, HEATMAP_SCALE,
)


def render():
    """Render the full Underwriting Risk Report."""

    # ── Load all data up front ────────────────────────────────────────────
    summary = dl.dataset_summary()
    n_accidents = summary["accident_reports"]
    cat_counts = dl.category_counts()

    page_header(
        "Underwriting Risk Report",
        "Risk segmentation analysis for aviation specialty insurance portfolios. "
        "Identifies pricing factors, exposure concentrations, and loss-driver "
        "patterns across fleet segments.",
    )

    # ══════════════════════════════════════════════════════════════════════
    # KPIs
    # ══════════════════════════════════════════════════════════════════════
    _render_kpis(summary, n_accidents, cat_counts)

    sample_note(
        n_accidents,
        "NTSB accident reports classified with CAST/ICAO Common Taxonomy "
        "Team (CICTT) taxonomy",
    )

    # ══════════════════════════════════════════════════════════════════════
    # SECTIONS — ordered for underwriting storytelling flow:
    # 1. Fleet segmentation (primary rating axis)
    # 2-3. Condition modifiers (weather, night)
    # 4-5. Correlated & compound exposure
    # 6. Trend direction
    # 7. Geographic accumulation
    # 8. Severity synthesis (bridge to profiler)
    # 9. Interactive profiler (climax)
    # 10. Methodology (reference)
    # ══════════════════════════════════════════════════════════════════════

    section_divider()
    _render_aircraft_heatmap(n_accidents)                  # Section 1

    section_divider()
    _render_weather_pricing(summary, n_accidents)          # Section 2

    section_divider()
    _render_night_ops(summary, n_accidents)                # Section 3

    section_divider()
    _render_cooccurrence(cat_counts)                       # Section 4

    section_divider()
    _render_complexity(n_accidents)                        # Section 5

    section_divider()
    _render_decade_trends()                                # Section 6

    section_divider()
    _render_geographic(n_accidents)                        # Section 7

    section_divider()
    _render_severity_spectrum()                            # Section 8

    section_divider()
    _render_bayesian_profiles()                            # Section 9

    section_divider()
    _render_methodology(summary, n_accidents)              # Section 10


# ══════════════════════════════════════════════════════════════════════════
# Section renderers
# ══════════════════════════════════════════════════════════════════════════


def _render_kpis(summary: dict, n_accidents: int, cat_counts: pd.DataFrame):
    """KPI cards: 2 rows (4 + 2)."""

    # KPI 1: Fatal Category Rate (LOC-I + CFIT)
    fatal_rate = summary["loci_cfit_pct"]

    # KPI 2: Multi-Peril Exposure (3+ categories)
    complexity = dl.multi_label_complexity()
    three_plus_pct = 0.0
    if not complexity.empty:
        three_plus = complexity[complexity["n_categories"].isin(["3", "4+"])]
        three_plus_pct = three_plus["pct"].sum()

    # KPI 3: IMC Risk Multiplier (avg risk_ratio for top 5 weather-sensitive cats)
    wrr = dl.weather_risk_ratios()
    imc_multiplier = 0.0
    if not wrr.empty:
        top_imc = wrr[wrr["risk_ratio"] > 1].nlargest(5, "risk_ratio")
        if not top_imc.empty:
            imc_multiplier = top_imc["risk_ratio"].mean()

    # KPI 4: Night Ops Share (night share of LOC-I + CFIT + UIMC accidents)
    night_severity = dl.night_high_severity_share()
    night_hi_sev_pct = 0.0
    if not night_severity.empty:
        total_hi_sev = night_severity["report_count"].sum()
        night_row = night_severity[night_severity["time_of_day"] == "Night"]
        night_count = night_row["report_count"].values[0] if not night_row.empty else 0
        if total_hi_sev > 0:
            night_hi_sev_pct = night_count / total_hi_sev * 100

    kpi_row([
        {
            "label": "Fatal Category Rate",
            "value": f"{fatal_rate:.0f}%",
            "detail": "Of accidents involve the two most lethal categories (LOC-I, CFIT)",
            "accent": CORAL,
        },
        {
            "label": "Multi-Peril Exposure",
            "value": f"{three_plus_pct:.0f}%",
            "detail": "Of accidents have 3+ contributing factors (compound risk)",
            "accent": AMBER,
        },
        {
            "label": "IMC Risk Multiplier",
            "value": f"{imc_multiplier:.1f}x",
            "detail": "Average IMC overrepresentation for weather-sensitive categories",
            "accent": STEEL,
        },
        {
            "label": "Night Ops Severity",
            "value": f"{night_hi_sev_pct:.0f}%",
            "detail": "Of LOC-I and CFIT accidents occur during night operations",
            "accent": NAVY,
        },
    ])

    # KPI row 2: geographic concentration + top fleet segment
    rs = dl.region_season_matrix()
    region_totals = rs.sum(axis=1).sort_values(ascending=False)
    top_region = region_totals.index[0] if not region_totals.empty else "N/A"
    top_region_count = int(region_totals.iloc[0]) if not region_totals.empty else 0

    ac_risk_kpi = dl.risk_by_aircraft_category()
    top_segment = "N/A"
    top_segment_pct = 0.0
    if not ac_risk_kpi.empty:
        type_totals_kpi = ac_risk_kpi.groupby("aircraft_category")["total_in_category"].first()
        total_reports_kpi = type_totals_kpi.sum()
        if total_reports_kpi > 0:
            top_segment = type_totals_kpi.idxmax()
            top_segment_pct = type_totals_kpi.max() / total_reports_kpi * 100

    kpi_row([
        {
            "label": "Top Concentration Region",
            "value": top_region,
            "detail": f"{top_region_count} reports -- highest geographic exposure",
            "accent": TEAL,
        },
        {
            "label": "Top Fleet Segment",
            "value": f"{top_segment_pct:.0f}%",
            "detail": f"Of the portfolio is {top_segment} aircraft (highest-volume segment)",
            "accent": STEEL,
        },
    ])


def _render_severity_spectrum():
    """Section 8: Loss Severity Spectrum — quadrant bubble scatter."""

    st.markdown("### Loss Severity Spectrum")
    st.markdown(
        "Where does compound risk concentrate in this portfolio? Each bubble "
        "represents an occurrence category, positioned by how frequently it "
        "appears (x-axis) and how often it involves complex multi-factor losses "
        "with 4+ concurrent hazards (y-axis). Bubble size reflects the absolute "
        "number of compound loss events — the direct driver of aggregate severity."
    )

    severity = dl.severity_ranked_categories()
    if severity.empty:
        st.warning("Severity data is not available.")
        return

    # Derived columns
    severity["high_complexity_pct"] = (
        severity["high_complexity_count"] / severity["report_count"] * 100
    ).round(1)
    severity["category_label"] = severity["category_code"].map(
        lambda c: ABBREVIATIONS.get(c, c)
    )

    # Keep all rows for the expander table, top 15 for the chart
    severity_all = severity.copy()
    top = severity.nlargest(15, "report_count")

    # Quadrant thresholds (medians of plotted categories)
    med_x = top["report_count"].median()
    med_y = top["high_complexity_pct"].median()

    # Assign colors by quadrant
    colors = []
    for _, row in top.iterrows():
        if row["report_count"] >= med_x and row["high_complexity_pct"] >= med_y:
            colors.append(CORAL)    # Critical: high frequency + high compound rate
        elif row["report_count"] >= med_x or row["high_complexity_pct"] >= med_y:
            colors.append(AMBER)    # Elevated: high on one dimension
        else:
            colors.append(STEEL)    # Moderate: lower on both

    # Smart label placement to avoid overlap
    codes = top["category_code"].tolist()
    counts = top["report_count"].tolist()
    pcts = top["high_complexity_pct"].tolist()
    textpositions = []
    for i in range(len(codes)):
        pos = "top center"
        for j in range(len(codes)):
            if i != j:
                if (abs(counts[i] - counts[j]) < 12
                        and abs(pcts[i] - pcts[j]) < 4
                        and counts[i] <= counts[j]):
                    pos = "bottom center"
                    break
        textpositions.append(pos)

    # Build bubble scatter
    max_size = 45
    sizeref = 2.0 * top["high_complexity_count"].max() / (max_size ** 2)

    customdata = np.column_stack([
        top["category_label"].values,
        top["report_count"].values,
        top["high_complexity_count"].values,
        top["high_complexity_pct"].values,
        top["avg_complexity"].values,
    ])

    fig = go.Figure()

    # Legend traces (invisible markers for the three tiers)
    for label, color in [
        ("Critical — high frequency + high compound rate", CORAL),
        ("Elevated — high on one dimension", AMBER),
        ("Moderate — lower on both dimensions", STEEL),
    ]:
        fig.add_trace(go.Scatter(
            x=[None], y=[None], mode="markers",
            marker=dict(size=10, color=color),
            name=label, showlegend=True,
        ))

    # Main scatter trace
    fig.add_trace(go.Scatter(
        x=top["report_count"],
        y=top["high_complexity_pct"],
        mode="markers+text",
        text=top["category_code"],
        textposition=textpositions,
        textfont=dict(size=10, color=NAVY, family="Inter, sans-serif"),
        marker=dict(
            size=top["high_complexity_count"],
            sizemode="area",
            sizeref=sizeref,
            sizemin=8,
            color=colors,
            line=dict(width=1.5, color="white"),
            opacity=0.85,
        ),
        customdata=customdata,
        hovertemplate=(
            "<b>%{text}</b> — %{customdata[0]}<br>"
            "Frequency: %{customdata[1]} reports<br>"
            "Compound losses (4+ categories): %{customdata[2]} (%{customdata[3]:.0f}%)<br>"
            "Avg categories per event: %{customdata[4]:.1f}"
            "<extra></extra>"
        ),
        showlegend=False,
    ))

    # Quadrant lines
    fig.add_hline(
        y=med_y, line_dash="dot", line_color="#cbd5e1", line_width=1,
        layer="below",
    )
    fig.add_vline(
        x=med_x, line_dash="dot", line_color="#cbd5e1", line_width=1,
        layer="below",
    )

    # Quadrant corner annotations
    quadrant_annotations = [
        dict(x=0.98, y=0.98, xanchor="right", yanchor="top",
             text="HIGH FREQUENCY +<br>HIGH COMPOUND RISK"),
        dict(x=0.02, y=0.98, xanchor="left", yanchor="top",
             text="LOWER FREQUENCY<br>HIGH COMPOUND RISK"),
        dict(x=0.98, y=0.02, xanchor="right", yanchor="bottom",
             text="HIGH FREQUENCY<br>LOWER COMPOUND RISK"),
        dict(x=0.02, y=0.02, xanchor="left", yanchor="bottom",
             text="LOWER FREQUENCY +<br>LOWER COMPOUND RISK"),
    ]
    for ann in quadrant_annotations:
        fig.add_annotation(
            xref="paper", yref="paper",
            font=dict(size=9, color="#94a3b8"),
            showarrow=False, opacity=0.7,
            **ann,
        )

    # Apply consistent layout
    fig = _apply_layout(fig, "Compound Loss Landscape (Top 15 Categories)", 480)
    fig.update_layout(
        xaxis_title="Frequency (Accident Reports)",
        yaxis_title="Compound Loss Rate (%)",
        legend=dict(
            orientation="h", yanchor="top", y=-0.12,
            xanchor="center", x=0.5, font=dict(size=10),
            bgcolor="rgba(0,0,0,0)",
        ),
    )
    fig.update_xaxes(
        range=[0, top["report_count"].max() * 1.15],
        dtick=50, showgrid=True, zeroline=True, zerolinecolor="#dee2e6",
    )
    fig.update_yaxes(
        range=[top["high_complexity_pct"].min() - 8,
               top["high_complexity_pct"].max() + 5],
        dtick=5, ticksuffix="%", showgrid=True,
    )

    # Dynamic insight: identify critical quadrant categories
    critical = top[
        (top["report_count"] >= med_x) & (top["high_complexity_pct"] >= med_y)
    ].sort_values("report_count", ascending=False)

    critical_codes = critical["category_code"].tolist()
    if len(critical_codes) <= 3:
        critical_display = ", ".join(f"<b>{c}</b>" for c in critical_codes)
    else:
        critical_display = (
            ", ".join(f"<b>{c}</b>" for c in critical_codes[:3])
            + f", and {len(critical_codes) - 3} others"
        )

    worst = critical.loc[critical["high_complexity_pct"].idxmax()]
    insight_text = (
        f"{len(critical_codes)} categories concentrate in the critical quadrant "
        f"(upper right): {critical_display}. These combine high frequency with "
        f"the highest compound-loss rates in the portfolio. "
        f"<b>{worst['category_code']}</b> stands out — "
        f"{worst['high_complexity_pct']:.0f}% of its events involve 4+ concurrent "
        f"hazard categories ({int(worst['high_complexity_count'])} compound events), "
        f"signaling cascading failure scenarios where multiple coverage sections "
        f"are triggered simultaneously. Categories in this quadrant warrant the "
        f"most conservative severity assumptions in pricing and reserving."
    )

    chart_with_insight(fig, insight_text, insight_type="warning",
                       chart_key="uw_severity_spectrum")

    # Full dataset in expander
    with st.expander(f"View all {len(severity_all)} categories"):
        severity_all["high_complexity_pct"] = severity_all["high_complexity_pct"].round(1)
        severity_all["avg_complexity"] = severity_all["avg_complexity"].round(2)
        display_df = (
            severity_all
            .sort_values("report_count", ascending=False)
            [["category_code", "category_label", "report_count",
              "high_complexity_count", "high_complexity_pct", "avg_complexity"]]
            .rename(columns={
                "category_code": "Code",
                "category_label": "Category",
                "report_count": "Total Reports",
                "high_complexity_count": "Compound Losses (4+)",
                "high_complexity_pct": "Compound Rate (%)",
                "avg_complexity": "Avg Categories",
            })
        )
        st.dataframe(
            display_df,
            use_container_width=True,
            hide_index=True,
            height=min(35 * len(display_df) + 38, 500),
        )
        st.caption(
            "Compound losses are reports with 4+ concurrent CICTT categories. "
            "The chart above shows the top 15 categories by frequency; "
            "this table includes all categories identified in the dataset."
        )


def _render_aircraft_heatmap(n_accidents: int):
    """Section 2: Risk Segmentation by Aircraft Type."""

    st.markdown("### Risk Segmentation by Aircraft Type")
    st.markdown(
        "How does the risk profile shift across fleet segments? Each cell shows "
        "the prevalence (%) of a category within that aircraft type's accident "
        "reports. This single view enables segment-specific pricing decisions."
    )

    ac_data = dl.category_by_feature("aircraft_category")
    if ac_data.empty:
        st.warning("Aircraft category data is not available.")
        return

    # Compute prevalence: normalize each row by total reports for that aircraft type
    # Get per-aircraft-type totals
    ac_risk = dl.risk_by_aircraft_category()
    if ac_risk.empty:
        st.warning("Aircraft risk data is not available.")
        return

    type_totals = (
        ac_risk.groupby("aircraft_category")["total_in_category"]
        .first()
    )

    # Build prevalence matrix
    prevalence = ac_data.copy()
    for aircraft_type in prevalence.index:
        total = type_totals.get(aircraft_type, 1)
        prevalence.loc[aircraft_type] = (
            prevalence.loc[aircraft_type] / total * 100
        ).round(1)

    # Filter to top 8 categories by total report count
    top_cats = (
        dl.category_counts()
        .head(8)["category_code"]
        .tolist()
    )
    visible_cats = [c for c in top_cats if c in prevalence.columns]
    prevalence = prevalence[visible_cats]

    # Filter out aircraft types with very few reports
    min_reports = 5
    keep_types = type_totals[type_totals >= min_reports].index.tolist()
    prevalence = prevalence.loc[prevalence.index.isin(keep_types)]

    # Sort by total reports descending
    sort_order = type_totals.reindex(prevalence.index).sort_values(ascending=False).index
    prevalence = prevalence.reindex(sort_order)

    fig = heatmap(
        prevalence,
        title="Aircraft Type x Category Prevalence (%)",
        height=max(400, len(prevalence) * 40 + 80),
        value_format="pct",
        colorbar_title="Prevalence %",
        hover_labels=ABBREVIATIONS,
    )

    # Find the highest-prevalence cell
    max_val = prevalence.max().max()
    max_col = prevalence.max().idxmax()
    max_row = prevalence[max_col].idxmax()
    portfolio_avg = prevalence[max_col].mean()

    chart_with_insight(
        fig,
        f"<b>{max_row}</b> shows the highest exposure to "
        f"<b>{ABBREVIATIONS.get(max_col, max_col)}</b> at {max_val:.0f}% prevalence "
        f"(vs {portfolio_avg:.0f}% portfolio average). Policies covering {max_row} "
        f"aircraft should be priced to reflect this elevated risk. Each cell is "
        f"normalized by the total accident reports for that aircraft type, enabling "
        f"fair comparison across segments of different sizes.",
        chart_key="uw_aircraft_heatmap",
    )


def _render_weather_pricing(summary: dict, n_accidents: int):
    """Section 3: Weather Pricing Factors (IMC vs VMC)."""

    st.markdown("### Weather Pricing Factors")
    st.markdown(
        "What premium adjustments does weather justify? This chart compares "
        "category prevalence in IMC (poor weather) versus VMC (clear weather). "
        "The divergence directly quantifies the weather surcharge for each "
        "risk category."
    )

    coverage_note("Weather", summary["weather_coverage_pct"], n_accidents)

    # IMC-share KPI cards for key underwriting categories
    weather_df = dl.vmc_imc_category_distribution()
    if not weather_df.empty:
        imc_kpis = []
        imc_accents = {"CFIT": CORAL, "ICE": AMBER, "UIMC": NAVY, "LOC-I": CORAL}
        for cat_code in ["CFIT", "ICE", "UIMC", "LOC-I"]:
            cat_data = weather_df[weather_df["category_code"] == cat_code]
            if not cat_data.empty:
                imc_row = cat_data[cat_data["weather_category"] == "IMC"]
                total_cat = cat_data["report_count"].sum()
                imc_count = (
                    imc_row["report_count"].values[0] if not imc_row.empty else 0
                )
                imc_pct = imc_count / total_cat * 100 if total_cat > 0 else 0
                imc_kpis.append({
                    "label": f"{cat_code} in IMC",
                    "value": f"{imc_pct:.0f}%",
                    "detail": ABBREVIATIONS.get(cat_code, cat_code),
                    "accent": imc_accents.get(cat_code, STEEL),
                })
        if imc_kpis:
            kpi_row(imc_kpis)
            st.markdown("")

    wrr = dl.weather_risk_ratios()
    if wrr.empty:
        st.warning("Weather risk data is not available.")
        return

    # Filter to categories with meaningful combined sample
    wrr_filtered = wrr[
        (wrr["imc_count"] + wrr["vmc_count"] >= 10)
        & (wrr["risk_ratio"].notna())
    ].copy()

    if wrr_filtered.empty:
        st.info("Insufficient weather data for comparison.")
        return

    wrr_filtered["category_label"] = wrr_filtered["category_code"].map(
        lambda c: ABBREVIATIONS.get(c, c)
    )

    # Sort by difference (IMC - VMC) for visual impact
    wrr_filtered["diff"] = (
        wrr_filtered["imc_prevalence"] - wrr_filtered["vmc_prevalence"]
    )
    wrr_filtered = wrr_filtered.sort_values("diff", ascending=True)

    fig = diverging_bar(
        wrr_filtered,
        y="category_label",
        left_col="vmc_prevalence",
        right_col="imc_prevalence",
        left_label="VMC (clear weather)",
        right_label="IMC (poor weather)",
        left_color=STEEL,
        right_color=CORAL,
        title="Category Prevalence: VMC vs IMC (Pricing Factors)",
        height=max(450, len(wrr_filtered) * 30 + 80),
    )

    most_imc = wrr_filtered.iloc[-1]
    most_vmc = wrr_filtered.iloc[0]
    chart_with_insight(
        fig,
        f"<b>{most_imc['category_label']}</b> is {most_imc['risk_ratio']:.1f}x more "
        f"prevalent in IMC than VMC, the largest weather-driven risk multiplier. For "
        f"an underwriter, this translates to a concrete pricing factor: policies "
        f"covering frequent IMC operations in {most_imc['category_code']}-prone "
        f"segments should carry a proportional surcharge. Categories that skew VMC "
        f"(left side), like <b>{most_vmc['category_label']}</b>, are primarily "
        f"piloting-skill risks rather than weather risks.",
        chart_key="uw_weather_diverge",
    )


def _render_night_ops(summary: dict, n_accidents: int):
    """Section 4: Night Operations Risk Premium."""

    st.markdown("### Night Operations Risk Premium")
    st.markdown(
        "What surcharge should night operations carry? Night flying introduces "
        "unique risks -- reduced visibility, fatigue, and spatial disorientation. "
        "This section quantifies the disproportionate nighttime occurrence of "
        "high-severity categories."
    )

    # Show time window definitions
    windows_text = " . ".join(f"**{k}** {v}" for k, v in TIME_WINDOWS.items())
    st.caption(windows_text)

    coverage_note("Time-of-day", summary["time_coverage_pct"], n_accidents)

    time_df = dl.time_of_day_distribution()
    if time_df.empty:
        st.warning("Time-of-day distribution data is not available.")
        return

    # Mini KPI row: night share of key high-severity categories
    night_kpis = []
    for cat_code in ["LOC-I", "CFIT", "UIMC", "ICE"]:
        cat_data = time_df[time_df["category_code"] == cat_code]
        if not cat_data.empty:
            total_cat = cat_data["report_count"].sum()
            night_row = cat_data[cat_data["time_of_day"] == "Night"]
            night_count = (
                night_row["report_count"].values[0] if not night_row.empty else 0
            )
            night_pct = night_count / total_cat * 100 if total_cat > 0 else 0
            night_kpis.append({
                "label": f"{cat_code} Night Share",
                "value": f"{night_pct:.0f}%",
                "detail": ABBREVIATIONS.get(cat_code, cat_code),
                "accent": NAVY,
            })

    if night_kpis:
        kpi_row(night_kpis)
        st.markdown("")

    # Grouped bar: top 6 categories by time of day
    totals_by_cat = (
        time_df.groupby("category_code")["report_count"]
        .sum()
        .nlargest(6)
        .index.tolist()
    )
    filtered = time_df[time_df["category_code"].isin(totals_by_cat)].copy()

    time_order = ["Morning", "Afternoon", "Evening", "Night"]
    filtered["time_of_day"] = pd.Categorical(
        filtered["time_of_day"], categories=time_order, ordered=True,
    )
    filtered = filtered.sort_values(["category_code", "time_of_day"])

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

    # Build night share insight from data
    overall_night = time_df[time_df["time_of_day"] == "Night"]["report_count"].sum()
    overall_total = time_df["report_count"].sum()
    overall_night_pct = (
        overall_night / overall_total * 100 if overall_total > 0 else 0
    )

    night_shares_text = []
    for cat in ["CFIT", "UIMC", "LOC-I"]:
        cat_data = time_df[time_df["category_code"] == cat]
        if not cat_data.empty:
            total_cat = cat_data["report_count"].sum()
            night_cat = cat_data[cat_data["time_of_day"] == "Night"][
                "report_count"
            ].sum()
            if total_cat > 0:
                pct = night_cat / total_cat * 100
                night_shares_text.append(f"{cat} ({pct:.0f}%)")

    chart_with_insight(
        fig,
        f"Night operations represent a disproportionate share of high-severity "
        f"categories: {', '.join(night_shares_text)} of each category's accidents "
        f"occur at night -- significantly above the overall night share of "
        f"{overall_night_pct:.0f}%. This quantifies the night operations surcharge: "
        f"categories with disproportionate nighttime occurrence justify premium "
        f"loading for night-approved policies.",
        chart_key="uw_night_ops",
    )


def _render_geographic(n_accidents: int):
    """Section 5: Geographic Exposure Concentration."""

    st.markdown("### Geographic Exposure Concentration")
    st.markdown(
        "Where is portfolio risk concentrated? Geographic clustering creates "
        "accumulation risk for insurers. Understanding not just *where* but "
        "*what* risks concentrate in each region informs geographic loading factors."
    )

    # Heatmap 1: Region x Season
    rs = dl.region_season_matrix()
    if rs.empty:
        st.warning("Region/season data is not available.")
        return

    fig_rs = heatmap(
        rs,
        title="Accident Reports by Region and Season",
        annotation_threshold=5,
        height=400,
    )

    max_val = rs.max().max()
    max_loc = rs.stack().idxmax()
    region_totals = rs.sum(axis=1)
    top_region = region_totals.idxmax()
    top_region_pct = region_totals.max() / region_totals.sum() * 100

    chart_with_insight(
        fig_rs,
        f"The <b>{max_loc[0]}</b> region in <b>{max_loc[1]}</b> has the highest "
        f"concentration with {int(max_val)} accident reports. Overall, "
        f"<b>{top_region}</b> accounts for {top_region_pct:.0f}% of the accident "
        f"portfolio. Underwriters with heavy {top_region} exposure should monitor "
        f"accumulation risk and consider geographic loading factors.",
        chart_key="uw_region_season",
    )

    # Heatmap 2: Region x Top Categories (prevalence %)
    st.markdown("#### Regional Category Profiles")
    region_cats = dl.category_by_feature("region")
    if not region_cats.empty:
        # Normalize by total reports per region (from region-season matrix)
        region_report_totals = rs.sum(axis=1)
        region_prev = region_cats.copy()
        for region in region_prev.index:
            total = region_report_totals.get(region, 1)
            region_prev.loc[region] = (region_prev.loc[region] / total * 100).round(1)

        # Top 8 categories by total
        top_cat_codes = (
            dl.category_counts()
            .head(8)["category_code"]
            .tolist()
        )
        visible = [c for c in top_cat_codes if c in region_prev.columns]
        region_prev = region_prev[visible]

        fig_rc = heatmap(
            region_prev,
            title="Region x Category Prevalence (%)",
            height=max(300, len(region_prev) * 45 + 80),
            value_format="pct",
            colorbar_title="Prevalence %",
            hover_labels=ABBREVIATIONS,
        )

        # Find the most distinctive region-category combination
        rc_max_val = region_prev.max().max()
        rc_max_col = region_prev.max().idxmax()
        rc_max_row = region_prev[rc_max_col].idxmax()
        rc_avg = region_prev[rc_max_col].mean()

        chart_with_insight(
            fig_rc,
            f"<b>{rc_max_row}</b> shows the highest relative concentration of "
            f"<b>{ABBREVIATIONS.get(rc_max_col, rc_max_col)}</b> at {rc_max_val:.0f}% "
            f"prevalence (vs {rc_avg:.0f}% national average). Understanding which risks "
            f"dominate in which regions enables geographic pricing differentiation.",
            chart_key="uw_region_categories",
        )

    sample_note(n_accidents)


def _render_complexity(n_accidents: int):
    """Section 6: Multi-Peril Complexity Distribution."""

    st.markdown("### Multi-Peril Complexity Distribution")
    st.markdown(
        "How complex are the losses in this portfolio? Each accident report is "
        "classified with one or more occurrence categories. Reports with many "
        "categories indicate complex, multi-factor events -- often the most severe "
        "and costly losses. This distribution is a key input for loss severity modeling."
    )

    complexity = dl.multi_label_complexity()
    if complexity.empty:
        st.warning("Complexity data is not available.")
        return

    # Vertical bar: complexity distribution
    complexity["label"] = complexity["n_categories"].astype(str).apply(
        lambda x: f"{x} category" if x == "1" else f"{x} categories"
    )

    four_plus = complexity[complexity["n_categories"] == "4+"]
    four_plus_pct = four_plus["pct"].values[0] if not four_plus.empty else 0
    four_plus_count = (
        int(four_plus["report_count"].values[0]) if not four_plus.empty else 0
    )

    single = complexity[complexity["n_categories"] == "1"]
    single_pct = single["pct"].values[0] if not single.empty else 0
    multi_pct = 100 - single_pct

    # Two-column layout: chart on left, summary stats on right
    col_chart, col_stats = st.columns([2, 1])

    with col_chart:
        fig_dist = vertical_bar(
            complexity,
            x="label",
            y="report_count",
            title="Categories per Accident Report",
            color=AMBER,
            height=350,
        )
        st.plotly_chart(fig_dist, use_container_width=True, key="uw_complexity_dist")

    with col_stats:
        st.markdown("#### Key Stats")
        st.markdown(
            f"**{multi_pct:.0f}%** of accidents involve multiple "
            f"categories, indicating complex chain-of-events scenarios."
        )
        st.markdown(
            f"**{four_plus_pct:.0f}%** ({four_plus_count} reports) have "
            f"4+ categories -- the most complex and costly loss events."
        )
        st.markdown(
            f"**{single_pct:.0f}%** are single-category -- simpler events "
            f"with more predictable loss profiles."
        )
        st.markdown("")
        st.markdown(
            '*For pricing, multi-category accidents should be weighted more '
            'heavily in loss models -- they represent compound risk where '
            'multiple policy sections may trigger simultaneously.*'
        )

    # Horizontal bar: top categories in 4+ label reports
    st.markdown("#### Complex Loss Drivers (4+ Category Reports)")
    high_complex = dl.high_complexity_categories()
    if not high_complex.empty:
        high_complex_top = high_complex.head(10).copy()
        high_complex_top["category_label"] = high_complex_top["category_code"].map(
            lambda c: ABBREVIATIONS.get(c, c)
        )

        fig_hc = horizontal_bar(
            high_complex_top,
            x="report_count",
            y="category_label",
            title="Most Frequent Categories in High-Complexity Accidents",
            color=CORAL,
            height=max(350, len(high_complex_top) * 30 + 60),
            show_values=True,
        )

        top1 = high_complex_top.iloc[0]
        top2 = high_complex_top.iloc[1] if len(high_complex_top) > 1 else top1
        chart_with_insight(
            fig_hc,
            f"Among high-complexity accidents, <b>{ABBREVIATIONS.get(top1['category_code'], top1['category_code'])}</b> "
            f"({int(top1['report_count'])} reports) and "
            f"<b>{ABBREVIATIONS.get(top2['category_code'], top2['category_code'])}</b> "
            f"({int(top2['report_count'])} reports) appear most frequently -- they are "
            f"the nucleus around which cascading failures develop. Loss reserves for "
            f"these multi-peril events should reflect the compound nature of the claims.",
            chart_key="uw_complex_drivers",
        )
    else:
        st.info("No high-complexity category data available.")

    sample_note(n_accidents)


def _render_cooccurrence(cat_counts: pd.DataFrame):
    """Section 7: Co-occurrence Risk Clustering."""

    st.markdown("### Co-occurrence Risk Clustering")
    st.markdown(
        "Which hazards travel together? When two categories frequently co-occur, "
        "a policy covering one hazard is implicitly exposed to the other. This "
        "correlated loss exposure should be reflected in aggregate risk modeling."
    )

    matrix = dl.cooccurrence_matrix()
    if matrix.empty:
        st.warning("Co-occurrence data is not available.")
        return

    # Filter to top 10 categories for readability
    top_10_codes = cat_counts.head(10)["category_code"].tolist()
    coocc_filtered = matrix.loc[
        matrix.index.isin(top_10_codes),
        matrix.columns.isin(top_10_codes),
    ]
    coocc_filtered = coocc_filtered.reindex(
        index=top_10_codes, columns=top_10_codes,
    )

    # Wrap long axis labels with <br> for horizontal display
    wrapped_labels = {
        code: ABBREVIATIONS.get(code, code).replace(" — ", "<br>").replace(" ", "<br>", 1)
        if len(ABBREVIATIONS.get(code, code)) > 12
        else ABBREVIATIONS.get(code, code)
        for code in top_10_codes
    }
    coocc_display = coocc_filtered.copy()
    coocc_display.columns = [wrapped_labels.get(c, c) for c in coocc_display.columns]
    coocc_display.index = [wrapped_labels.get(c, c) for c in coocc_display.index]

    fig = heatmap(
        coocc_display,
        title="Category Co-occurrence (Top 10 Categories)",
        mask_diagonal=True,
        lower_triangle_only=True,
        annotation_threshold=0,
        height=520,
        colorbar_title="Shared Reports",
    )
    # Override default 45° x-axis to horizontal
    fig.update_xaxes(tickangle=0)

    # Find top 3 co-occurring pairs
    vals = coocc_filtered.values.astype(float).copy()
    np.fill_diagonal(vals, 0)
    vals[np.triu_indices_from(vals, k=1)] = 0

    top_pairs = []
    vals_flat = vals.copy()
    for _ in range(3):
        flat_idx = np.argmax(vals_flat)
        r, c = divmod(flat_idx, vals_flat.shape[1])
        if vals_flat[r, c] == 0:
            break
        cat1 = coocc_filtered.index[r]
        cat2 = coocc_filtered.columns[c]
        count = int(vals_flat[r, c])
        top_pairs.append((cat1, cat2, count))
        vals_flat[r, c] = 0

    if top_pairs:
        pair_text = ", ".join(
            f"{p[0]}-{p[1]} ({p[2]} reports)" for p in top_pairs
        )
        first = top_pairs[0]
        total_reports_cat1 = int(
            coocc_filtered.loc[first[0], first[0]]
            if first[0] in coocc_filtered.index
            else matrix.loc[first[0], first[0]]
        )
        co_pct = (
            first[2] / total_reports_cat1 * 100
            if total_reports_cat1 > 0
            else 0
        )

        chart_with_insight(
            fig,
            f"<b>{first[0]}</b> and <b>{first[1]}</b> co-occur in {first[2]} reports, "
            f"the strongest pairing. For underwriters, this means that a policy "
            f"triggered by a {first[0]} event has a {co_pct:.0f}% probability of also "
            f"involving {first[1]} -- correlated exposure that should be reflected in "
            f"aggregate risk modeling. The top 3 co-occurrence pairs are: {pair_text}.",
            chart_key="uw_cooccurrence",
        )
    else:
        st.plotly_chart(fig, use_container_width=True, key="uw_cooccurrence")

    # Expandable acronym reference
    abbr_ref = " . ".join(
        f"**{code}** = {ABBREVIATIONS.get(code, code)}"
        for code in top_10_codes
        if code in ABBREVIATIONS
    )
    with st.expander("Category Acronym Reference"):
        st.markdown(abbr_ref, unsafe_allow_html=True)


def _render_decade_trends():
    """Section 8: High-Severity Category Trends by Decade."""

    st.markdown("### High-Severity Category Trends by Decade")
    st.markdown(
        "Are the most expensive risks growing or shrinking? Tracking decade-over-decade "
        "prevalence for the highest-severity, most pricing-relevant categories informs "
        "long-term pricing strategy. Rising categories suggest rate hardening; declining "
        "categories support rate stability."
    )

    underwriting_cats = ["LOC-I", "CFIT", "SCF-PP", "ICE", "FUEL", "UIMC"]
    trends = dl.category_prevalence_by_decade(categories=underwriting_cats)

    if trends.empty:
        st.info("No decade trend data available.")
        return

    # Map codes to full names for legend
    trends["category_label"] = trends["category_code"].map(
        lambda c: f"{c} ({ABBREVIATIONS.get(c, c)})"
    )

    fig = line_chart(
        trends,
        x="decade",
        y="prevalence_pct",
        color="category_label",
        title="High-Severity Category Prevalence by Decade",
        height=450,
        y_label="Prevalence (%)",
    )

    # Identify biggest movers
    decades = sorted(trends["decade"].unique())
    if len(decades) >= 2:
        earliest = trends[trends["decade"] == decades[0]]
        latest = trends[trends["decade"] == decades[-1]]
        merged = earliest[["category_code", "prevalence_pct"]].merge(
            latest[["category_code", "prevalence_pct"]],
            on="category_code",
            suffixes=("_early", "_late"),
        )
        merged["change"] = (
            merged["prevalence_pct_late"] - merged["prevalence_pct_early"]
        )

        if not merged.empty:
            biggest_rise = merged.loc[merged["change"].idxmax()]
            biggest_drop = merged.loc[merged["change"].idxmin()]

            rising_name = ABBREVIATIONS.get(
                biggest_rise["category_code"], biggest_rise["category_code"]
            )
            declining_name = ABBREVIATIONS.get(
                biggest_drop["category_code"], biggest_drop["category_code"]
            )

            trend_insight = (
                f"Over the last {len(decades)} decades, "
                f"<b>{rising_name}</b> prevalence has "
                f"{'increased' if biggest_rise['change'] > 0 else 'decreased'} by "
                f"{abs(biggest_rise['change']):.0f} percentage points, while "
                f"<b>{declining_name}</b> showed a "
                f"{abs(biggest_drop['change']):.0f} pp decline. "
                f"For long-term pricing, rising categories suggest rate hardening may "
                f"be warranted, while declining categories support rate stability. "
                f"Note: earlier decades have fewer reports, so trend confidence "
                f"improves in recent periods."
            )
        else:
            trend_insight = "Insufficient data for trend comparison."
    else:
        trend_insight = "Insufficient decades for trend comparison."

    chart_with_insight(fig, trend_insight, chart_key="uw_decade_trends")


def _render_bayesian_profiles():
    """Section 9: Bayesian Risk Profiling (hero visualization)."""

    st.markdown("### Bayesian Risk Profiling")
    st.markdown(
        "How do operational profiles compare for pricing? The Bayesian model "
        "estimates the probability of each occurrence category given a set of "
        "operational characteristics. Build a custom profile below and compare "
        "it against three benchmark segments."
    )

    # ── Interactive Profile Builder ───────────────────────────────────────
    st.markdown("#### Build a Custom Risk Profile")
    st.markdown(
        "Select operational characteristics to generate a risk estimate, "
        "compared against three benchmark profiles."
    )

    c1, c2, c3, c4, c5 = st.columns(5)
    with c1:
        custom_aircraft = st.selectbox(
            "Aircraft Type",
            options=[
                "single-piston", "multi-piston", "turboprop",
                "jet-narrow", "jet-regional", "jet-wide",
                "helicopter",
            ],
            index=0,
            key="uw_aircraft",
        )
    with c2:
        custom_season = st.selectbox(
            "Season",
            options=["Spring", "Summer", "Fall", "Winter"],
            index=1,
            key="uw_season",
        )
    with c3:
        custom_region = st.selectbox(
            "Region",
            options=["Northeast", "South", "Midwest", "West"],
            index=1,
            key="uw_region",
        )
    with c4:
        custom_weather = st.selectbox(
            "Weather",
            options=["VMC", "IMC"],
            index=0,
            key="uw_weather",
        )
    with c5:
        custom_time = st.selectbox(
            "Time of Day",
            options=["Morning", "Afternoon", "Evening", "Night"],
            index=1,
            key="uw_time",
        )

    # Build the custom profile label from selections
    weather_short = custom_weather
    custom_label = (
        f"Custom ({custom_aircraft}, {custom_season}, "
        f"{custom_region}, {weather_short}, {custom_time})"
    )

    # 3 fixed benchmark profiles + 1 custom
    profiles = [
        {
            "label": "GA Day VFR",
            "aircraft_category": "single-piston",
            "season": "Summer",
            "region": "South",
            "weather_category": "VMC",
            "time_of_day": "Afternoon",
        },
        {
            "label": "Corporate IFR Night",
            "aircraft_category": "turboprop",
            "season": "Winter",
            "region": "Northeast",
            "weather_category": "IMC",
            "time_of_day": "Night",
        },
        {
            "label": "Helicopter Ops",
            "aircraft_category": "helicopter",
            "season": "Summer",
            "region": "West",
            "weather_category": "VMC",
            "time_of_day": "Morning",
        },
        {
            "label": custom_label,
            "aircraft_category": custom_aircraft,
            "season": custom_season,
            "region": custom_region,
            "weather_category": custom_weather,
            "time_of_day": custom_time,
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

    # Show top 10 categories by max probability across profiles
    top10 = comparison.head(10).copy()

    # Convert probabilities to percentages for the heatmap
    top10_pct = (top10 * 100).round(1)

    # Wrap long profile names for horizontal x-axis display
    def _wrap_profile_name(name: str) -> str:
        """Wrap profile names at natural break points for horizontal display."""
        if len(name) <= 18:
            return name
        # Wrap at comma or parenthesis
        if ", " in name:
            parts = name.split(", ", 1)
            return parts[0] + ",<br>" + parts[1]
        return name

    top10_pct_display = top10_pct.rename(
        columns={c: _wrap_profile_name(c) for c in top10_pct.columns}
    )

    # Build the heatmap matrix: categories (rows) x profiles (columns)
    fig = heatmap(
        top10_pct_display,
        title="Predicted Risk by Operational Profile (Top 10 Categories)",
        height=max(500, len(top10_pct_display) * 35 + 100),
        value_format="pct",
        colorscale=HEATMAP_SCALE,
        colorbar_title="Probability",
    )
    # Override default 45° x-axis to horizontal with wrapped text
    fig.update_xaxes(tickangle=0)

    # Find the category with the largest pricing gap
    max_gap_cat = (top10.max(axis=1) - top10.min(axis=1)).idxmax()
    max_gap_val = (top10.max(axis=1) - top10.min(axis=1)).max() * 100
    max_profile = top10.loc[max_gap_cat].idxmax()
    max_prob = top10.loc[max_gap_cat].max() * 100
    min_profile = top10.loc[max_gap_cat].idxmin()
    min_prob = top10.loc[max_gap_cat].min() * 100

    # Custom profile insight
    custom_col = custom_label
    custom_top_cat = top10[custom_col].idxmax() if custom_col in top10.columns else None
    custom_top_prob = (
        top10[custom_col].max() * 100 if custom_col in top10.columns else 0
    )

    insight_parts = (
        f"The largest pricing gap is in <b>{max_gap_cat}</b>: "
        f"<b>{max_profile}</b> shows a {max_prob:.0f}% probability vs "
        f"{min_prob:.0f}% for <b>{min_profile}</b> -- "
        f"a {max_gap_val:.0f} percentage point differential that directly justifies "
        f"segment-specific pricing."
    )

    if custom_top_cat:
        insight_parts += (
            f" Your custom profile's highest risk is "
            f"<b>{custom_top_cat}</b> at {custom_top_prob:.0f}%."
        )

    chart_with_insight(fig, insight_parts, chart_key="uw_bayesian_heatmap")

    # Acronym reference for category codes
    shown_codes = top10.index.tolist()
    abbr_ref = " . ".join(
        f"**{code}** = {ABBREVIATIONS.get(code, code)}"
        for code in shown_codes
        if code in ABBREVIATIONS
    )
    with st.expander("Category Acronym Reference"):
        st.markdown(abbr_ref, unsafe_allow_html=True)

    st.markdown(
        '<div class="coverage-note">'
        "Probabilities are generated by a Binary Relevance Naive Bayes model "
        "trained on 431 accident reports with 5 features (ECE = 0.021). "
        "See Methodology for calibration details."
        "</div>",
        unsafe_allow_html=True,
    )


def _render_methodology(summary: dict, n_accidents: int):
    """Section 10: Methodology and data notes."""

    avg_cats = summary["avg_categories_per_report"]

    methodology_section(f"""
**Data Source:** {summary['total_reports']} National Transportation Safety Board (NTSB)
aviation accident and incident reports (1966--present), of which {n_accidents} are
classified as accident reports and form the analysis population.

**Taxonomy:** Reports are classified using the CAST/ICAO Common Taxonomy Team (CICTT)
occurrence categories. Each report may be assigned multiple categories (multi-label
classification), reflecting the complex, multi-factor nature of aviation accidents.
The taxonomy includes 27 Level 1 categories and 32 Level 2 subcategories. Average of
{avg_cats:.1f} categories per report.

**Features:** Five operational features are extracted from each report:
- **Aircraft category** (single-piston, multi-piston, turboprop, helicopter, etc.)
- **Season** (Spring, Summer, Fall, Winter)
- **Region** (Northeast, South, Midwest, West -- US Census regions)
- **Weather conditions** (VMC or IMC) -- available for {summary['weather_coverage_pct']:.0f}% of reports
- **Time of day** (Morning, Afternoon, Evening, Night) -- available for {summary['time_coverage_pct']:.0f}% of reports

**Severity proxy:** Multi-label complexity (number of CICTT categories per report)
is used as a proxy for loss severity. While the dataset does not contain direct cost
or fatality data, research shows that accidents with more contributing factors tend
to be more severe.

**Risk multipliers:** IMC risk multipliers are computed as the ratio of category
prevalence in IMC conditions to prevalence in VMC conditions. A multiplier of 2.0x
means the category appears in twice the share of IMC accidents vs VMC accidents.

**Pricing factors:** All risk differentials shown in this report are descriptive
analytics based on historical accident data. They quantify relative risk between
segments but should be combined with exposure data, claims history, and actuarial
judgment for actual premium calculations.

**Bayesian Model:** Binary Relevance Naive Bayes classifier trained on {n_accidents}
accident reports. Each of the 27 categories has an independent binary classifier.
The model achieves an Expected Calibration Error (ECE) of 0.021, indicating
well-calibrated probability estimates. Laplace smoothing is applied to handle
unseen feature combinations.

**Limitations:**
- Sample size of {n_accidents} reports limits statistical power for rare categories and
  multi-way cross-tabulations.
- Weather and time-of-day features have incomplete coverage; analyses using these features
  are based on the available subset.
- The Naive Bayes assumption of feature independence is a simplification -- in practice,
  weather and time of day are correlated (e.g., IMC is more common at night).
- Historical data may not reflect current fleet composition or operational practices.
- Co-occurrence counts reflect shared categorization, not proven causal chains.
""")
