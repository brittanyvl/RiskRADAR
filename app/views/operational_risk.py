"""
Operational Risk Report — Chief Pilot / Safety Officer briefing.

Answers: "What should I brief my pilots about?"

Sections:
1. KPI summary (LOC-I rate, CFIT rate, IMC+Night count, top hazard)
2. LOC-I deep dive (subcategory breakdown + HFACS in expander)
3. CFIT deep dive (same pattern)
4. Weather x Time risk matrix (heatmap)
5. Seasonal patterns (line chart, 4 series)
6. Aircraft type risk signatures (selectbox + dynamic chart)
7. Critical conditions analysis (IMC+Night risk ratios)
8. Methodology (expander)
"""

import streamlit as st
import pandas as pd
import numpy as np

from app.components import data_loader as dl
from app.components.charts import horizontal_bar, heatmap, line_chart
from app.components.report_layout import (
    page_header, kpi_row, section_divider, insight, coverage_note,
    methodology_section, chart_with_insight, abbr,
)
from app.components.theme import (
    STEEL, CORAL, AMBER, TEAL, HEATMAP_SCALE,
    TIME_COLORS, TIME_WINDOWS,
)


def render():
    """Render the full Operational Risk Report."""

    # ── Load all data up front ────────────────────────────────────────────
    summary = dl.dataset_summary()
    cat_df = dl.category_counts()
    loc_i_df = dl.loc_i_breakdown()
    cfit_df = dl.cfit_breakdown()
    wx_time = dl.weather_time_matrix()
    seasonal_df = dl.seasonal_patterns(["LOC-I", "CFIT", "ICE", "UIMC"])
    aircraft_df = dl.aircraft_type_risk_signatures()
    critical_df = dl.critical_phase_categories()

    n_accidents = summary["accident_reports"]

    # ── 1. Page Header + KPIs ─────────────────────────────────────────────
    page_header(
        "Operational Risk Report",
        "Flight safety hazards for chief pilots and safety officers"
    )

    loc_i_row = cat_df[cat_df["category_code"] == "LOC-I"]
    cfit_row = cat_df[cat_df["category_code"] == "CFIT"]
    loc_i_pct = loc_i_row["pct_of_reports"].iloc[0] if len(loc_i_row) else 0
    cfit_pct = cfit_row["pct_of_reports"].iloc[0] if len(cfit_row) else 0

    # IMC + Night count from the weather x time matrix
    imc_night_count = 0
    if "Night" in wx_time.columns and "IMC" in wx_time.index:
        imc_night_count = int(wx_time.loc["IMC", "Night"])

    # Top operational hazard (highest prevalence L1 category)
    top_hazard = cat_df.iloc[0]["category_code"] if len(cat_df) else "N/A"
    top_hazard_pct = cat_df.iloc[0]["pct_of_reports"] if len(cat_df) else 0

    kpi_row([
        {
            "label": "LOC-I Prevalence",
            "value": f"{loc_i_pct:.1f}%",
            "detail": "Loss of Control — In Flight",
        },
        {
            "label": "CFIT Prevalence",
            "value": f"{cfit_pct:.1f}%",
            "detail": "Controlled Flight Into Terrain",
        },
        {
            "label": "IMC + Night Reports",
            "value": f"{imc_night_count}",
            "detail": "Highest-risk operating condition",
        },
        {
            "label": "Top Hazard",
            "value": top_hazard,
            "detail": f"Appears in {top_hazard_pct:.1f}% of accidents",
        },
    ])

    st.markdown("")  # spacer

    insight(
        f"Based on {n_accidents:,} National Transportation Safety Board (NTSB) "
        f"accident reports. Loss of Control — In Flight (LOC-I) and Controlled Flight "
        f"Into Terrain (CFIT) are the two most consequential operational hazards in "
        f"general aviation. This report breaks down the specific failure modes, "
        f"environmental conditions, and aircraft types where these risks concentrate.",
    )

    # ── 2. LOC-I Deep Dive ────────────────────────────────────────────────
    section_divider()
    st.markdown("### Loss of Control — In Flight (LOC-I)")

    _render_hazard_breakdown(
        df=loc_i_df,
        prefix="LOC-I-",
        color=CORAL,
        chart_title="LOC-I Subcategory Breakdown",
        chart_key="loc_i_bar",
        insight_text=(
            "Aerodynamic stalls and upsets dominate Loss of Control — In Flight "
            "(LOC-I) accidents. Upset Prevention and Recovery Training (UPRT) "
            "directly targets the top two failure modes. Safety officers should "
            "verify that recurrent training includes slow-flight maneuvering, "
            "unusual attitude recovery, and spin awareness appropriate to "
            "aircraft type."
        ),
        hfacs_label="LOC-I Contributing Human Factors (HFACS)",
    )

    # ── 3. CFIT Deep Dive ─────────────────────────────────────────────────
    section_divider()
    st.markdown("### Controlled Flight Into Terrain (CFIT)")

    _render_hazard_breakdown(
        df=cfit_df,
        prefix="CFIT-",
        color=STEEL,
        chart_title="CFIT Subcategory Breakdown",
        chart_key="cfit_bar",
        insight_text=(
            "Navigation errors and procedural deviations are the primary "
            "Controlled Flight Into Terrain (CFIT) drivers. Standardized "
            "approach procedures, mandatory use of Terrain Awareness and Warning "
            "Systems (TAWS/GPWS), and strict adherence to minimum descent "
            "altitudes address the majority of these scenarios. Brief crews on "
            "terrain awareness for every flight into unfamiliar airports."
        ),
        hfacs_label="CFIT Contributing Human Factors (HFACS)",
    )

    # ── 4. Weather x Time Risk Matrix ─────────────────────────────────────
    section_divider()
    st.markdown("### Weather and Time-of-Day Risk Matrix")
    # Show time window definitions
    windows_text = " · ".join(f"**{k}** {v}" for k, v in TIME_WINDOWS.items())
    st.caption(windows_text)

    coverage_note(
        "Weather and time-of-day",
        min(summary.get("weather_coverage_pct", 0),
            summary.get("time_coverage_pct", 0)),
        n_accidents,
    )

    if not wx_time.empty:
        fig = heatmap(
            wx_time,
            title="Accident Count by Weather Condition and Time of Day",
            height=300,
            annotation_threshold=0,  # annotate all cells (small matrix)
            colorscale=HEATMAP_SCALE,
        )
        # Find the peak cell for the insight
        max_val = wx_time.max().max()
        max_weather = wx_time.max(axis=1).idxmax()
        max_time = wx_time.max(axis=0).idxmax()

        imc_total = int(wx_time.loc["IMC"].sum()) if "IMC" in wx_time.index else 0
        vmc_total = int(wx_time.loc["VMC"].sum()) if "VMC" in wx_time.index else 0

        # Spell out conditions in the insight
        wx_full = {"VMC": "Visual Meteorological Conditions (VMC)",
                    "IMC": "Instrument Meteorological Conditions (IMC)"}

        insight_parts = []
        insight_parts.append(
            f"The highest concentration of accidents occurs during "
            f"<b>{wx_full.get(max_weather, max_weather)}</b> in the "
            f"<b>{max_time}</b> ({int(max_val)} reports)."
        )
        if imc_night_count > 0:
            insight_parts.append(
                f"IMC at night accounts for {imc_night_count} accidents "
                f"despite being the least common operating environment. "
                f"This combination warrants special attention in crew briefings."
            )
        insight_parts.append(
            f"VMC accidents ({vmc_total} total) outnumber IMC ({imc_total}), "
            f"reinforcing that good weather does not eliminate risk — "
            f"complacency in VMC is itself a hazard."
        )

        chart_with_insight(
            fig, " ".join(insight_parts), insight_type="warning",
            chart_key="wx_time_heatmap",
        )
    else:
        st.info("Weather and time-of-day data is not available.")

    # ── 5. Seasonal Patterns ──────────────────────────────────────────────
    section_divider()
    st.markdown("### Seasonal Hazard Patterns")

    if not seasonal_df.empty:
        # Ensure consistent season ordering: Spring → Summer → Fall → Winter
        season_order = ["Spring", "Summer", "Fall", "Winter"]
        seasonal_df["season"] = pd.Categorical(
            seasonal_df["season"], categories=season_order, ordered=True
        )
        seasonal_df = seasonal_df.sort_values(["season", "category_code"])

        fig = line_chart(
            seasonal_df,
            x="season",
            y="prevalence_pct",
            color="category_code",
            title="Hazard Prevalence by Season",
            height=380,
            y_label="Prevalence (%)",
        )

        # Build seasonal insight
        ice_winter = seasonal_df[
            (seasonal_df["category_code"] == "ICE")
            & (seasonal_df["season"] == "Winter")
        ]
        ice_pct = ice_winter["prevalence_pct"].iloc[0] if len(ice_winter) else 0

        uimc_winter = seasonal_df[
            (seasonal_df["category_code"] == "UIMC")
            & (seasonal_df["season"] == "Winter")
        ]
        uimc_pct = uimc_winter["prevalence_pct"].iloc[0] if len(uimc_winter) else 0

        seasonal_insight = (
            f"Icing (ICE) risk peaks in winter at {ice_pct:.1f}% prevalence. "
            f"Unintended Flight in Instrument Meteorological Conditions (UIMC) "
            f"also rises in winter ({uimc_pct:.1f}%), when lower ceilings and "
            f"reduced visibility are more common. Winter operations briefings "
            f"should emphasize known-ice limitations, pilot weather report "
            f"(PIREP) checking, and go/no-go decision frameworks."
        )

        chart_with_insight(fig, seasonal_insight, chart_key="seasonal_line")
    else:
        st.info("Seasonal pattern data is not available.")

    # ── 6. Aircraft Type Risk Signatures ──────────────────────────────────
    section_divider()
    st.markdown("### Risk Signatures by Aircraft Type")

    st.markdown(
        "Different aircraft types have distinct risk profiles. "
        "Select an aircraft type to see its top five hazard categories "
        "and where training emphasis should differ."
    )

    if not aircraft_df.empty:
        ac_types = sorted(aircraft_df["aircraft_category"].unique())

        selected_ac = st.selectbox(
            "Select aircraft type",
            ac_types,
            index=0,
            key="ops_ac_type_select",
        )

        subset = aircraft_df[aircraft_df["aircraft_category"] == selected_ac].copy()
        if not subset.empty:
            total_in_type = subset["report_count"].sum()
            top_cat = subset.iloc[0]

            # Build display label for bar chart
            subset["display_label"] = (
                subset["category_code"] + " — " + subset["category_name"]
            )
            fig = horizontal_bar(
                subset,
                x="prevalence_pct",
                y="display_label",
                title=f"Top Hazards: {selected_ac}",
                color=STEEL,
                height=max(250, len(subset) * 40 + 80),
                show_values=True,
            )
            # Override value text to show percentage
            fig.update_traces(
                text=subset.sort_values("prevalence_pct", ascending=True)[
                    "prevalence_pct"
                ].apply(lambda v: f"{v:.1f}%"),
                textposition="outside",
            )
            fig.update_xaxes(title_text="Prevalence (%)")

            chart_with_insight(
                fig,
                f"For <b>{selected_ac}</b> aircraft, the dominant hazard is "
                f"<b>{top_cat['category_name']}</b> ({top_cat['category_code']}) "
                f"at {top_cat['prevalence_pct']:.1f}% prevalence. "
                f"Training programs for this fleet type should prioritize "
                f"scenarios involving {top_cat['category_name'].lower()}.",
                chart_key="ops_ac_type_chart",
            )
    else:
        st.info("Aircraft type risk data is not available.")

    # ── 7. Critical Conditions Analysis ───────────────────────────────────
    section_divider()
    st.markdown("### Critical Conditions: IMC at Night")

    st.markdown(
        "A **risk ratio** compares how often a hazard occurs during Instrument "
        "Meteorological Conditions (IMC) night operations versus how often IMC "
        "night operations occur overall. A ratio above 1.0 means the hazard is "
        "**more concentrated** in IMC night conditions than expected."
    )

    if not critical_df.empty:
        # Filter to categories with meaningful IMC+Night sample (>= 3)
        crit_filtered = critical_df[critical_df["imc_night_count"] >= 3].copy()

        if not crit_filtered.empty:
            crit_filtered = crit_filtered.sort_values(
                "risk_ratio", ascending=False
            ).head(10)

            fig = horizontal_bar(
                crit_filtered,
                x="risk_ratio",
                y="category_code",
                title="Risk Ratio: IMC Night Concentration by Hazard Category",
                color=CORAL,
                height=max(280, len(crit_filtered) * 35 + 80),
                show_values=True,
            )
            fig.update_traces(
                text=crit_filtered.sort_values("risk_ratio", ascending=True)[
                    "risk_ratio"
                ].apply(lambda v: f"{v:.1f}x"),
                textposition="outside",
            )
            fig.update_xaxes(title_text="Risk Ratio vs. Baseline")

            # Add reference line at 1.0
            fig.add_vline(
                x=1.0, line_dash="dash", line_color="#adb5bd",
                annotation_text="Baseline (1.0x)",
                annotation_position="top",
                annotation_font_size=10,
                annotation_font_color="#6c757d",
            )

            top_crit = crit_filtered.iloc[0]
            chart_with_insight(
                fig,
                f"<b>{top_crit['category_code']}</b> has the highest IMC night "
                f"concentration at {top_crit['risk_ratio']:.1f}x the baseline rate, "
                f"based on {int(top_crit['imc_night_count'])} IMC night reports "
                f"out of {int(top_crit['total_count'])} total. "
                f"Categories above the 1.0x baseline line are "
                f"disproportionately represented in IMC night accidents. "
                f"Night Instrument Flight Rules (IFR) operations in these hazard "
                f"areas deserve dedicated crew briefing items.",
                insight_type="critical",
                chart_key="critical_bar",
            )
        else:
            insight(
                "No hazard categories have three or more IMC night reports, "
                "so risk ratio analysis is not shown. This may reflect "
                "limited data coverage rather than low risk.",
                type="warning",
            )
    else:
        st.info("Critical conditions data is not available.")

    # ── 8. Methodology ────────────────────────────────────────────────────
    section_divider()
    methodology_section(
        f"""
**Data source:** {summary.get('total_reports', 510)} National Transportation Safety Board
(NTSB) aviation reports, of which {n_accidents} are classified as accident reports.
Reports span {summary.get('date_range', 'multiple decades')}.

**Taxonomy:** Each accident is tagged with one or more of 27 CAST/ICAO Common Taxonomy
Team (CICTT) Level 1 occurrence categories, plus 32 Level 2 subcategories for high-priority
hazards. A single accident may carry multiple tags (multi-label classification).

**Prevalence:** Reported as the percentage of accident reports tagged with a given
category. Because reports can have multiple tags, prevalence figures sum to more
than 100%.

**Weather and time coverage:**
- Weather condition (VMC/IMC) is available for {summary.get('weather_coverage_pct', 0):.0f}%
  of accident reports.
- Time of day is available for {summary.get('time_coverage_pct', 0):.0f}% of accident reports.
- Analyses involving these features are based on the subset with known values.

**Time-of-day buckets:** Morning (05:00–10:59), Afternoon (11:00–16:59),
Evening (17:00–20:59), Night (21:00–04:59), all in local time.

**Risk ratio:** For the critical conditions analysis, the risk ratio is calculated as:

*Risk Ratio = (IMC Night rate for category) / (Overall IMC Night rate)*

A ratio of 2.0x means the category appears in IMC night conditions at twice the rate
you would expect from the overall IMC night frequency. Only categories with 3 or more
IMC night reports are included to avoid unstable ratios from small samples.

**Limitations:**
- Survivorship bias: only investigated accidents are included; incidents and
  unreported events are not captured.
- Feature coverage gaps mean some analyses are based on subsets of the data.
- Multi-label tagging means a single accident can appear in multiple category counts.
- Risk ratios are observational and do not imply causation.
"""
    )


# ── Helpers ───────────────────────────────────────────────────────────────


def _render_hazard_breakdown(
    df: pd.DataFrame,
    prefix: str,
    color: str,
    chart_title: str,
    chart_key: str,
    insight_text: str,
    hfacs_label: str,
):
    """
    Render a hazard subcategory breakdown: specific subcats as horizontal bar,
    HFACS codes (HF-*) tucked into an expander.
    """
    if df.empty:
        st.info("No subcategory data available.")
        return

    # Split into specific subcategories and HFACS human factors
    specific = df[df["category_code"].str.startswith(prefix)].copy()
    hfacs = df[df["category_code"].str.startswith("HF-")].copy()

    if not specific.empty:
        fig = horizontal_bar(
            specific,
            x="report_count",
            y="category_name",
            title=chart_title,
            color=color,
            height=max(250, len(specific) * 40 + 80),
        )
        chart_with_insight(fig, insight_text, chart_key=chart_key)
    else:
        insight(insight_text)

    if not hfacs.empty:
        with st.expander(hfacs_label):
            st.markdown(
                "Human factors codes from the Human Factors Analysis and "
                "Classification System (HFACS) framework that appear alongside "
                "this hazard category. These represent the crew-level "
                "contributing factors behind the accidents."
            )
            fig_hf = horizontal_bar(
                hfacs,
                x="report_count",
                y="category_name",
                color=AMBER,
                height=max(220, len(hfacs) * 35 + 60),
            )
            st.plotly_chart(fig_hf, use_container_width=True,
                            key=f"{chart_key}_hfacs")
