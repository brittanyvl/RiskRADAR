"""
Operational Risk Report — Chief Pilot / Safety Officer briefing.

Persona: Chief Pilot and Director of Safety at a Part 121/135 operator.
Core question: "What should I brief my pilots about, and where should our
SMS focus?"

Narrative arc (3 acts, 11 sections + methodology):
  Act 1 — The Landscape (S1-S3): Top hazards overview, LOC-I deep dive,
      CFIT deep dive.
  Act 2 — The Conditions (S4-S7): Weather x Time matrix, seasonal patterns,
      regional risk, IMC at Night critical conditions.
  Act 3 — The Levers (S8-S11): Human factors training priorities,
      fleet-specific risk signatures, decade trends, co-occurrence cascading
      failure chains.
"""

import streamlit as st
import pandas as pd
import numpy as np

from app.components import data_loader as dl
from app.components.charts import (
    horizontal_bar, heatmap, line_chart, grouped_bar,
)
from app.components.report_layout import (
    page_header, kpi_row, section_divider, insight, coverage_note,
    sample_note, methodology_section, chart_with_insight, ABBREVIATIONS,
    abbr,
)
from app.components.theme import (
    STEEL, CORAL, AMBER, TEAL, NAVY,
    HEATMAP_SCALE, SEQUENTIAL_SCALE, TIME_WINDOWS,
)


def render():
    """Render the full Operational Risk Report."""

    # ── Load all data up front ────────────────────────────────────────────
    summary = dl.dataset_summary()
    cat_df = dl.category_counts()
    loc_i_df = dl.loc_i_breakdown()
    cfit_df = dl.cfit_breakdown()
    wx_time = dl.weather_time_matrix()
    seasonal_df = dl.seasonal_patterns(["LOC-I", "CFIT", "ICE", "UIMC", "FUEL"])
    region_matrix = dl.region_category_matrix(top_n_categories=5)
    critical_df = dl.critical_phase_categories()
    hf_by_cat = dl.human_factors_by_category()
    sigs = dl.aircraft_type_risk_signatures()
    decade_trends = dl.category_prevalence_by_decade(
        ["LOC-I", "CFIT", "UIMC", "ICE", "SCF-PP"]
    )
    coocc = dl.cooccurrence_matrix()
    night_data = dl.night_accident_share()
    hf_totals = dl.human_factors_totals()

    n_accidents = summary["accident_reports"]
    avg_cats = summary["avg_categories_per_report"]

    # ══════════════════════════════════════════════════════════════════════
    # PAGE HEADER + KPIs
    # ══════════════════════════════════════════════════════════════════════

    page_header(
        "Operational Risk Report",
        "Flight safety hazard analysis for chief pilots and safety officers. "
        "Identifies the highest-priority risks, operating conditions that "
        "concentrate danger, and actionable training targets — based on "
        "NTSB accident investigation findings.",
    )

    # ── KPI Row 1 (4 KPIs): Headline Risk Indicators ──
    loc_i_row = cat_df[cat_df["category_code"] == "LOC-I"]
    cfit_row = cat_df[cat_df["category_code"] == "CFIT"]
    loc_i_pct = loc_i_row["pct_of_reports"].iloc[0] if len(loc_i_row) else 0
    cfit_pct = cfit_row["pct_of_reports"].iloc[0] if len(cfit_row) else 0

    night_pct = night_data["night_pct"]

    kpi_row([
        {
            "label": "LOC-I Rate",
            "value": f"{loc_i_pct:.0f}%",
            "detail": "Loss of Control — In Flight — leading cause of fatal GA accidents",
            "accent": CORAL,
        },
        {
            "label": "CFIT Rate",
            "value": f"{cfit_pct:.0f}%",
            "detail": "Controlled Flight Into Terrain — most preventable through procedures",
            "accent": STEEL,
        },
        {
            "label": "IMC Accident Share",
            "value": f"{summary['imc_pct']:.0f}%",
            "detail": "Of weather-classified accidents occurred in Instrument Meteorological Conditions",
            "accent": STEEL,
        },
        {
            "label": "Night Accident Rate",
            "value": f"{night_pct:.0f}%",
            "detail": f"{night_data['night_count']} of {night_data['total_with_time']} "
                      f"accidents with known time occurred at night",
            "accent": NAVY,
        },
    ])

    # ── KPI Row 2 (2 KPIs): Operational Depth Indicators ──
    _hf_display = {
        "HF-VIOLATION": "Procedural Violation",
        "HF-PERCEPTUAL": "Perceptual Error",
        "HF-DECISION": "Decision Error",
        "HF-SKILL": "Skill-Based Error",
        "HF-CONDITION": "Adverse Condition",
    }

    # Use hf_totals (distinct counts) for the KPI
    if not hf_totals.empty:
        top_hf = hf_totals.iloc[0]
        top_hf_name = _hf_display.get(top_hf["category_code"], top_hf["category_code"])
        top_hf_count = int(top_hf["report_count"])
    else:
        top_hf_name = "N/A"
        top_hf_count = 0

    kpi_row([
        {
            "label": "Avg Contributing Factors",
            "value": f"{avg_cats:.1f}",
            "detail": "CICTT categories per accident — higher means more complex chains",
            "accent": TEAL,
        },
        {
            "label": "Top Human Factor",
            "value": top_hf_name,
            "detail": f"{top_hf_count} reports — the #1 crew-level training priority",
            "accent": NAVY,
        },
    ])

    sample_note(
        n_accidents,
        "National Transportation Safety Board (NTSB) accident reports "
        "classified with CAST/ICAO Common Taxonomy Team (CICTT) taxonomy",
    )

    # ══════════════════════════════════════════════════════════════════════
    # ACT 1 — THE LANDSCAPE
    # ══════════════════════════════════════════════════════════════════════

    # ── Section 1: Top 10 Operational Hazards ─────────────────────────────
    section_divider()
    st.markdown("### Top 10 Operational Hazards")
    st.markdown(
        "A ranked overview of the most prevalent risk categories across all "
        "accident reports. This is the risk landscape that shapes every "
        "subsequent section of this report. Prevalence is the percentage of "
        "accident reports tagged with each category — because a single accident "
        "can involve multiple categories, figures sum to more than 100%."
    )

    top10 = cat_df.head(10).copy()
    top10["category_label"] = top10["category_code"].map(
        lambda c: ABBREVIATIONS.get(c, c)
    )

    fig_top10 = horizontal_bar(
        top10,
        x="pct_of_reports",
        y="category_label",
        title="Top 10 Hazard Categories by Prevalence",
        color=STEEL,
        height=max(300, len(top10) * 36 + 80),
        show_values=True,
        value_format="1f",
    )
    fig_top10.update_traces(
        text=top10.sort_values("pct_of_reports", ascending=True)[
            "pct_of_reports"
        ].apply(lambda v: f"{v:.1f}%"),
        textposition="outside",
    )
    fig_top10.update_xaxes(title_text="Prevalence (%)")

    top3 = top10.head(3)
    top3_text = ", ".join(
        f"<b>{abbr(row['category_code'])}</b>"
        for _, row in top3.iterrows()
    )
    top3_pcts = top3["pct_of_reports"].sum()

    chart_with_insight(
        fig_top10,
        f"The top three operational hazards — {top3_text} — collectively "
        f"appear in a combined {top3_pcts:.0f}% of accident reports (with overlap). "
        f"Because a single accident averages {avg_cats:.1f} contributing categories, "
        f"these figures reflect overlapping risk chains rather than independent events. "
        f"Safety officers should treat these as interconnected priorities, not a "
        f"pick-one checklist.",
        chart_key="ops_top10_bar",
    )

    # ── Section 2: LOC-I Deep Dive ────────────────────────────────────────
    section_divider()
    st.markdown(
        "### LOC-I (Loss of Control — In Flight): Failure Mode Analysis"
    )
    st.markdown(
        f"{abbr('LOC-I')} is the leading cause of fatal "
        "accidents in general aviation. This section breaks down the specific "
        f"failure modes within {abbr('LOC-I')} to identify where "
        f"{abbr('UPRT')} should focus.",
        unsafe_allow_html=True,
    )

    _render_hazard_breakdown(
        df=loc_i_df,
        prefix="LOC-I-",
        color=CORAL,
        chart_title="LOC-I Subcategory Breakdown",
        chart_key="ops_loci_bar",
        hfacs_label="LOC-I Contributing Human Factors — Training Implications",
    )

    # ── Section 3: CFIT Deep Dive ─────────────────────────────────────────
    section_divider()
    st.markdown(
        "### CFIT (Controlled Flight Into Terrain): Failure Mode Analysis"
    )
    st.markdown(
        f"{abbr('CFIT')} is the second most consequential "
        "operational hazard and among the most preventable through procedural "
        f"compliance, terrain awareness, and proper use of {abbr('TAWS')}/{abbr('GPWS')}. "
        "This breakdown reveals the specific failure modes that training and "
        f"{abbr('SOPs')} must address.",
        unsafe_allow_html=True,
    )

    _render_hazard_breakdown(
        df=cfit_df,
        prefix="CFIT-",
        color=STEEL,
        chart_title="CFIT Subcategory Breakdown",
        chart_key="ops_cfit_bar",
        hfacs_label="CFIT Contributing Human Factors — Training Implications",
    )

    # ══════════════════════════════════════════════════════════════════════
    # ACT 2 — THE CONDITIONS
    # ══════════════════════════════════════════════════════════════════════

    # ── Section 4: Weather x Time Risk Matrix ─────────────────────────────
    section_divider()
    st.markdown("### Operating Environment Risk Matrix")
    st.markdown(
        "When and in what weather do accidents concentrate? This matrix maps "
        f"accident counts by weather condition — {abbr('VMC')} versus "
        f"{abbr('IMC')} — against time of day. The result is a go/no-go "
        "decision support tool.",
        unsafe_allow_html=True,
    )

    windows_text = " · ".join(f"**{k}** {v}" for k, v in TIME_WINDOWS.items())
    st.caption(windows_text)

    coverage_note(
        "Weather and time-of-day",
        min(summary.get("weather_coverage_pct", 0),
            summary.get("time_coverage_pct", 0)),
        n_accidents,
    )

    if not wx_time.empty:
        fig_wx = heatmap(
            wx_time,
            title="Accident Count by Weather Condition and Time of Day",
            height=300,
            annotation_threshold=0,
            colorscale=HEATMAP_SCALE,
        )
        fig_wx.update_xaxes(tickangle=0)

        max_val = wx_time.max().max()
        max_weather = wx_time.max(axis=1).idxmax()
        max_time = wx_time.max(axis=0).idxmax()
        imc_night_count = (
            int(wx_time.loc["IMC", "Night"])
            if "Night" in wx_time.columns and "IMC" in wx_time.index
            else 0
        )
        imc_total = (
            int(wx_time.loc["IMC"].sum()) if "IMC" in wx_time.index else 0
        )
        vmc_total = (
            int(wx_time.loc["VMC"].sum()) if "VMC" in wx_time.index else 0
        )

        wx_full = {
            "VMC": "Visual Meteorological Conditions (VMC)",
            "IMC": "Instrument Meteorological Conditions (IMC)",
        }
        parts = [
            f"The highest concentration of accidents occurs during "
            f"<b>{wx_full.get(max_weather, max_weather)}</b> in the "
            f"<b>{max_time}</b> ({int(max_val)} reports).",
        ]
        if imc_night_count > 0:
            parts.append(
                f"{abbr('IMC')} at night accounts "
                f"for {imc_night_count} accidents despite being the least common "
                f"operating environment — this combination warrants special "
                f"attention in crew briefings and dispatch decisions."
            )
        parts.append(
            f"{abbr('VMC')} accidents ({vmc_total} total) outnumber "
            f"{abbr('IMC')} ({imc_total}), "
            f"reinforcing that clear weather does not eliminate risk — "
            f"complacency in {abbr('VMC')} is itself a hazard. Brief crews on "
            f"{abbr('VMC')} risk factors (maneuvering flight, low-altitude "
            f"operations) as vigorously as {abbr('IMC')} threats."
        )

        chart_with_insight(
            fig_wx, " ".join(parts),
            insight_type="warning",
            chart_key="ops_wx_heatmap",
        )
    else:
        st.info("Weather and time-of-day data is not available.")

    # ── Section 5: Seasonal Hazard Patterns ───────────────────────────────
    section_divider()
    st.markdown("### Seasonal Risk Patterns")
    st.markdown(
        "Hazard prevalence shifts with the seasons. Understanding these "
        "rhythms allows chief pilots to adjust seasonal briefing topics and "
        f"safety officers to plan quarterly {abbr('SMS')} "
        "focus areas proactively rather than reactively.",
        unsafe_allow_html=True,
    )

    if not seasonal_df.empty:
        season_order = ["Spring", "Summer", "Fall", "Winter"]
        seasonal_df["season"] = pd.Categorical(
            seasonal_df["season"], categories=season_order, ordered=True
        )
        seasonal_df = seasonal_df.sort_values(["season", "category_code"])

        # Map codes to readable legend labels
        seasonal_df["category_label"] = seasonal_df["category_code"].map(
            lambda c: f"{c} ({ABBREVIATIONS.get(c, c)})"
        )

        fig_seasonal = grouped_bar(
            seasonal_df,
            x="season",
            y="prevalence_pct",
            group="category_label",
            title="Hazard Prevalence by Season",
            height=420,
        )
        fig_seasonal.update_yaxes(title_text="Prevalence (%)")
        fig_seasonal.update_xaxes(tickangle=0)

        # Build dynamic insight from the data
        ice_winter = seasonal_df[
            (seasonal_df["category_code"] == "ICE")
            & (seasonal_df["season"] == "Winter")
        ]
        ice_pct = ice_winter["prevalence_pct"].iloc[0] if len(ice_winter) else 0

        uimc_winter = seasonal_df[
            (seasonal_df["category_code"] == "UIMC")
            & (seasonal_df["season"] == "Winter")
        ]
        uimc_pct = (
            uimc_winter["prevalence_pct"].iloc[0] if len(uimc_winter) else 0
        )

        fuel_summer = seasonal_df[
            (seasonal_df["category_code"] == "FUEL")
            & (seasonal_df["season"] == "Summer")
        ]
        fuel_pct = (
            fuel_summer["prevalence_pct"].iloc[0] if len(fuel_summer) else 0
        )

        seasonal_insight = (
            f"<b>{abbr('ICE')}</b> risk peaks in winter at {ice_pct:.1f}% prevalence. "
            f"<b>{abbr('UIMC')}</b> also rises in winter "
            f"({uimc_pct:.1f}%), when lower ceilings and reduced visibility "
            f"catch {abbr('VFR')} pilots off guard. "
        )
        if fuel_pct > 0:
            seasonal_insight += (
                f"<b>Fuel Related</b> events reach {fuel_pct:.1f}% in "
                f"summer, likely driven by longer cross-country {abbr('VFR')} flights. "
            )
        seasonal_insight += (
            "Adjust seasonal briefings accordingly: winter operations should "
            f"emphasize known-ice limitations, {abbr('PIREP')} "
            "checking, and go/no-go decision frameworks. Summer briefings "
            "should reinforce fuel planning discipline."
        )

        chart_with_insight(
            fig_seasonal, seasonal_insight, chart_key="ops_seasonal_line"
        )
    else:
        st.info("Seasonal pattern data is not available.")

    # ── Section 6: Regional Risk Concentrations ───────────────────────────
    section_divider()
    st.markdown("### Geographic Risk Distribution")
    st.markdown(
        "Accident risks are not distributed uniformly across geography. "
        "Regional patterns should inform route-specific crew briefings, "
        f"base-specific training emphasis, and {abbr('SMS')} "
        "resource allocation. Values show report counts for each region and "
        "hazard category combination.",
        unsafe_allow_html=True,
    )

    if not region_matrix.empty:
        fig_region = heatmap(
            region_matrix,
            title="Region × Top Hazard Categories (Report Counts)",
            height=max(300, len(region_matrix) * 45 + 80),
            colorscale=SEQUENTIAL_SCALE,
            colorbar_title="Reports",
            hover_labels=ABBREVIATIONS,
            annotation_threshold=1,
        )

        # Find the peak cell
        max_region = region_matrix.max(axis=1).idxmax()
        max_cat_for_region = region_matrix.loc[max_region].idxmax()
        max_region_val = int(region_matrix.loc[max_region, max_cat_for_region])
        max_cat_full = ABBREVIATIONS.get(max_cat_for_region, max_cat_for_region)

        chart_with_insight(
            fig_region,
            f"<b>{max_region}</b> has the highest concentration of "
            f"<b>{max_cat_full}</b> ({abbr(max_cat_for_region)}) events "
            f"({max_region_val} reports). Regional risk patterns should inform "
            f"route-specific crew briefings — pilots operating in high-"
            f"concentration regions should receive targeted scenario training "
            f"for the dominant hazards in their area of operations.",
            chart_key="ops_region_heatmap",
        )
    else:
        st.info("Regional risk data is not available.")

    # ── Section 7: Critical Conditions — IMC at Night ─────────────────────
    section_divider()
    st.markdown(
        "### Highest-Risk Operating Condition: IMC (Instrument Meteorological Conditions) at Night"
    )
    st.markdown(
        f"A **risk ratio** compares how often a hazard occurs during "
        f"{abbr('IMC')} night operations versus "
        f"how often {abbr('IMC')} night operations occur overall. A ratio above 1.0 "
        f"means the hazard is **more concentrated** in {abbr('IMC')} night conditions "
        f"than expected. This analysis identifies which hazard categories "
        f"are disproportionately dangerous when flying {abbr('IFR')} at night.",
        unsafe_allow_html=True,
    )

    if not critical_df.empty:
        crit_filtered = critical_df[critical_df["imc_night_count"] >= 3].copy()

        if not crit_filtered.empty:
            crit_filtered = crit_filtered.sort_values(
                "risk_ratio", ascending=False
            ).head(10)

            crit_filtered["category_label"] = crit_filtered[
                "category_code"
            ].map(lambda c: ABBREVIATIONS.get(c, c))

            fig_crit = horizontal_bar(
                crit_filtered,
                x="risk_ratio",
                y="category_label",
                title="Risk Ratio: IMC Night Concentration by Hazard",
                color=CORAL,
                height=max(280, len(crit_filtered) * 35 + 80),
                show_values=True,
                value_format="ratio",
            )
            fig_crit.update_xaxes(title_text="Risk Ratio vs. Baseline")
            fig_crit.add_vline(
                x=1.0,
                line_dash="dash",
                line_color="#adb5bd",
                annotation_text="Baseline (1.0x)",
                annotation_position="top",
                annotation_font_size=10,
                annotation_font_color="#6c757d",
            )

            top_crit = crit_filtered.iloc[0]
            chart_with_insight(
                fig_crit,
                f"<b>{top_crit['category_label']}</b> "
                f"({abbr(top_crit['category_code'])}) has the highest {abbr('IMC')} night "
                f"concentration at {top_crit['risk_ratio']:.1f}x the baseline "
                f"rate, based on {int(top_crit['imc_night_count'])} {abbr('IMC')} night "
                f"reports out of {int(top_crit['total_count'])} total. "
                f"Categories above the 1.0x baseline are disproportionately "
                f"represented in {abbr('IMC')} night accidents. Night {abbr('IFR')} "
                f"operations in these hazard areas deserve "
                f"dedicated crew briefing items, minimum experience "
                f"requirements, and dispatch-level risk gates.",
                insight_type="critical",
                chart_key="ops_critical_bar",
            )
        else:
            insight(
                f"No hazard categories have three or more {abbr('IMC')} night reports, "
                "so risk ratio analysis is not shown. This may reflect "
                "limited data coverage rather than low risk.",
                type="warning",
            )
    else:
        st.info("Critical conditions data is not available.")

    # ══════════════════════════════════════════════════════════════════════
    # ACT 3 — THE LEVERS
    # ══════════════════════════════════════════════════════════════════════

    # ── Section 8: Human Factors Training Priorities ──────────────────────
    section_divider()
    st.markdown(
        "### Human Factors (HFACS) — Training Priorities"
    )
    st.markdown(
        f"Human factors subcategories from the {abbr('HFACS')} framework "
        "reveal <b>why</b> accidents happen, not just what happened. These "
        f"findings directly map to {abbr('CRM')} curriculum, {abbr('UPRT')}, "
        "and procedural compliance programs. Understanding which error types "
        "dominate tells chief pilots exactly where training investment yields "
        "the greatest safety return.",
        unsafe_allow_html=True,
    )

    if not hf_totals.empty:
        hf_bar_display = hf_totals.copy()
        hf_bar_display["display_name"] = hf_bar_display["category_code"].map(
            lambda c: _hf_display.get(c, c)
        )

        fig_hf = horizontal_bar(
            hf_bar_display,
            x="report_count",
            y="display_name",
            title="Human Factors in Accident Reports (Distinct Counts)",
            color=NAVY,
            height=max(250, len(hf_bar_display) * 45 + 60),
            show_values=True,
        )

        top_hf_row = hf_bar_display.iloc[0]
        second_hf = hf_bar_display.iloc[1] if len(hf_bar_display) > 1 else None
        hf_insight = (
            f"<b>{top_hf_row['display_name']}</b> is the most common human "
            f"factor, appearing in {int(top_hf_row['report_count'])} reports."
        )
        if second_hf is not None:
            hf_insight += (
                f" <b>{second_hf['display_name']}</b> follows with "
                f"{int(second_hf['report_count'])} reports."
            )
        hf_insight += (
            " Compliance-focused training targets procedural violations; "
            "situational awareness and spatial orientation training targets "
            "perceptual errors. Verify that recurrent training addresses "
            "both error types with scenario-based exercises, not just "
            "ground school lectures."
        )

        chart_with_insight(fig_hf, hf_insight, chart_key="ops_hf_bar")

        # ── HF × Parent Category Interaction Heatmap ──
        if not hf_by_cat.empty:
            st.markdown("#### How Human Factors Interact with Risk Categories")
            st.markdown(
                "Human factor findings are classified under specific parent "
                "risk categories. This heatmap shows where each human factor "
                "type concentrates, revealing which risk areas are most driven "
                "by which type of human error — and therefore which training "
                "approaches to pair with which hazard categories."
            )

            _hf_short = {
                "HF-VIOLATION": "Procedural<br>Violation",
                "HF-PERCEPTUAL": "Perceptual<br>Error",
                "HF-DECISION": "Decision<br>Error",
                "HF-SKILL": "Skill-Based<br>Error",
            }
            hf_by_cat_display = hf_by_cat.copy()
            hf_by_cat_display["hf_label"] = hf_by_cat_display["hf_code"].map(
                lambda c: _hf_short.get(c, c)
            )
            hf_by_cat_display["parent_label"] = hf_by_cat_display[
                "parent_code"
            ].map(lambda c: ABBREVIATIONS.get(c, c))

            hf_pivot = hf_by_cat_display.pivot_table(
                index="parent_label",
                columns="hf_label",
                values="report_count",
                fill_value=0,
            )

            fig_hf_cat = heatmap(
                hf_pivot,
                title="Human Factors × Risk Category",
                height=max(280, len(hf_pivot) * 50 + 80),
                colorbar_title="Reports",
            )
            fig_hf_cat.update_xaxes(tickangle=0)

            # Build data-driven insight: find dominant HF per parent
            hf_insight_parts = []
            for parent in hf_pivot.index:
                row = hf_pivot.loc[parent]
                if row.max() > 0:
                    top_hf_type = row.idxmax()
                    top_hf_val = int(row.max())
                    hf_insight_parts.append(
                        f"<b>{parent}</b> is most associated with "
                        f"{top_hf_type.replace('<br>', ' ')} ({top_hf_val} reports)"
                    )

            hf_cat_insight = ". ".join(hf_insight_parts) + "." if hf_insight_parts else ""
            hf_cat_insight += (
                " This means <b>compliance-focused training</b> should be "
                "emphasized for categories dominated by procedural violations, "
                "while <b>decision-making training</b> should be the focus for "
                "categories driven by decision errors."
            )

            chart_with_insight(
                fig_hf_cat,
                hf_cat_insight,
                chart_key="ops_hf_category",
            )

            # Build dynamic parent list from the data
            parent_codes = hf_by_cat["parent_code"].unique().tolist()
            parent_names = [
                f"{ABBREVIATIONS.get(c, c)}" for c in parent_codes
            ]
            parent_list_str = ", ".join(parent_names[:-1]) + f", and {parent_names[-1]}" if len(parent_names) > 1 else parent_names[0] if parent_names else ""

            st.markdown(
                '<div class="coverage-note">'
                f"Note: {abbr('HFACS')} subcategories are only classified under "
                f"{parent_list_str} in the {abbr('CICTT')} taxonomy. Other risk "
                "categories do not have human factors Level 2 classifications "
                "in this dataset."
                "</div>",
                unsafe_allow_html=True,
            )
    else:
        st.info("No human factors data available.")

    # ── Section 9: Fleet-Specific Risk Signatures ─────────────────────────
    section_divider()
    st.markdown("### Risk Profile by Aircraft Type")
    st.markdown(
        "Different aircraft types face fundamentally different risk profiles. "
        "This matrix shows the top risk concentrations for each aircraft type, "
        "enabling chief pilots to prioritize training emphasis and safety "
        "briefings for their specific fleet."
    )

    if not sigs.empty:
        # Filter out non-meaningful aircraft categories
        _exclude = {"not-applicable", "other"}
        top3_sigs = sigs[
            (sigs["rank"] <= 3) & (~sigs["aircraft_category"].isin(_exclude))
        ].copy()

        # Wrap long column labels with <br> so they stay horizontal
        def _wrap(name: str, max_line: int = 14) -> str:
            if len(name) <= max_line:
                return name
            name = name.replace(" — ", "<br>").replace(" - ", "<br>")
            if "<br>" in name:
                return name
            words = name.split()
            lines, cur = [], ""
            for w in words:
                if cur and len(cur) + 1 + len(w) > max_line:
                    lines.append(cur)
                    cur = w
                else:
                    cur = f"{cur} {w}" if cur else w
            if cur:
                lines.append(cur)
            return "<br>".join(lines)

        top3_sigs["category_label"] = top3_sigs["category_code"].map(
            lambda c: _wrap(ABBREVIATIONS.get(c, c))
        )

        pivot = top3_sigs.pivot_table(
            index="aircraft_category",
            columns="category_label",
            values="prevalence_pct",
            fill_value=0,
        )

        fig_sigs = heatmap(
            pivot,
            title="Top Risk Categories by Aircraft Type",
            value_format="pct",
            height=max(len(pivot) * 50 + 100, 350),
            colorbar_title="Prevalence %",
        )
        fig_sigs.update_xaxes(tickangle=0)
        fig_sigs.update_layout(margin=dict(b=80))

        chart_with_insight(
            fig_sigs,
            "This matrix shows the top risk concentrations for each aircraft "
            "type. Use it to prioritize training emphasis and safety briefings "
            "for your specific fleet. Differences between aircraft types point "
            "to fleet-specific training needs — brief crews on the hazards most "
            "relevant to the aircraft they fly, not a one-size-fits-all syllabus.",
            chart_key="ops_fleet_heatmap",
        )
    else:
        st.info("Aircraft type risk signature data is not available.")

    # ── Section 10: Decade Trends ─────────────────────────────────────────
    section_divider()
    st.markdown("### How Are Operational Risks Evolving?")
    st.markdown(
        "Tracking hazard prevalence by decade reveals whether key risks are "
        "improving, worsening, or holding steady — essential context for "
        f"safety officers presenting quarterly {abbr('SMS')} "
        "metrics and chief pilots justifying continued investment in training "
        "programs for declining-but-not-eliminated risks.",
        unsafe_allow_html=True,
    )

    if not decade_trends.empty:
        decade_display = decade_trends.copy()
        decade_display["category_label"] = decade_display["category_code"].map(
            lambda c: f"{c} ({ABBREVIATIONS.get(c, c)})"
        )

        fig_trends = line_chart(
            decade_display,
            x="decade",
            y="prevalence_pct",
            color="category_label",
            title="Key Operational Risk Trends by Decade",
            height=450,
            y_label="Prevalence (%)",
        )

        # Identify biggest movers
        decades = sorted(decade_trends["decade"].unique())
        if len(decades) >= 2:
            earliest = decade_trends[decade_trends["decade"] == decades[0]]
            latest = decade_trends[decade_trends["decade"] == decades[-1]]
            merged = earliest[["category_code", "prevalence_pct"]].merge(
                latest[["category_code", "prevalence_pct"]],
                on="category_code",
                suffixes=("_early", "_late"),
            )
            merged["change"] = (
                merged["prevalence_pct_late"] - merged["prevalence_pct_early"]
            )
            biggest_rise = merged.loc[merged["change"].idxmax()]
            biggest_drop = merged.loc[merged["change"].idxmin()]

            rise_name = ABBREVIATIONS.get(
                biggest_rise["category_code"],
                biggest_rise["category_code"],
            )
            drop_name = ABBREVIATIONS.get(
                biggest_drop["category_code"],
                biggest_drop["category_code"],
            )
            trend_insight = (
                f"From the {decades[0]}s to the {decades[-1]}s, "
                f"<b>{rise_name}</b> ({abbr(biggest_rise['category_code'])}) "
                f"saw the largest increase "
                f"(+{biggest_rise['change']:.0f} percentage points), while "
                f"<b>{drop_name}</b> ({abbr(biggest_drop['category_code'])}) "
                f"showed the greatest decline "
                f"({biggest_drop['change']:.0f} percentage points). "
                f"Prevalence is normalized per decade to account for varying "
                f"report volumes. A declining trend does not mean a risk is "
                f"eliminated — it means interventions may be working and "
                f"should be sustained."
            )
        else:
            trend_insight = (
                "Insufficient decades for trend comparison. Additional data "
                "is needed for meaningful temporal analysis."
            )

        chart_with_insight(
            fig_trends, trend_insight, chart_key="ops_decade_trends"
        )
    else:
        st.info("No decade trend data available.")

    # ── Section 11: Co-occurrence Patterns ────────────────────────────────
    section_divider()
    st.markdown("### Risk Factor Co-occurrence: Cascading Failure Chains")
    st.markdown(
        "When two risk categories frequently appear together in the same "
        "accident report, it signals a cascading failure pattern — one hazard "
        "triggers or compounds another. Training scenarios should combine "
        f"co-occurring hazards rather than treating them in isolation. "
        f"{abbr('SMS')} should monitor these pairs as compound risk indicators.",
        unsafe_allow_html=True,
    )

    # Filter to top 10 categories for readability
    top10_codes = cat_df.head(10)["category_code"].tolist()
    coocc_filtered = coocc.loc[
        coocc.index.isin(top10_codes),
        coocc.columns.isin(top10_codes),
    ]
    coocc_filtered = coocc_filtered.reindex(
        index=top10_codes, columns=top10_codes
    )

    fig_coocc = heatmap(
        coocc_filtered,
        title="Category Co-occurrence (Top 10 Hazards)",
        height=560,
        mask_diagonal=True,
        lower_triangle_only=True,
        hover_labels=ABBREVIATIONS,
        colorbar_title="Shared Reports",
    )

    # Find strongest off-diagonal co-occurrence
    coocc_vals = coocc_filtered.values.astype(float).copy()
    np.fill_diagonal(coocc_vals, 0)
    # Mask upper triangle (already masked visually, but zero for argmax)
    coocc_vals[np.triu_indices_from(coocc_vals, k=1)] = 0
    max_idx = np.unravel_index(coocc_vals.argmax(), coocc_vals.shape)
    max_cat1 = coocc_filtered.index[max_idx[0]]
    max_cat2 = coocc_filtered.columns[max_idx[1]]
    max_co_val = int(coocc_vals[max_idx])

    max_cat1_full = ABBREVIATIONS.get(max_cat1, max_cat1)
    max_cat2_full = ABBREVIATIONS.get(max_cat2, max_cat2)

    chart_with_insight(
        fig_coocc,
        f"The strongest co-occurrence is between <b>{abbr(max_cat1)}</b> "
        f"({max_cat1_full}) and <b>{abbr(max_cat2)}</b> ({max_cat2_full}), "
        f"appearing together in {max_co_val} reports. Co-occurring "
        f"categories suggest cascading failure chains — add combined-hazard "
        f"scenarios to recurrent training (e.g., an engine failure leading "
        f"to loss of control, or weather deterioration leading to {abbr('CFIT')}). "
        f"Hover over any cell for full category names.",
        chart_key="ops_coocc_heatmap",
    )

    # Acronym reference expander
    abbr_ref = " · ".join(
        f"**{code}** = {ABBREVIATIONS.get(code, code)}"
        for code in top10_codes
        if code in ABBREVIATIONS
    )
    with st.expander("Category Acronym Reference"):
        st.markdown(abbr_ref, unsafe_allow_html=True)

    # ══════════════════════════════════════════════════════════════════════
    # METHODOLOGY
    # ══════════════════════════════════════════════════════════════════════

    section_divider()
    methodology_section(
        f"""
**Data source:** {summary.get('total_reports', 510)} National Transportation Safety Board
(NTSB) aviation reports, of which {n_accidents} are classified as accident reports.
Reports span from the {summary.get('date_range_start', '1960s')[:4]}s to the
{summary.get('date_range_end', '2020s')[:4]}s.

**Taxonomy:** Each accident is tagged with one or more of 27 CAST/ICAO Common Taxonomy
Team (CICTT) Level 1 occurrence categories, plus 32 Level 2 subcategories for high-priority
hazards. Classification was performed using a fine-tuned language model with human review
of edge cases. A single accident may carry multiple tags (multi-label classification).

**Prevalence:** Reported as the percentage of accident reports tagged with a given
category. Because reports can have multiple tags (average of {avg_cats:.1f} per report),
prevalence figures across categories sum to more than 100%.

**Weather and time coverage:**
- Weather condition (VMC/IMC) is available for {summary.get('weather_coverage_pct', 0):.0f}%
  of weather-classified accident reports.
- Time of day is available for {summary.get('time_coverage_pct', 0):.0f}% of accident
  reports.
- Analyses involving these features are based on the subset with known values.

**Time-of-day buckets:** Morning (05:00–10:59), Afternoon (11:00–16:59),
Evening (17:00–20:59), Night (21:00–04:59), all in local time.

**Night accident rate (KPI):** Calculated as the number of accidents with
time_of_day = 'Night' divided by the total with known time-of-day
({night_data['total_with_time']} reports, {summary.get('time_coverage_pct', 0):.0f}% coverage).

**Risk ratio (IMC Night analysis):** Calculated as:
*Risk Ratio = (IMC Night rate for category) / (Overall IMC Night rate)*

A ratio of 2.0x means the category appears in IMC night conditions at twice the rate
expected from the overall IMC night frequency. Only categories with 3 or more IMC night
reports are included to avoid unstable ratios from small samples.

**Human factors:** Level 2 subcategories (HF-DECISION, HF-PERCEPTUAL, HF-SKILL,
HF-VIOLATION) from the Human Factors Analysis and Classification System (HFACS) are
counted as distinct reports per HF type. The bar chart uses deduplicated counts (a report
is counted once per HF type regardless of how many parent categories it appears under).
The heatmap uses per-parent breakdowns to show interaction patterns.

**Regional classification:** Reports are classified by geographic region based on
accident location data extracted from report metadata.

**Limitations:**
- Survivorship bias: only investigated accidents are included; incidents and unreported
  events are not captured.
- Feature coverage gaps mean some analyses are based on subsets of the data.
- Multi-label tagging means a single accident can appear in multiple category counts.
- Risk ratios and co-occurrence counts are observational and do not imply causation.
- Earlier decades have fewer reports, which may reduce the stability of decade-level
  prevalence calculations.
"""
    )


# ── Helpers ───────────────────────────────────────────────────────────────


def _render_hazard_breakdown(
    df: pd.DataFrame,
    prefix: str,
    color: str,
    chart_title: str,
    chart_key: str,
    hfacs_label: str,
):
    """
    Render a hazard subcategory breakdown: specific subcats as horizontal bar,
    HFACS codes (HF-*) in an expander below.
    """
    if df.empty:
        st.info("No subcategory data available.")
        return

    # Split into specific subcategories and HFACS human factors
    specific = df[df["category_code"].str.startswith(prefix)].copy()
    hfacs = df[df["category_code"].str.startswith("HF-")].copy()

    if not specific.empty:
        top_sub = specific.iloc[0]
        second_sub = specific.iloc[1] if len(specific) > 1 else None

        fig = horizontal_bar(
            specific,
            x="report_count",
            y="category_name",
            title=chart_title,
            color=color,
            height=max(250, len(specific) * 40 + 80),
        )

        # Build dynamic insight
        insight_parts = [
            f"<b>{top_sub['category_name']}</b> leads with "
            f"{int(top_sub['report_count'])} reports."
        ]
        if second_sub is not None:
            insight_parts.append(
                f"<b>{second_sub['category_name']}</b> follows with "
                f"{int(second_sub['report_count'])}."
            )

        # Add persona-specific recommendations based on the hazard type
        if prefix == "LOC-I-":
            insight_parts.append(
                f"{abbr('UPRT')} directly "
                "targets the top failure modes. Verify that recurrent "
                "training includes slow-flight maneuvering, unusual attitude "
                "recovery, and spin awareness appropriate to aircraft type."
            )
        elif prefix == "CFIT-":
            insight_parts.append(
                f"Standardized approach procedures, mandatory use of "
                f"{abbr('TAWS')}/{abbr('GPWS')}, and "
                "strict adherence to minimum descent altitudes address "
                "the majority of these scenarios. Brief crews on terrain "
                "awareness for every flight into unfamiliar or mountainous "
                "airports."
            )

        chart_with_insight(
            fig,
            " ".join(insight_parts),
            insight_type="warning",
            chart_key=chart_key,
        )
    else:
        st.info(f"No {prefix} subcategory data available.")

    if not hfacs.empty:
        with st.expander(hfacs_label):
            st.markdown(
                f"Human factors codes from the {abbr('HFACS')} framework that appear "
                "alongside this hazard category. These represent the "
                "crew-level contributing factors behind the accidents — "
                "each one maps to a specific training intervention.",
                unsafe_allow_html=True,
            )
            fig_hf = horizontal_bar(
                hfacs,
                x="report_count",
                y="category_name",
                color=AMBER,
                height=max(220, len(hfacs) * 35 + 60),
            )
            st.plotly_chart(
                fig_hf, use_container_width=True, key=f"{chart_key}_hfacs"
            )
