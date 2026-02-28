"""
Fleet Safety Report — Professional consulting-style dashboard.

Persona: Fleet safety manager at a Part 121/135 operator.
Core question: "What risks should I prioritize for my fleet?"
"""

import streamlit as st
from app.components import data_loader as dl
from app.components.charts import (
    horizontal_bar, grouped_bar, heatmap, line_chart, diverging_bar,
    vertical_bar,
)
from app.components.report_layout import (
    page_header, kpi_row, section_divider, insight, sample_note,
    methodology_section, chart_with_insight, ABBREVIATIONS, abbr,
    coverage_note,
)
from app.components.theme import STEEL, CORAL, AMBER, TEAL, NAVY, CHART_PALETTE


def render():
    # ── Load data ─────────────────────────────────────────────────────────
    summary = dl.dataset_summary()
    ac_risk = dl.risk_by_aircraft_category()
    scf_pp = dl.scf_pp_breakdown()
    scf_np = dl.scf_np_breakdown()
    trends = dl.failure_trends_by_decade()

    n_accidents = summary["accident_reports"]
    avg_cats = summary["avg_categories_per_report"]

    # ══════════════════════════════════════════════════════════════════════
    # 1. HEADER + KPIs
    # ══════════════════════════════════════════════════════════════════════

    page_header(
        "Fleet Safety Report",
        "Risk patterns across aircraft types, manufacturers, and component systems "
        "to support fleet safety prioritization.",
    )

    # Compute engine failure share — SCF-PP only denominator
    _total_pp = scf_pp["report_count"].sum() if not scf_pp.empty else 0
    _engine_count = scf_pp.iloc[0]["report_count"] if not scf_pp.empty else 0
    _engine_pct = round(_engine_count / _total_pp * 100) if _total_pp > 0 else 0

    # Night accident rate
    night_stats = dl.night_accident_share()

    # Row 1 (4 KPIs): LOC-I/CFIT, Component Failure, Engine Failure Share, IMC
    kpi_row([
        {
            "label": "LOC-I / CFIT Rate",
            "value": f"{summary['loci_cfit_pct']:.0f}%",
            "detail": "Of accidents involve loss of control or CFIT",
            "accent": CORAL,
        },
        {
            "label": "Component Failures",
            "value": f"{summary['component_failure_pct']:.0f}%",
            "detail": "Of accidents cite a system/component failure",
            "accent": AMBER,
        },
        {
            "label": "Engine Failure Share",
            "value": f"{_engine_pct}%",
            "detail": f"{_engine_count} of {_total_pp} powerplant failures are engine-related",
            "accent": CORAL,
        },
        {
            "label": "IMC Involvement",
            "value": f"{summary['imc_pct']:.0f}%",
            "detail": "Of weather-classified accidents occurred in Instrument Meteorological Conditions",
            "accent": STEEL,
        },
    ])

    # Row 2 (2 KPIs): Night Accident Rate, Avg Contributing Factors
    kpi_row([
        {
            "label": "Night Accident Rate",
            "value": f"{night_stats['night_pct']:.0f}%",
            "detail": f"Of {night_stats['total_with_time']:,} accidents with known time occurred at night",
            "accent": NAVY,
        },
        {
            "label": "Avg. Contributing Factors",
            "value": f"{avg_cats:.1f}",
            "detail": "CICTT categories assigned per accident report",
            "accent": TEAL,
        },
    ])

    sample_note(n_accidents, "NTSB accident reports classified with CAST/ICAO Common Taxonomy Team (CICTT) taxonomy")

    # ══════════════════════════════════════════════════════════════════════
    # 2. ACCIDENT CATEGORIES BY AIRCRAFT TYPE
    # ══════════════════════════════════════════════════════════════════════

    section_divider()
    st.markdown("### Accident Categories by Aircraft Type")
    st.markdown(
        "Different aircraft types face fundamentally different risk profiles. "
        "Understanding which hazards dominate your fleet type helps focus safety programs. "
        "Select one or more aircraft types below to compare."
    )

    # Get distinct aircraft categories, default to top 3 by report volume
    type_totals = (
        ac_risk.groupby("aircraft_category")["total_in_category"]
        .first()
        .sort_values(ascending=False)
    )
    aircraft_types = type_totals.index.tolist()
    default_types = aircraft_types[:3]

    if len(aircraft_types) == 0:
        st.warning("No aircraft category data available.")
    else:
        selected_types = st.multiselect(
            "Select aircraft types to compare",
            options=aircraft_types,
            default=default_types,
            key="fleet_ac_types",
        )

        if not selected_types:
            st.info("Select at least one aircraft type above.")
        elif len(selected_types) == 1:
            # ── Single aircraft type view ──
            sel = selected_types[0]
            type_data = ac_risk[ac_risk["aircraft_category"] == sel].copy()
            type_data = type_data.nlargest(10, "report_count")
            total_reports = type_data["total_in_category"].iloc[0] if not type_data.empty else 0

            # Use full category names on y-axis
            type_data["category_label"] = type_data["category_code"].map(
                lambda c: ABBREVIATIONS.get(c, c)
            )

            fig = horizontal_bar(
                type_data,
                x="report_count",
                y="category_label",
                title=f"Top Risk Factors: {sel}",
                color=STEEL,
                height=max(300, len(type_data) * 36 + 60),
                show_values=True,
            )

            top3 = type_data.head(3)
            top3_text = ", ".join(
                f"<b>{ABBREVIATIONS.get(row['category_code'], row['category_code'])}</b> ({row['prevalence_pct']:.0f}%)"
                for _, row in top3.iterrows()
            )
            chart_with_insight(
                fig,
                f"Based on {total_reports:,} {sel} accident reports, the top risk factors are "
                f"{top3_text}. Fleet managers operating {sel} aircraft should prioritize "
                f"training and mitigation programs targeting these categories.",
                chart_key="fleet_ac_single",
            )

        else:
            # ── Multi-type comparison (top 6 categories across selected types) ──
            multi_data = ac_risk[ac_risk["aircraft_category"].isin(selected_types)].copy()
            top_cats = (
                multi_data.groupby("category_code")["report_count"]
                .sum()
                .nlargest(6)
                .index.tolist()
            )
            compare_data = multi_data[multi_data["category_code"].isin(top_cats)].copy()

            # Use full names on x-axis, wrapped with <br> for long labels
            def _wrap_label(code):
                name = ABBREVIATIONS.get(code, code)
                words = name.split()
                if len(words) <= 3:
                    return name
                mid = len(words) // 2
                return " ".join(words[:mid]) + "<br>" + " ".join(words[mid:])

            compare_data["category_label"] = compare_data["category_code"].map(_wrap_label)

            # Assign colors from palette
            type_colors = {t: CHART_PALETTE[i % len(CHART_PALETTE)] for i, t in enumerate(selected_types)}

            fig = grouped_bar(
                compare_data,
                x="category_label",
                y="prevalence_pct",
                group="aircraft_category",
                title=f"Risk Profile Comparison ({len(selected_types)} aircraft types)",
                height=max(420, 420 + (len(selected_types) - 3) * 20),
                colors=type_colors,
            )
            fig.update_yaxes(title_text="Prevalence (%)")
            fig.update_xaxes(tickangle=0)

            type_list = ", ".join(f"<b>{t}</b>" for t in selected_types)
            chart_with_insight(
                fig,
                f"Comparing risk profiles across {type_list}. "
                f"Prevalence is the percentage of accident reports in each aircraft type "
                f"that involve a given category, making fleet sizes comparable regardless "
                f"of total report volume.",
                chart_key="fleet_ac_compare",
            )

    # ══════════════════════════════════════════════════════════════════════
    # 3. MANUFACTURER RISK PROFILES (Heatmap)
    # ══════════════════════════════════════════════════════════════════════

    section_divider()
    st.markdown("### Manufacturer Risk Profiles")
    st.markdown(
        "Which risk categories are most associated with each manufacturer? "
        "Values show what <b>percentage</b> of each manufacturer's accident reports "
        "involve a given category — normalizing for fleet size so high-volume "
        "manufacturers don't overwhelm the view.",
        unsafe_allow_html=True,
    )

    mfr_heatmap_data = dl.manufacturer_category_heatmap(top_n=10)

    if not mfr_heatmap_data.empty:
        # Filter to categories that appear in at least one manufacturer at >= 10%
        col_max = mfr_heatmap_data.max(axis=0)
        visible_cols = col_max[col_max >= 10].index.tolist()
        if visible_cols:
            mfr_heatmap_data = mfr_heatmap_data[visible_cols]

        fig_mfr = heatmap(
            mfr_heatmap_data,
            title="Manufacturer x Category Prevalence (%)",
            height=max(450, len(mfr_heatmap_data) * 40 + 80),
            value_format="pct",
            colorbar_title="Prevalence %",
            hover_labels=ABBREVIATIONS,
        )

        chart_with_insight(
            fig_mfr,
            "Each cell shows the percentage of a manufacturer's accident reports that "
            "involve a given risk category. This normalization means a manufacturer with "
            "10 reports and one with 100 reports are compared on equal footing. "
            "Note: Douglas, McDonnell Douglas, and McDonnell-Douglas are merged into "
            "a single entry.",
            chart_key="fleet_mfr_heatmap",
        )
    else:
        st.info("No manufacturer heatmap data available.")

    # ══════════════════════════════════════════════════════════════════════
    # 4. COMPONENT FAILURES (SCF-PP & SCF-NP)
    # ══════════════════════════════════════════════════════════════════════

    section_divider()
    st.markdown("### Component Failure Breakdown")
    st.markdown(
        f"System/Component Failure categories — {abbr('SCF-PP')} and {abbr('SCF-NP')} "
        "— are among the most actionable findings for maintenance and fleet "
        "programs. Understanding which subsystems fail most often guides inspection priorities.",
        unsafe_allow_html=True,
    )

    col_pp, col_np = st.columns(2)

    with col_pp:
        if not scf_pp.empty:
            fig_pp = horizontal_bar(
                scf_pp,
                x="report_count",
                y="category_name",
                title="Powerplant Failures",
                color=CORAL,
                height=max(260, len(scf_pp) * 36 + 60),
            )
            st.plotly_chart(fig_pp, use_container_width=True, key="fleet_scf_pp")

            top_pp = scf_pp.iloc[0]
            total_pp = scf_pp["report_count"].sum()
            pp_pct = top_pp['report_count'] / total_pp * 100 if total_pp > 0 else 0
            insight(
                f"<b>{top_pp['category_name']}</b> accounts for "
                f"{top_pp['report_count']} of {total_pp} powerplant failure reports "
                f"({pp_pct:.0f}%). "
                f"Engine-related programs should be the primary focus of powerplant safety initiatives.",
            )
        else:
            st.info("No SCF-PP subcategory data available.")

    with col_np:
        if not scf_np.empty:
            fig_np = horizontal_bar(
                scf_np,
                x="report_count",
                y="category_name",
                title="Non-Powerplant Failures",
                color=TEAL,
                height=max(260, len(scf_np) * 36 + 60),
            )
            st.plotly_chart(fig_np, use_container_width=True, key="fleet_scf_np")

            top_np = scf_np.iloc[0]
            total_np = scf_np["report_count"].sum()
            np_pct = top_np['report_count'] / total_np * 100 if total_np > 0 else 0
            insight(
                f"<b>{top_np['category_name']}</b> leads non-powerplant failures with "
                f"{top_np['report_count']} reports ({np_pct:.0f}%). "
                f"Non-powerplant systems require dedicated inspection and maintenance tracking.",
            )
        else:
            st.info("No SCF-NP subcategory data available.")

    # ══════════════════════════════════════════════════════════════════════
    # 5. RISK COMPLEXITY — CONCURRENT FAILURE CATEGORIES
    # ══════════════════════════════════════════════════════════════════════

    section_divider()
    st.markdown("### Risk Complexity — Concurrent Failure Categories")
    st.markdown(
        "How many failure categories does each accident involve? Accidents with "
        "multiple concurrent categories indicate compounding hazards that require "
        "broader mitigation strategies, not just single-issue fixes."
    )

    complexity = dl.multi_label_complexity()
    if not complexity.empty:
        complexity["label"] = complexity["n_categories"].astype(str).apply(
            lambda x: f"{x} category" if x == "1" else f"{x} categories"
        )

        fig_complexity = vertical_bar(
            complexity,
            x="label",
            y="report_count",
            title="Accidents by Number of Concurrent Failure Categories",
            color=STEEL,
            height=350,
        )

        three_plus = complexity[complexity["n_categories"].isin(["3", "4+"])]
        three_plus_pct = three_plus["pct"].sum() if not three_plus.empty else 0

        chart_with_insight(
            fig_complexity,
            f"<b>{three_plus_pct:.0f}%</b> of accidents involve 3+ concurrent failure "
            f"categories, meaning maintenance and training programs must address "
            f"compounding hazards, not isolated failures. Single-category accidents "
            f"({complexity.iloc[0]['pct']:.0f}%) are the minority — most events involve "
            f"multiple interacting risk factors.",
            chart_key="fleet_complexity",
        )
    else:
        st.info("No complexity data available.")

    # ══════════════════════════════════════════════════════════════════════
    # 6. TEMPORAL TRENDS (SCF-PP & SCF-NP)
    # ══════════════════════════════════════════════════════════════════════

    section_divider()
    st.markdown("### Component Failure Trends Over Time")
    st.markdown(
        "How has the prevalence of powerplant and non-powerplant failures "
        "changed across decades? Trends may reflect improvements in design, "
        "maintenance practices, or shifts in fleet composition."
    )

    if not trends.empty:
        # Map category codes to full names for the legend and hover
        trends_display = trends.copy()
        trends_display["category_label"] = trends_display["category_code"].map(
            lambda c: f"{c} ({ABBREVIATIONS.get(c, c)})"
        )

        fig_trend = line_chart(
            trends_display,
            x="decade",
            y="prevalence_pct",
            color="category_label",
            title="Powerplant vs Non-Powerplant Failure Prevalence by Decade",
            height=400,
            y_label="Prevalence (%)",
        )

        # Build narrative from the data
        latest_decade = trends["decade"].max()
        latest_data = trends[trends["decade"] == latest_decade]
        trend_parts = []
        for _, row in latest_data.iterrows():
            full_name = ABBREVIATIONS.get(row["category_code"], row["category_code"])
            trend_parts.append(
                f"{row['category_code']} ({full_name}) at {row['prevalence_pct']:.0f}%"
            )
        trend_summary = " and ".join(trend_parts) if trend_parts else "data unavailable"

        chart_with_insight(
            fig_trend,
            f"In the most recent decade ({latest_decade}s), component failure prevalence stands at "
            f"{trend_summary}. Prevalence is measured as the percentage of accident reports "
            f"in each decade that cite the given failure category, controlling for the "
            f"varying number of reports across time periods.",
            chart_key="fleet_trends",
        )
    else:
        st.info("No temporal trend data available for component failures.")

    # ══════════════════════════════════════════════════════════════════════
    # 7. KEY RISK TRENDS BY DECADE
    # ══════════════════════════════════════════════════════════════════════

    section_divider()
    st.markdown("### Fleet Risk Trends by Decade")
    st.markdown(
        "How have the most critical fleet safety categories evolved over time? "
        "Tracking prevalence by decade reveals whether risks are improving, "
        "worsening, or holding steady — essential for long-range safety planning."
    )

    risk_trends = dl.key_risk_trends_by_decade()
    if not risk_trends.empty:
        # Map codes to full names for legend
        risk_trends["category_label"] = risk_trends["category_code"].map(
            lambda c: f"{c} ({ABBREVIATIONS.get(c, c)})"
        )

        fig_risk_trends = line_chart(
            risk_trends,
            x="decade",
            y="prevalence_pct",
            color="category_label",
            title="Key Risk Category Prevalence by Decade",
            height=450,
            y_label="Prevalence (%)",
        )

        # Identify biggest movers (latest vs earliest decade)
        decades = sorted(risk_trends["decade"].unique())
        if len(decades) >= 2:
            earliest = risk_trends[risk_trends["decade"] == decades[0]]
            latest = risk_trends[risk_trends["decade"] == decades[-1]]
            merged = earliest[["category_code", "prevalence_pct"]].merge(
                latest[["category_code", "prevalence_pct"]],
                on="category_code", suffixes=("_early", "_late"),
            )
            merged["change"] = merged["prevalence_pct_late"] - merged["prevalence_pct_early"]
            biggest_rise = merged.loc[merged["change"].idxmax()]
            biggest_drop = merged.loc[merged["change"].idxmin()]
            trend_insight = (
                f"From the {decades[0]}s to the {decades[-1]}s, "
                f"<b>{ABBREVIATIONS.get(biggest_rise['category_code'], biggest_rise['category_code'])}</b> "
                f"saw the largest increase (+{biggest_rise['change']:.0f} pp), while "
                f"<b>{ABBREVIATIONS.get(biggest_drop['category_code'], biggest_drop['category_code'])}</b> "
                f"showed the greatest decline ({biggest_drop['change']:.0f} pp). "
                f"Prevalence is normalized per decade to account for varying report volumes."
            )
        else:
            trend_insight = "Insufficient decades for trend comparison."

        chart_with_insight(fig_risk_trends, trend_insight, chart_key="fleet_risk_trends")
    else:
        st.info("No decade trend data available.")

    # ══════════════════════════════════════════════════════════════════════
    # 8. LOC-I SUBTYPES BY AIRCRAFT TYPE
    # ══════════════════════════════════════════════════════════════════════

    section_divider()
    st.markdown("### LOC-I (Loss of Control — In Flight) Subtypes")
    st.markdown(
        f"{abbr('LOC-I')} is the leading cause of fatal accidents. "
        "Understanding <b>which type</b> of control loss dominates for each fleet type "
        f"guides specific {abbr('UPRT')} priorities.",
        unsafe_allow_html=True,
    )

    loci_data = dl.loci_subtypes_by_aircraft()
    if not loci_data.empty:
        # Build heatmap: aircraft_category x LOC-I subtype (% of LOC-I accidents)
        _loci_names = {
            "LOC-I-UPSET": "Aircraft<br>Upset",
            "LOC-I-STALL": "Aerodynamic<br>Stall",
            "LOC-I-ENV": "Environmental<br>LOC",
            "LOC-I-SYS": "System-Induced<br>LOC",
            "LOC-I-LOAD": "Loading/CG<br>Issues",
            "LOC-I-SD": "Spatial<br>Disorientation",
        }
        loci_data["subtype_name"] = loci_data["category_code"].map(
            lambda c: _loci_names.get(c, c)
        )

        loci_pivot = loci_data.pivot_table(
            index="aircraft_category",
            columns="subtype_name",
            values="pct_of_loci",
            fill_value=0,
        )
        # Sort by total LOC-I reports
        loci_totals = loci_data.groupby("aircraft_category")["report_count"].sum().sort_values(ascending=False)
        loci_pivot = loci_pivot.reindex(loci_totals.index)

        # Filter to aircraft types with enough LOC-I data
        loci_pivot = loci_pivot[loci_totals >= 3]

        if not loci_pivot.empty:
            fig_loci = heatmap(
                loci_pivot,
                title="LOC-I Subtypes by Aircraft Type (% of LOC-I accidents)",
                height=max(350, len(loci_pivot) * 38 + 80),
                colorbar_title="% of LOC-I",
                value_format="pct",
            )
            fig_loci.update_xaxes(tickangle=0)

            chart_with_insight(
                fig_loci,
                "Each cell shows what percentage of that aircraft type's "
                f"{abbr('LOC-I')} accidents involved the given subtype. This normalization "
                "means aircraft types with few LOC-I events are compared fairly against "
                "those with many. <b>Stall</b> and <b>Upset</b> dominate across most fleet "
                "types, but the balance varies — training programs should target the dominant "
                "LOC-I mechanism for each fleet type.",
                chart_key="fleet_loci_subtypes",
            )
        else:
            st.info("Insufficient LOC-I subtype data for aircraft type comparison.")
    else:
        st.info("No LOC-I subtype data available.")

    # ══════════════════════════════════════════════════════════════════════
    # 9. WEATHER-CONDITIONAL RISK — IMC vs VMC DIVERGING CHART
    # ══════════════════════════════════════════════════════════════════════

    section_divider()
    st.markdown("### IMC vs VMC Risk Comparison")
    st.markdown(
        "How does the risk profile change when flying in poor weather? This chart "
        f"compares the prevalence of each category in {abbr('IMC')} versus "
        f"{abbr('VMC')}. Categories that skew heavily to one side indicate "
        "weather-sensitive risks.",
        unsafe_allow_html=True,
    )

    wrr = dl.weather_risk_ratios()
    if not wrr.empty:
        # Filter to categories with meaningful combined sample
        wrr_filtered = wrr[
            (wrr["imc_count"] + wrr["vmc_count"] >= 10)
            & (wrr["risk_ratio"].notna())
        ].copy()

        if not wrr_filtered.empty:
            wrr_filtered["category_label"] = wrr_filtered["category_code"].map(
                lambda c: ABBREVIATIONS.get(c, c)
            )

            # Sort by difference (IMC - VMC) for visual impact
            wrr_filtered["diff"] = wrr_filtered["imc_prevalence"] - wrr_filtered["vmc_prevalence"]
            wrr_filtered = wrr_filtered.sort_values("diff", ascending=True)

            fig_div = diverging_bar(
                wrr_filtered,
                y="category_label",
                left_col="vmc_prevalence",
                right_col="imc_prevalence",
                left_label="VMC (clear weather)",
                right_label="IMC (poor weather)",
                left_color=TEAL,
                right_color=CORAL,
                title="Category Prevalence: VMC vs IMC",
                height=max(450, len(wrr_filtered) * 30 + 80),
            )

            # Find biggest IMC-skewed and VMC-skewed categories
            most_imc = wrr_filtered.iloc[-1]
            most_vmc = wrr_filtered.iloc[0]
            chart_with_insight(
                fig_div,
                f"<b>{most_imc['category_label']}</b> is most overrepresented in "
                f"{abbr('IMC')} ({most_imc['imc_prevalence']:.0f}% vs "
                f"{most_imc['vmc_prevalence']:.0f}% in {abbr('VMC')}), while "
                f"<b>{most_vmc['category_label']}</b> skews heavily toward "
                f"{abbr('VMC')} ({most_vmc['vmc_prevalence']:.0f}% vs "
                f"{most_vmc['imc_prevalence']:.0f}% in {abbr('IMC')}). "
                f"Categories that extend further right warrant focused instrument training; "
                f"those extending left are primarily clear-weather operational risks.",
                chart_key="fleet_weather_diverge",
            )

            coverage_note("Weather", summary["weather_coverage_pct"], n_accidents)
        else:
            st.info("Insufficient weather data for comparison.")
    else:
        st.info("No weather risk data available.")

    # ══════════════════════════════════════════════════════════════════════
    # 10. METHODOLOGY
    # ══════════════════════════════════════════════════════════════════════

    section_divider()
    methodology_section(
        f"""
<b>Data source:</b> {n_accidents} National Transportation Safety Board (NTSB) accident
investigation reports, classified using the CAST/ICAO Common Taxonomy Team (CICTT)
occurrence category framework.

<b>Classification:</b> Each report is assigned one or more Level 1 categories (27 total) and,
where applicable, Level 2 subcategories (32 total). Classification was performed using a
fine-tuned language model with human review of edge cases.

<b>Prevalence calculation:</b> Prevalence is the percentage of reports in a given group
(aircraft type, manufacturer, decade) that mention a particular risk category. Because
reports can have multiple categories (average of {avg_cats:.1f} per report), prevalence
percentages across categories will sum to more than 100%.

<b>Manufacturer merging:</b> Douglas, McDonnell Douglas, and McDonnell-Douglas entries are
combined into a single "Douglas / McDonnell Douglas" manufacturer, reflecting the
corporate lineage of these aircraft.

<b>Headline KPIs:</b> LOC-I/CFIT rate counts reports involving either Loss of Control in
Flight or Controlled Flight Into Terrain. Component Failure rate counts reports citing
SCF-PP or SCF-NP. IMC involvement counts reports where weather was classified as
Instrument Meteorological Conditions ({summary['weather_coverage_pct']:.0f}% coverage).
Night Accident Rate uses the subset of reports with known time of day
({night_stats['total_with_time']:,} reports).

<b>Risk complexity:</b> Each report's number of concurrent failure categories is counted.
The "4+" bucket aggregates reports with four or more categories. This metric reflects
the multi-causal nature of most aviation accidents.

<b>Risk ratios (IMC vs VMC):</b> Computed as (category prevalence in IMC) / (category
prevalence in VMC). A ratio of 2.0 means the category appears in twice the share of
IMC accidents vs VMC accidents. Weather data is available for {summary['weather_coverage_pct']:.0f}%
of accident reports.

<b>Limitations:</b>
- The dataset reflects investigated accidents only, not all aviation incidents or the broader fleet population.
- Report availability varies by decade; earlier periods have fewer reports.
- Category assignments depend on the information available in each report and may not capture all contributing factors.
"""
    )
