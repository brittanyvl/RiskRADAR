"""
Fleet Safety Report — Professional consulting-style dashboard.

Persona: Fleet safety manager at a Part 121/135 operator.
Core question: "What risks should I prioritize for my fleet?"
"""

import streamlit as st
import pandas as pd
import numpy as np
from app.components import data_loader as dl
from app.components.charts import (
    horizontal_bar, grouped_bar, heatmap, line_chart, stacked_bar,
)
from app.components.report_layout import (
    page_header, kpi_row, section_divider, insight, sample_note,
    methodology_section, chart_with_insight,
)
from app.components.theme import STEEL, CORAL, AMBER, TEAL, NAVY, CHART_PALETTE


def render():
    # ── Load data ─────────────────────────────────────────────────────────
    summary = dl.dataset_summary()
    cat_counts = dl.category_counts()
    ac_risk = dl.risk_by_aircraft_category()
    mfr_risk = dl.risk_by_manufacturer(top_n=15)
    scf_pp = dl.scf_pp_breakdown()
    scf_np = dl.scf_np_breakdown()
    coocc = dl.cooccurrence_matrix()
    trends = dl.failure_trends_by_decade()

    n_accidents = summary["accident_reports"]
    n_manufacturers = summary["unique_manufacturers"]
    avg_cats = summary["avg_categories_per_report"]
    top_category = cat_counts.iloc[0]

    # ══════════════════════════════════════════════════════════════════════
    # 1. HEADER + KPIs
    # ══════════════════════════════════════════════════════════════════════

    page_header(
        "Fleet Safety Report",
        "Risk patterns across aircraft types, manufacturers, and component systems "
        "to support fleet safety prioritization.",
    )

    kpi_row([
        {
            "label": "Accident Reports",
            "value": f"{n_accidents:,}",
            "detail": f"{summary['date_range_start'][:4]}–{summary['date_range_end'][:4]}" if summary.get("date_range_start") else "",
        },
        {
            "label": "Top Risk Factor",
            "value": top_category["category_code"],
            "detail": f"{top_category['pct_of_reports']:.0f}% of reports",
        },
        {
            "label": "Aircraft Types",
            "value": f"{len(ac_risk['aircraft_category'].unique())}",
            "detail": "Distinct fleet categories",
        },
        {
            "label": "Avg. Categories",
            "value": f"{avg_cats:.1f}",
            "detail": "Per accident report",
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
        "Understanding which hazards dominate your fleet type helps focus safety programs."
    )

    # Get distinct aircraft categories
    aircraft_types = sorted(ac_risk["aircraft_category"].unique())

    if len(aircraft_types) == 0:
        st.warning("No aircraft category data available.")
    else:
        col_filter, col_compare = st.columns([1, 1])
        with col_filter:
            selected_type = st.selectbox(
                "Select aircraft type",
                aircraft_types,
                index=0,
                key="fleet_ac_type",
            )
        with col_compare:
            compare_options = ["None"] + [t for t in aircraft_types if t != selected_type]
            compare_type = st.selectbox(
                "Compare with (optional)",
                compare_options,
                index=0,
                key="fleet_ac_compare_sel",
            )

        if compare_type == "None":
            # ── Single aircraft type view ──
            type_data = ac_risk[ac_risk["aircraft_category"] == selected_type].copy()
            type_data = type_data.nlargest(10, "report_count")
            total_reports = type_data["total_in_category"].iloc[0] if not type_data.empty else 0

            fig = horizontal_bar(
                type_data,
                x="report_count",
                y="category_code",
                title=f"Top Risk Factors: {selected_type}",
                color=STEEL,
                height=max(300, len(type_data) * 32 + 60),
                show_values=True,
            )

            top3 = type_data.head(3)
            top3_text = ", ".join(
                f"**{row['category_code']}** ({row['prevalence_pct']:.0f}%)"
                for _, row in top3.iterrows()
            )
            chart_with_insight(
                fig,
                f"Based on {total_reports:,} {selected_type} accident reports, the top risk factors are "
                f"{top3_text}. Fleet managers operating {selected_type} aircraft should prioritize "
                f"training and mitigation programs targeting these categories.",
                chart_key="fleet_ac_single",
            )

        else:
            # ── Comparison view (top 4 categories across both types) ──
            both_types = ac_risk[ac_risk["aircraft_category"].isin([selected_type, compare_type])].copy()
            # Find top 4 categories by combined report count
            top_cats = (
                both_types.groupby("category_code")["report_count"]
                .sum()
                .nlargest(4)
                .index.tolist()
            )
            compare_data = both_types[both_types["category_code"].isin(top_cats)].copy()

            fig = grouped_bar(
                compare_data,
                x="category_code",
                y="prevalence_pct",
                group="aircraft_category",
                title=f"Risk Profile Comparison: {selected_type} vs {compare_type}",
                height=420,
                colors={selected_type: STEEL, compare_type: CORAL},
            )
            fig.update_yaxes(title_text="Prevalence (%)")

            chart_with_insight(
                fig,
                f"This comparison highlights how risk profiles differ between {selected_type} and "
                f"{compare_type} operations. Prevalence is the percentage of accident reports "
                f"in each aircraft type that involve a given category, making fleet sizes comparable.",
                chart_key="fleet_ac_compare",
            )

    # ══════════════════════════════════════════════════════════════════════
    # 3. MANUFACTURER RISK PROFILES
    # ══════════════════════════════════════════════════════════════════════

    section_divider()
    st.markdown("### Manufacturer Risk Profiles")
    st.markdown(
        "Which risk categories are most associated with each manufacturer? "
        "Differences may reflect aircraft design, operational use, or fleet age."
    )

    # Build stacked bar data for top 10 manufacturers with top 5 categories
    top_10_makes = (
        mfr_risk.groupby("aircraft_make")["report_count"]
        .sum()
        .nlargest(10)
        .index.tolist()
    )
    top_5_cats = (
        mfr_risk[mfr_risk["aircraft_make"].isin(top_10_makes)]
        .groupby("category_code")["report_count"]
        .sum()
        .nlargest(5)
        .index.tolist()
    )

    mfr_stacked = mfr_risk[
        (mfr_risk["aircraft_make"].isin(top_10_makes))
        & (mfr_risk["category_code"].isin(top_5_cats))
    ].copy()

    # Sort manufacturers by total report count
    make_order = (
        mfr_risk[mfr_risk["aircraft_make"].isin(top_10_makes)]
        .groupby("aircraft_make")["report_count"]
        .sum()
        .sort_values(ascending=True)
        .index.tolist()
    )
    mfr_stacked["aircraft_make"] = pd.Categorical(
        mfr_stacked["aircraft_make"], categories=make_order, ordered=True,
    )
    mfr_stacked = mfr_stacked.sort_values("aircraft_make")

    fig_mfr = stacked_bar(
        mfr_stacked,
        x="aircraft_make",
        y="report_count",
        group="category_code",
        title="Top 10 Manufacturers by Dominant Risk Categories",
        height=max(420, len(top_10_makes) * 38 + 60),
        orientation="h",
    )

    # Find the leading manufacturer and its dominant category
    lead_make = make_order[-1] if make_order else "Unknown"
    lead_cats = (
        mfr_risk[mfr_risk["aircraft_make"] == lead_make]
        .nlargest(2, "report_count")
    )
    if not lead_cats.empty:
        lead_text = " and ".join(lead_cats["category_code"].tolist())
        lead_detail = (
            f"**{lead_make}** has the most accident reports in the dataset, "
            f"with {lead_text} as the most frequent categories. "
        )
    else:
        lead_detail = ""

    chart_with_insight(
        fig_mfr,
        f"{lead_detail}"
        f"Note: 'Douglas,' 'McDonnell Douglas,' and 'McDonnell-Douglas' are merged "
        f"into a single manufacturer entry. Report counts reflect how many reports mention "
        f"each risk category, not the number of events per manufacturer.",
        chart_key="fleet_mfr_stacked",
    )

    # ══════════════════════════════════════════════════════════════════════
    # 4. COMPONENT FAILURES (SCF-PP & SCF-NP)
    # ══════════════════════════════════════════════════════════════════════

    section_divider()
    st.markdown("### Component Failure Breakdown")
    st.markdown(
        "System/Component Failure categories — Powerplant (SCF-PP) and Non-Powerplant "
        "(SCF-NP) — are among the most actionable findings for maintenance and fleet "
        "programs. Understanding which subsystems fail most often guides inspection priorities."
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
            insight(
                f"**{top_pp['category_name']}** accounts for "
                f"{top_pp['report_count']} of {total_pp} powerplant failure reports "
                f"({top_pp['report_count'] / total_pp * 100:.0f}%). "
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
            insight(
                f"**{top_np['category_name']}** leads non-powerplant failures with "
                f"{top_np['report_count']} reports ({top_np['report_count'] / total_np * 100:.0f}%). "
                f"Non-powerplant systems require dedicated inspection and maintenance tracking.",
            )
        else:
            st.info("No SCF-NP subcategory data available.")

    # ══════════════════════════════════════════════════════════════════════
    # 5. CO-OCCURRENCE PATTERNS
    # ══════════════════════════════════════════════════════════════════════

    section_divider()
    st.markdown("### Risk Factor Co-occurrence")
    st.markdown(
        "When two risk categories frequently appear together in the same report, "
        "it signals a cascading failure pattern. These combinations deserve "
        "attention in safety management systems and training scenarios."
    )

    # Filter to top 12 categories by total report count for readability
    top_12_codes = cat_counts.head(12)["category_code"].tolist()
    coocc_filtered = coocc.loc[
        coocc.index.isin(top_12_codes),
        coocc.columns.isin(top_12_codes),
    ]
    # Reorder to match frequency ranking
    coocc_filtered = coocc_filtered.reindex(
        index=top_12_codes, columns=top_12_codes,
    )

    fig_coocc = heatmap(
        coocc_filtered,
        title="Category Co-occurrence (top 12 categories)",
        height=520,
        mask_diagonal=True,
        lower_triangle_only=True,
        annotation_threshold=15,
    )

    # Find the strongest off-diagonal co-occurrence
    coocc_vals = coocc_filtered.values.astype(float).copy()
    np.fill_diagonal(coocc_vals, 0)
    max_idx = np.unravel_index(coocc_vals.argmax(), coocc_vals.shape)
    max_cat1 = coocc_filtered.index[max_idx[0]]
    max_cat2 = coocc_filtered.columns[max_idx[1]]
    max_val = int(coocc_vals[max_idx])

    chart_with_insight(
        fig_coocc,
        f"The strongest co-occurrence is between **{max_cat1}** and **{max_cat2}**, "
        f"appearing together in {max_val} reports. The diagonal and upper triangle "
        f"are removed — each pair appears once in the lower half. Annotated cells "
        f"(15 or more shared reports) indicate meaningful risk combinations "
        f"that warrant integrated training and mitigation strategies.",
        chart_key="fleet_coocc",
    )

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
        fig_trend = line_chart(
            trends,
            x="decade",
            y="prevalence_pct",
            color="category_code",
            title="Powerplant vs Non-Powerplant Failure Prevalence by Decade",
            height=400,
            y_label="Prevalence (%)",
        )

        # Build narrative from the data
        latest_decade = trends["decade"].max()
        latest_data = trends[trends["decade"] == latest_decade]
        trend_parts = []
        for _, row in latest_data.iterrows():
            trend_parts.append(f"{row['category_code']} at {row['prevalence_pct']:.0f}%")
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
    # 7. METHODOLOGY
    # ══════════════════════════════════════════════════════════════════════

    section_divider()
    methodology_section(
        f"""
**Data source:** {n_accidents} National Transportation Safety Board (NTSB) accident
investigation reports, classified using the CAST/ICAO Common Taxonomy Team (CICTT)
occurrence category framework.

**Classification:** Each report is assigned one or more Level 1 categories (27 total) and,
where applicable, Level 2 subcategories (32 total). Classification was performed using a
fine-tuned language model with human review of edge cases.

**Prevalence calculation:** Prevalence is the percentage of reports in a given group
(aircraft type, manufacturer, decade) that mention a particular risk category. Because
reports can have multiple categories (average of {avg_cats:.1f} per report), prevalence
percentages across categories will sum to more than 100%.

**Manufacturer merging:** Douglas, McDonnell Douglas, and McDonnell-Douglas entries are
combined into a single "Douglas / McDonnell Douglas" manufacturer, reflecting the
corporate lineage of these aircraft.

**Limitations:**
- The dataset reflects investigated accidents only, not all aviation incidents or the broader fleet population.
- Report availability varies by decade; earlier periods have fewer reports.
- Category assignments depend on the information available in each report and may not capture all contributing factors.
- Co-occurrence counts reflect shared categorization, not proven causal chains.
"""
    )
