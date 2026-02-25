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
    horizontal_bar, grouped_bar, heatmap, line_chart, diverging_bar,
)
from app.components.report_layout import (
    page_header, kpi_row, section_divider, insight, sample_note,
    methodology_section, chart_with_insight, ABBREVIATIONS,
)
from app.components.theme import STEEL, CORAL, AMBER, TEAL, NAVY, CHART_PALETTE


def render():
    # ── Load data ─────────────────────────────────────────────────────────
    summary = dl.dataset_summary()
    cat_counts = dl.category_counts()
    ac_risk = dl.risk_by_aircraft_category()
    scf_pp = dl.scf_pp_breakdown()
    scf_np = dl.scf_np_breakdown()
    coocc = dl.cooccurrence_matrix()
    trends = dl.failure_trends_by_decade()

    n_accidents = summary["accident_reports"]
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

    # Compute top human factor and top component for KPIs
    _hf_data = dl.human_factors_breakdown()
    _top_hf = _hf_data.iloc[0] if not _hf_data.empty else None

    # Compute engine failure share for a more specific maintenance KPI
    _total_scf = scf_pp["report_count"].sum() + scf_np["report_count"].sum() if (not scf_pp.empty and not scf_np.empty) else 0
    _engine_count = scf_pp.iloc[0]["report_count"] if not scf_pp.empty else 0
    _engine_pct = round(_engine_count / _total_scf * 100) if _total_scf > 0 else 0

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
            "label": "IMC Involvement",
            "value": f"{summary['imc_pct']:.0f}%",
            "detail": "Of accidents occurred in instrument conditions",
            "accent": STEEL,
        },
        {
            "label": "Avg. Contributing Factors",
            "value": f"{avg_cats:.1f}",
            "detail": "CICTT categories assigned per accident report",
            "accent": TEAL,
        },
    ])

    # Second KPI row: human factors + maintenance specifics
    if _top_hf is not None:
        kpi_row([
            {
                "label": "Top Human Factor",
                "value": "Procedural Violation",
                "detail": f"{_top_hf['report_count']} reports — priority for compliance training",
                "accent": NAVY,
            },
            {
                "label": "Engine Failure Share",
                "value": f"{_engine_pct}%",
                "detail": f"{_engine_count} of {_total_scf} component failures are engine-related",
                "accent": CORAL,
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
                f"**{ABBREVIATIONS.get(row['category_code'], row['category_code'])}** ({row['prevalence_pct']:.0f}%)"
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

            type_list = ", ".join(f"**{t}**" for t in selected_types)
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
        "Values show what **percentage** of each manufacturer's accident reports "
        "involve a given category — normalizing for fleet size so high-volume "
        "manufacturers don't overwhelm the view."
    )

    mfr_heatmap_data = dl.manufacturer_category_heatmap(top_n=10)

    if not mfr_heatmap_data.empty:
        # Filter to categories that appear in at least one manufacturer at ≥ 10%
        col_max = mfr_heatmap_data.max(axis=0)
        visible_cols = col_max[col_max >= 10].index.tolist()
        if visible_cols:
            mfr_heatmap_data = mfr_heatmap_data[visible_cols]

        fig_mfr = heatmap(
            mfr_heatmap_data,
            title="Manufacturer × Category Prevalence (%)",
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
        height=560,
        mask_diagonal=True,
        lower_triangle_only=True,
        hover_labels=ABBREVIATIONS,
        colorbar_title="Shared Reports",
    )

    # Find the strongest off-diagonal co-occurrence
    coocc_vals = coocc_filtered.values.astype(float).copy()
    np.fill_diagonal(coocc_vals, 0)
    max_idx = np.unravel_index(coocc_vals.argmax(), coocc_vals.shape)
    max_cat1 = coocc_filtered.index[max_idx[0]]
    max_cat2 = coocc_filtered.columns[max_idx[1]]
    max_val = int(coocc_vals[max_idx])

    # Acronym reference for axis labels
    coocc_codes = top_12_codes
    abbr_ref = " · ".join(
        f"**{code}** = {ABBREVIATIONS.get(code, code)}"
        for code in coocc_codes if code in ABBREVIATIONS
    )

    chart_with_insight(
        fig_coocc,
        f"The strongest co-occurrence is between **{max_cat1}** and **{max_cat2}**, "
        f"appearing together in {max_val} reports. Each cell shows the number of "
        f"reports where both categories were assigned. Hover over any cell to see "
        f"full category names.",
        chart_key="fleet_coocc",
    )
    # Expandable acronym reference
    with st.expander("Category Acronym Reference"):
        st.markdown(abbr_ref, unsafe_allow_html=True)

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
                f"**{ABBREVIATIONS.get(biggest_rise['category_code'], biggest_rise['category_code'])}** "
                f"saw the largest increase (+{biggest_rise['change']:.0f} pp), while "
                f"**{ABBREVIATIONS.get(biggest_drop['category_code'], biggest_drop['category_code'])}** "
                f"showed the greatest decline ({biggest_drop['change']:.0f} pp). "
                f"Prevalence is normalized per decade to account for varying report volumes."
            )
        else:
            trend_insight = "Insufficient decades for trend comparison."

        chart_with_insight(fig_risk_trends, trend_insight, chart_key="fleet_risk_trends")
    else:
        st.info("No decade trend data available.")

    # ══════════════════════════════════════════════════════════════════════
    # 8. HUMAN FACTORS — TRAINING PRIORITIES
    # ══════════════════════════════════════════════════════════════════════

    section_divider()
    st.markdown("### Human Factors — Training Priorities")
    st.markdown(
        "Human factors subcategories reveal **why** accidents happen, not just what "
        "happened. These findings directly inform crew resource management (CRM), "
        "upset prevention (UPRT), and procedural compliance training programs."
    )

    hf_data = dl.human_factors_breakdown()
    if not hf_data.empty:
        # Add friendly names
        _hf_names = {
            "HF-VIOLATION": "Procedural Violation",
            "HF-PERCEPTUAL": "Perceptual Error",
            "HF-DECISION": "Decision Error",
            "HF-SKILL": "Skill-Based Error",
        }
        hf_data["display_name"] = hf_data["category_code"].map(
            lambda c: _hf_names.get(c, c)
        )

        fig_hf = horizontal_bar(
            hf_data,
            x="report_count",
            y="display_name",
            title="Human Factors in Accident Reports",
            color=NAVY,
            height=max(250, len(hf_data) * 45 + 60),
            show_values=True,
        )

        total_hf = hf_data["report_count"].sum()
        top_hf = hf_data.iloc[0]
        chart_with_insight(
            fig_hf,
            f"**{top_hf['display_name']}** is the most common human factor, appearing in "
            f"{top_hf['report_count']} reports. Violations and perceptual errors together "
            f"account for the majority of human-factors findings — these are the highest-priority "
            f"targets for CRM and awareness training programs.",
            chart_key="fleet_hf",
        )

        # ── HF × Parent Category interaction heatmap ──
        hf_by_cat = dl.human_factors_by_category()
        if not hf_by_cat.empty:
            st.markdown("#### How Human Factors Interact with Risk Categories")
            st.markdown(
                "Human factor findings don't exist in isolation — they are classified "
                "under specific parent risk categories. This heatmap shows where each "
                "human factor type concentrates, revealing which risk areas are most "
                "driven by human error."
            )

            _hf_short = {
                "HF-VIOLATION": "Procedural<br>Violation",
                "HF-PERCEPTUAL": "Perceptual<br>Error",
                "HF-DECISION": "Decision<br>Error",
                "HF-SKILL": "Skill-Based<br>Error",
            }
            hf_by_cat["hf_label"] = hf_by_cat["hf_code"].map(
                lambda c: _hf_short.get(c, c)
            )
            hf_by_cat["parent_label"] = hf_by_cat["parent_code"].map(
                lambda c: ABBREVIATIONS.get(c, c)
            )

            hf_pivot = hf_by_cat.pivot_table(
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

            chart_with_insight(
                fig_hf_cat,
                "CFIT accidents are most associated with procedural violations and perceptual "
                "errors, while LOC-I also shows a strong violation and perceptual component. "
                "Fuel-related accidents show only decision errors (e.g., fuel planning failures). "
                "This tells fleet managers that **compliance-focused training** (for violations) "
                "and **situational awareness training** (for perceptual errors) should be "
                "emphasized in both CFIT and LOC-I prevention programs.",
                chart_key="fleet_hf_category",
            )
            st.markdown(
                '<div class="coverage-note">'
                'Note: Human factors subcategories are only classified under LOC-I, CFIT, and '
                'Fuel Related in the CICTT taxonomy. Other risk categories (e.g., SCF-PP, ICE, RE) '
                'do not have human factors L2 classifications in this dataset.'
                '</div>',
                unsafe_allow_html=True,
            )
    else:
        st.info("No human factors data available.")

    # ══════════════════════════════════════════════════════════════════════
    # 9. LOC-I SUBTYPES BY AIRCRAFT TYPE
    # ══════════════════════════════════════════════════════════════════════

    section_divider()
    st.markdown("### Loss of Control — Subtype Analysis")
    st.markdown(
        "Loss of Control — In Flight (LOC-I) is the leading cause of fatal accidents. "
        "Understanding **which type** of control loss dominates for each fleet type "
        "guides specific UPRT (Upset Prevention and Recovery Training) priorities."
    )

    loci_data = dl.loci_subtypes_by_aircraft()
    if not loci_data.empty:
        # Build heatmap: aircraft_category × LOC-I subtype (% of LOC-I accidents)
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
                "Each cell shows what percentage of that aircraft type's LOC-I accidents "
                "involved the given subtype. This normalization means aircraft types with "
                "few LOC-I events are compared fairly against those with many. "
                "**Stall** and **Upset** dominate across most fleet types, but the balance varies — "
                "training programs should target the dominant LOC-I mechanism for each fleet type.",
                chart_key="fleet_loci_subtypes",
            )
        else:
            st.info("Insufficient LOC-I subtype data for aircraft type comparison.")
    else:
        st.info("No LOC-I subtype data available.")

    # ══════════════════════════════════════════════════════════════════════
    # 10. WEATHER-CONDITIONAL RISK — IMC vs VMC DIVERGING CHART
    # ══════════════════════════════════════════════════════════════════════

    section_divider()
    st.markdown("### IMC vs VMC Risk Comparison")
    st.markdown(
        "How does the risk profile change when flying in poor weather? This chart "
        "compares the prevalence of each category in **IMC** (Instrument Meteorological "
        "Conditions) versus **VMC** (Visual Meteorological Conditions). Categories "
        "that skew heavily to one side indicate weather-sensitive risks."
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
                f"**{most_imc['category_label']}** is most overrepresented in IMC "
                f"({most_imc['imc_prevalence']:.0f}% vs {most_imc['vmc_prevalence']:.0f}% in VMC), "
                f"while **{most_vmc['category_label']}** skews heavily toward VMC "
                f"({most_vmc['vmc_prevalence']:.0f}% vs {most_vmc['imc_prevalence']:.0f}% in IMC). "
                f"Categories that extend further right warrant focused instrument training; "
                f"those extending left are primarily clear-weather operational risks.",
                chart_key="fleet_weather_diverge",
            )

            from app.components.report_layout import coverage_note
            coverage_note("Weather", summary["weather_coverage_pct"], n_accidents)
        else:
            st.info("Insufficient weather data for comparison.")
    else:
        st.info("No weather risk data available.")

    # ══════════════════════════════════════════════════════════════════════
    # 11. METHODOLOGY
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

**Headline KPIs:** LOC-I/CFIT rate counts reports involving either Loss of Control in
Flight or Controlled Flight Into Terrain. Component Failure rate counts reports citing
SCF-PP or SCF-NP. IMC involvement counts reports where weather was classified as
Instrument Meteorological Conditions ({summary['weather_coverage_pct']:.0f}% coverage).

**Human factors:** L2 subcategories (HF-DECISION, HF-PERCEPTUAL, HF-SKILL, HF-VIOLATION)
are aggregated across all parent categories. A single report may contribute to multiple
HF subtypes if classified under more than one parent (e.g., LOC-I and CFIT).

**Risk ratios (IMC vs VMC):** Computed as (category prevalence in IMC) / (category
prevalence in VMC). A ratio of 2.0 means the category appears in twice the share of
IMC accidents vs VMC accidents. Weather data is available for {summary['weather_coverage_pct']:.0f}%
of accident reports.

**Limitations:**
- The dataset reflects investigated accidents only, not all aviation incidents or the broader fleet population.
- Report availability varies by decade; earlier periods have fewer reports.
- Category assignments depend on the information available in each report and may not capture all contributing factors.
- Co-occurrence counts reflect shared categorization, not proven causal chains.
"""
    )
