"""Risk Profiler page — wraps existing page with new layout."""

import streamlit as st
import pandas as pd
from app.components import data_loader as dl
from app.components.report_layout import page_header, kpi_row, section_divider, methodology_section
from app.components.charts import horizontal_bar
from app.components.theme import STEEL, CORAL, AMBER, TEAL


def render():
    page_header(
        "Risk Profiler",
        "Estimate accident category probabilities for any flight profile using "
        "Bayesian conditional probability."
    )

    model = dl.get_bayesian_model()

    # ── Feature Selection ─────────────────────────────────────────────
    st.markdown("#### Select Flight Profile")

    c1, c2, c3, c4, c5 = st.columns(5)

    with c1:
        aircraft = st.selectbox("Aircraft Type", [
            "Any", "jet-wide", "jet-narrow", "jet-regional",
            "turboprop", "multi-piston", "single-piston", "helicopter",
        ])
    with c2:
        season = st.selectbox("Season", ["Any", "Winter", "Spring", "Summer", "Fall"])
    with c3:
        region = st.selectbox("Region", ["Any", "Northeast", "South", "Midwest", "West"])
    with c4:
        weather = st.selectbox("Weather", ["Any", "VMC", "IMC"])
    with c5:
        time_of_day = st.selectbox("Time of Day", ["Any", "Morning", "Afternoon", "Evening", "Night"])

    # Convert "Any" to None
    ac_val = None if aircraft == "Any" else aircraft
    se_val = None if season == "Any" else season
    re_val = None if region == "Any" else region
    wx_val = None if weather == "Any" else weather
    td_val = None if time_of_day == "Any" else time_of_day

    section_divider()

    # ── Predictions ───────────────────────────────────────────────────
    predictions = model.predict(
        aircraft_category=ac_val, season=se_val, region=re_val,
        weather_category=wx_val, time_of_day=td_val, top_k=10,
    )

    if predictions:
        top = predictions[0]
        kpi_row([
            {"label": "Highest Risk Category", "value": top["category_code"],
             "detail": top["category_name"]},
            {"label": "Probability", "value": top["percentage"],
             "detail": f"Risk Level: {top['risk_level']}"},
            {"label": "Training Reports", "value": f"{model.training_report_count:,}"},
            {"label": "Model Calibration (ECE)", "value": "0.021",
             "detail": "Lower is better (0 = perfect)"},
        ])

    st.markdown("")
    st.markdown("#### Predicted Risk Distribution")
    st.markdown(
        "Each category has an independent probability — they do not sum to 100%. "
        "Multiple categories can be elevated simultaneously, reflecting the "
        "multi-causal nature of aviation accidents."
    )

    df = pd.DataFrame(predictions)
    fig = horizontal_bar(
        df, x="probability", y="category_code",
        title="Top 10 Accident Categories by Predicted Probability",
        color=STEEL,
        show_values=True,
        height=380,
    )
    # Format x-axis as percentage
    fig.update_xaxes(tickformat=".0%", title_text="Probability")
    fig.update_traces(
        text=df.sort_values("probability", ascending=True)["percentage"],
        textposition="outside",
    )
    st.plotly_chart(fig, use_container_width=True)

    section_divider()

    # ── Base Rates ────────────────────────────────────────────────────
    with st.expander("View Base Rates (Historical Frequencies)"):
        st.markdown(
            "These are the overall frequencies of each category in the accident dataset, "
            "before any feature adjustments. The model updates these base rates based on "
            "your selected flight profile."
        )
        base_rates = model.get_base_rates(top_k=15)
        base_df = pd.DataFrame(base_rates).rename(columns={
            "category_code": "Code", "category_name": "Category",
            "base_rate": "Base Rate", "percentage": "Historical Frequency",
        })
        st.dataframe(base_df, use_container_width=True, hide_index=True)

    methodology_section("""
**Algorithm:** Binary Relevance Naive Bayes — 27 independent binary classifiers,
one per CICTT accident category. Each computes P(category | features) via sigmoid
on log-odds.

**Features:** Aircraft category, season, region, weather (VMC/IMC), time of day.

**Training data:** Accident reports only (excludes safety studies and supplements).
Smoothing: Laplace (alpha = 1.0).

**Calibration:** ECE = 0.021 (expected calibration error). Predictions are well-calibrated:
a 30% prediction corresponds to roughly 30% actual occurrence.

**Limitations:** Model reflects historical patterns, not predictive risk. Selection bias:
only includes investigated accidents, not safe flights. Some feature combinations have
limited training examples.
    """)
