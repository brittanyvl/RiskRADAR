"""
Risk Profiler Page - Bayesian Risk Analysis

This page allows users to:
1. Select flight profile features (aircraft, season, region, weather, time)
2. View predicted accident category probabilities
3. Understand risk factors through Bayesian inference
"""

import streamlit as st
import sqlite3
import sys
from pathlib import Path

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from risk_profiler.bayesian_model import load_model

st.set_page_config(
    page_title="Risk Profiler | RiskRADAR",
    page_icon="📊",
    layout="wide"
)

st.title("📊 Risk Profiler")
st.markdown("""
Analyze accident risk profiles using **Bayesian conditional probability**.

Select flight profile features below to see how they affect the probability
distribution over accident categories.
""")


@st.cache_resource
def get_model():
    """Load the Bayesian model (cached for performance)."""
    return load_model()


# Load model
try:
    model = get_model()
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()

# Sidebar with feature selection
st.sidebar.header("Flight Profile")
st.sidebar.markdown("Select features to analyze risk profile:")

# Aircraft Category dropdown
aircraft_options = ["(None)"] + sorted([
    "jet-wide", "jet-narrow", "jet-regional",
    "turboprop", "multi-piston", "single-piston",
    "helicopter", "other"
])
aircraft = st.sidebar.selectbox(
    "Aircraft Category",
    aircraft_options,
    index=0,
    help="FAA-based aircraft category classification"
)

# Season dropdown
season_options = ["(None)", "Winter", "Spring", "Summer", "Fall"]
season = st.sidebar.selectbox(
    "Season",
    season_options,
    index=0,
    help="Time of year based on accident date"
)

# Region dropdown
region_options = ["(None)", "Northeast", "South", "Midwest", "West", "Territory"]
region = st.sidebar.selectbox(
    "Region",
    region_options,
    index=0,
    help="US Census Bureau region classification"
)

# Weather category dropdown
weather_options = ["(None)", "VMC", "IMC"]
weather = st.sidebar.selectbox(
    "Weather Conditions",
    weather_options,
    index=0,
    help="VMC = Visual Meteorological Conditions, IMC = Instrument Meteorological Conditions"
)

# Time of day dropdown
time_options = ["(None)", "Morning", "Afternoon", "Evening", "Night"]
time_of_day = st.sidebar.selectbox(
    "Time of Day",
    time_options,
    index=0,
    help="Morning (05-10), Afternoon (11-16), Evening (17-20), Night (21-04)"
)

# Convert "(None)" to None for the model
aircraft_val = None if aircraft == "(None)" else aircraft
season_val = None if season == "(None)" else season
region_val = None if region == "(None)" else region
weather_val = None if weather == "(None)" else weather
time_val = None if time_of_day == "(None)" else time_of_day

# Main content area
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("Predicted Risk Distribution")

    # Get predictions
    predictions = model.predict(
        aircraft_category=aircraft_val,
        season=season_val,
        region=region_val,
        weather_category=weather_val,
        time_of_day=time_val,
        top_k=10
    )

    # Display as a bar chart
    import pandas as pd

    df = pd.DataFrame(predictions)
    df = df.rename(columns={
        "category_code": "Category",
        "category_name": "Description",
        "probability": "Probability",
        "percentage": "Pct",
        "risk_level": "Risk"
    })

    # Create a visual bar chart
    st.markdown("**Top 10 Most Likely Categories**")

    for idx, row in df.iterrows():
        col_code, col_bar, col_pct, col_risk = st.columns([1.5, 4, 1, 1])

        with col_code:
            st.write(f"**{row['Category']}**")

        with col_bar:
            # Create a progress bar
            progress = row['Probability']
            st.progress(min(progress * 3, 1.0))  # Scale for visibility

        with col_pct:
            st.write(row['Pct'])

        with col_risk:
            if row['Risk'] == "HIGH":
                st.markdown("🔴 HIGH")
            elif row['Risk'] == "MODERATE":
                st.markdown("🟡 MOD")
            else:
                st.markdown("🟢 LOW")

    # Description of top category
    if predictions:
        top = predictions[0]
        st.markdown("---")
        st.markdown(f"**Most Likely:** {top['category_name']} ({top['percentage']})")

with col2:
    st.subheader("Selected Profile")

    # Show current selection
    st.markdown("**Your Selection:**")
    if aircraft_val:
        st.info(f"✈️ Aircraft: {aircraft_val}")
    else:
        st.text("✈️ Aircraft: Any")

    if season_val:
        st.info(f"🗓️ Season: {season_val}")
    else:
        st.text("🗓️ Season: Any")

    if region_val:
        st.info(f"📍 Region: {region_val}")
    else:
        st.text("📍 Region: Any")

    if weather_val:
        st.info(f"🌤️ Weather: {weather_val}")
    else:
        st.text("🌤️ Weather: Any")

    if time_val:
        st.info(f"🕐 Time: {time_val}")
    else:
        st.text("🕐 Time: Any")

    st.markdown("---")
    st.subheader("How It Works")
    st.markdown(f"""
    This model uses **Bayesian inference** to compute:

    ```
    P(category | features) ∝
        P(category) × P(features | category)
    ```

    Where:
    - **P(category)** is the base rate from historical data
    - **P(features | category)** is the likelihood of your
      flight profile given each accident category

    The model learns from **{model.training_report_count} classified
    accident reports** with L1 CICTT taxonomy assignments.

    Risk thresholds are **data-driven** (percentile-based):
    - 🔴 HIGH: > {model.risk_thresholds['high']:.1%}
    - 🟡 MODERATE: > {model.risk_thresholds['moderate']:.1%}
    - 🟢 LOW: below moderate threshold
    """)

# Expandable section for base rates
with st.expander("📈 View Base Rates (Prior Probabilities)"):
    st.markdown("These are the historical frequencies of each accident category:")

    base_rates = model.get_base_rates(top_k=15)
    base_df = pd.DataFrame(base_rates)
    base_df = base_df.rename(columns={
        "category_code": "Code",
        "category_name": "Category",
        "base_rate": "Rate",
        "percentage": "Historical %"
    })
    st.dataframe(base_df, use_container_width=True, hide_index=True)

# Expandable section for methodology
with st.expander("📚 Methodology"):
    st.markdown("""
    ### Data Sources

    - **Accident Reports**: 510 NTSB aviation accident reports (1966-present)
    - **Taxonomy**: CICTT Aviation Occurrence Categories v4.7
    - **Aircraft Database**: OpenFlights (ODbL license) + manual additions
    - **Regions**: US Census Bureau regions (public domain)

    ### Features (5)

    | Feature | Description | Source |
    |---------|-------------|--------|
    | Aircraft Category | FAA category (jet, turboprop, piston, etc.) | Report title |
    | Season | Winter/Spring/Summer/Fall | Accident date |
    | Region | Northeast/South/Midwest/West | Location |
    | Weather | VMC/IMC | Report text extraction |
    | Time of Day | Morning/Afternoon/Evening/Night | Report text extraction |

    ### Model Details

    - **Algorithm**: Naive Bayes with Laplace smoothing
    - **Smoothing Parameter**: α = 1.0
    - **Categories**: 27 CICTT Level 1 categories
    - **Training data**: Accident reports only (excludes safety studies)
    - **Risk thresholds**: Data-driven (90th/50th percentile)

    ### Limitations

    - **Selection bias**: Data only includes accidents, not safe flights
    - **Sparse data**: Some feature combinations have few examples
    - **Independence assumption**: Naive Bayes assumes features are independent

    ### Citations

    - CICTT Aviation Occurrence Categories, CAST/ICAO, v4.7
    - OpenFlights Aircraft Database (https://openflights.org/data.html)
    - US Census Bureau Regional Classifications
    """)

# Footer
st.markdown("---")
st.caption("RiskRADAR Risk Profiler | Data from NTSB accident reports | Not for operational use")
