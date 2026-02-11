"""
RiskRADAR - Aviation Accident Analysis Application

Main entry point for the Streamlit application.

Run with: streamlit run app/main.py
"""

import streamlit as st

st.set_page_config(
    page_title="RiskRADAR",
    page_icon="✈️",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("✈️ RiskRADAR")
st.subheader("Retrieval and Discovery of Aviation Accident Reports")

st.markdown("""
Welcome to RiskRADAR, an end-to-end data + ML portfolio project that transforms
unstructured NTSB aviation accident PDFs into searchable, structured, explainable
safety insights.

### Available Pages

Use the sidebar to navigate to different sections:

- **Risk Profiler**: Bayesian risk analysis based on flight profile features
  - Select aircraft type, season, and region
  - See predicted accident category probabilities
  - Understand risk factors for different flight profiles

### Data Overview

This application analyzes **510 NTSB aviation accident reports** spanning from
1966 to present, with:

- **27 CICTT accident categories** (Level 1 taxonomy)
- **32 industry-standard subcategories** (Level 2 taxonomy)
- **Feature coverage**:
  - Aircraft category: 72.9%
  - Location/region: 90.4%
  - Season: 95.9%

### Technical Stack

- **Embeddings**: MiniLM and NASA MIKA models
- **Vector DB**: Qdrant Cloud
- **Taxonomy**: CICTT + IATA/HFACS subcategories
- **Risk Model**: Bayesian conditional probability

---
*Built as a portfolio project demonstrating data engineering, ML, and full-stack development.*
""")
