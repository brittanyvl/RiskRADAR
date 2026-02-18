"""
RiskRADAR - Aviation Accident Analysis Application

Top-level router with horizontal navigation.
Run with: streamlit run app/main.py
"""

import streamlit as st
import sys
from pathlib import Path

# Ensure project root is on path
sys.path.insert(0, str(Path(__file__).parent.parent))

# ── Page config (must be first Streamlit call) ────────────────────────────
st.set_page_config(
    page_title="RiskRADAR",
    page_icon="✈️",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── Inject custom CSS ────────────────────────────────────────────────────
from app.components.theme import inject_css
inject_css()

# ── Top navigation ───────────────────────────────────────────────────────
from streamlit_option_menu import option_menu

# Primary nav bar
selected = option_menu(
    menu_title=None,
    options=["Search", "Explore", "Analytics", "Risk Profiler", "Glossary", "About"],
    icons=["search", "diagram-3", "bar-chart-line", "sliders", "book", "info-circle"],
    default_index=0,
    orientation="horizontal",
    styles={
        "container": {
            "padding": "0.3rem 0",
            "background-color": "#f8f9fa",
            "border-bottom": "2px solid #e9ecef",
            "margin-bottom": "1rem",
        },
        "icon": {"font-size": "0.85rem"},
        "nav-link": {
            "font-size": "0.9rem",
            "font-weight": "500",
            "padding": "0.5rem 1.2rem",
            "margin": "0 0.15rem",
            "border-radius": "6px",
            "--hover-color": "#e9ecef",
        },
        "nav-link-selected": {
            "background-color": "#1B2A4A",
            "color": "white",
            "font-weight": "600",
        },
    },
)

# ── Analytics sub-navigation ─────────────────────────────────────────────
analytics_page = None
if selected == "Analytics":
    analytics_page = option_menu(
        menu_title=None,
        options=["Fleet Safety", "Underwriting Risk", "Operational Risk"],
        icons=["airplane", "shield-check", "exclamation-triangle"],
        default_index=0,
        orientation="horizontal",
        styles={
            "container": {
                "padding": "0.2rem 0",
                "background-color": "transparent",
                "margin-bottom": "0.5rem",
            },
            "icon": {"font-size": "0.8rem"},
            "nav-link": {
                "font-size": "0.85rem",
                "font-weight": "400",
                "padding": "0.4rem 1rem",
                "border-radius": "20px",
                "color": "#495057",
                "--hover-color": "#e9ecef",
            },
            "nav-link-selected": {
                "background-color": "#e7f0fd",
                "color": "#1B2A4A",
                "font-weight": "600",
            },
        },
    )

# ── Route to page ────────────────────────────────────────────────────────
if selected == "Search":
    from app.views.search import render
    render()
elif selected == "Explore":
    from app.views.explore import render
    render()
elif selected == "Analytics":
    if analytics_page == "Fleet Safety":
        from app.views.fleet_safety import render
        render()
    elif analytics_page == "Underwriting Risk":
        from app.views.underwriting import render
        render()
    elif analytics_page == "Operational Risk":
        from app.views.operational_risk import render
        render()
elif selected == "Risk Profiler":
    from app.views.risk_profiler import render
    render()
elif selected == "Glossary":
    from app.views.glossary import render
    render()
elif selected == "About":
    from app.views.about import render
    render()
