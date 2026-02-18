"""
app.components.theme — Professional CSS theme for RiskRADAR.

Call inject_css() once at the top of main.py to apply globally.
"""

import streamlit as st

# ── Brand Colors ─────────────────────────────────────────────────────────

# Primary palette (colorblind-safe, tested with Coblis simulator)
NAVY = "#1B2A4A"         # Primary text, nav selected
STEEL = "#4A6FA5"        # Secondary, chart primary
SLATE = "#6C757D"        # Muted text
CORAL = "#C44E52"        # Danger / attention (distinct from green for CB)
AMBER = "#DD8452"        # Warning / moderate
TEAL = "#55A868"         # Success / positive
SKY = "#64B5F6"          # Accent / highlight
GOLD = "#DDAA33"         # Morning / warm accent

# Time-of-day color scheme (intuitive, CB-safe)
TIME_COLORS = {
    "Morning": "#DDAA33",    # Gold / warm sunrise
    "Afternoon": "#DD8452",  # Orange / midday warmth
    "Evening": "#88CCEE",    # Light blue / dusk
    "Night": "#1B2A4A",      # Deep navy / darkness
}
TIME_WINDOWS = {
    "Morning": "05:00–10:59",
    "Afternoon": "11:00–16:59",
    "Evening": "17:00–20:59",
    "Night": "21:00–04:59",
}
LIGHT_BG = "#F8F9FA"     # Page background
CARD_BG = "#FFFFFF"       # Card background
BORDER = "#E9ECEF"        # Borders

# Sequential palette for charts (max 5 series — if you need more, rethink the chart)
CHART_PALETTE = [STEEL, CORAL, AMBER, TEAL, "#8C6BB1"]

# Diverging heatmap scale (blue → white → red, CB-safe)
HEATMAP_SCALE = [
    [0.0, "#f7f7f7"],
    [0.25, "#d1e5f0"],
    [0.5, "#92c5de"],
    [0.75, "#f4a582"],
    [1.0, "#b2182b"],
]

# Sequential heatmap (single-hue blue)
SEQUENTIAL_SCALE = [
    [0.0, "#f7fbff"],
    [0.25, "#c6dbef"],
    [0.5, "#6baed6"],
    [0.75, "#2171b5"],
    [1.0, "#084594"],
]


def inject_css():
    """Inject global CSS overrides for a polished, professional look."""
    st.markdown("""
    <style>
    /* ── Hide default Streamlit chrome ─────────────────────────── */
    #MainMenu {visibility: hidden;}
    header {visibility: hidden;}
    footer {visibility: hidden;}
    [data-testid="collapsedControl"] {display: none;}

    /* ── Typography ───────────────────────────────────────────── */
    html, body, [class*="css"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
    }
    h1 { font-size: 1.75rem !important; font-weight: 700 !important; color: #1B2A4A; }
    h2 { font-size: 1.35rem !important; font-weight: 600 !important; color: #1B2A4A; }
    h3 { font-size: 1.1rem !important; font-weight: 600 !important; color: #2c3e50; }

    /* ── Global spacing ───────────────────────────────────────── */
    .block-container {
        padding-top: 1rem !important;
        padding-bottom: 2rem !important;
        max-width: 1100px;
    }

    /* ── KPI Card styling ─────────────────────────────────────── */
    .kpi-card {
        background: white;
        border: 1px solid #e9ecef;
        border-radius: 8px;
        padding: 1rem 1.2rem;
        text-align: center;
        box-shadow: 0 1px 3px rgba(0,0,0,0.04);
        height: 100%;
    }
    .kpi-card .kpi-value {
        font-size: 1.6rem;
        font-weight: 700;
        color: #1B2A4A;
        line-height: 1.2;
        margin-bottom: 0.25rem;
    }
    .kpi-card .kpi-label {
        font-size: 0.78rem;
        font-weight: 500;
        color: #6c757d;
        text-transform: uppercase;
        letter-spacing: 0.03em;
    }
    .kpi-card .kpi-detail {
        font-size: 0.72rem;
        color: #adb5bd;
        margin-top: 0.2rem;
    }

    /* ── Section dividers ─────────────────────────────────────── */
    .section-divider {
        border: none;
        border-top: 1px solid #e9ecef;
        margin: 2rem 0 1.5rem 0;
    }

    /* ── Insight callout ──────────────────────────────────────── */
    .insight-box {
        background: #f0f4f8;
        border-left: 4px solid #4A6FA5;
        border-radius: 0 6px 6px 0;
        padding: 0.8rem 1rem;
        margin: 0.8rem 0;
        font-size: 0.9rem;
        color: #2c3e50;
        line-height: 1.5;
    }
    .insight-box.warning {
        background: #fff8f0;
        border-left-color: #DD8452;
    }
    .insight-box.critical {
        background: #fdf0f0;
        border-left-color: #C44E52;
    }

    /* ── Data quality note ────────────────────────────────────── */
    .coverage-note {
        font-size: 0.75rem;
        color: #868e96;
        font-style: italic;
        padding: 0.3rem 0;
    }

    /* ── Chart container ──────────────────────────────────────── */
    [data-testid="stPlotlyChart"] {
        border: 1px solid #f0f0f0;
        border-radius: 8px;
        padding: 0.5rem;
    }

    /* ── Expander styling ─────────────────────────────────────── */
    .streamlit-expanderHeader {
        font-size: 0.9rem !important;
        font-weight: 500 !important;
    }

    /* ── Dataframe styling ────────────────────────────────────── */
    [data-testid="stDataFrame"] {
        font-size: 0.85rem;
    }

    /* ── Abbreviation tooltips ───────────────────────────────── */
    abbr[title] {
        text-decoration: underline dotted #adb5bd;
        text-underline-offset: 2px;
        cursor: help;
        font-weight: inherit;
    }
    </style>
    """, unsafe_allow_html=True)
