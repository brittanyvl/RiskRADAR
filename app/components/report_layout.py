"""
app.components.report_layout — Professional report components.

Design principles:
- KPI cards that don't overflow — contained, centered, responsive
- Plain English for non-technical readers ("Based on 431 accident reports")
- No raw "n=X" notation in user-facing text
- Insight callouts with left-border styling
- Clean section dividers
- Abbreviation tooltips on first use
"""

import re
import streamlit as st


# ── Abbreviation Tooltip Dictionary ───────────────────────────────────────
# Maps abbreviation → full name. Used by abbr() for <abbr title="..."> tags.
ABBREVIATIONS = {
    "LOC-I": "Loss of Control — In Flight",
    "CFIT": "Controlled Flight Into Terrain",
    "SCF-PP": "System/Component Failure — Powerplant",
    "SCF-NP": "System/Component Failure — Non-Powerplant",
    "MAC": "Mid-Air Collision",
    "RE": "Runway Excursion",
    "ARC": "Abnormal Runway Contact",
    "UIMC": "Unintended Flight in IMC",
    "ICE": "Icing",
    "FUEL": "Fuel Related",
    "LALT": "Low Altitude Operations",
    "OTHR": "Other",
    "UNK": "Unknown or Undetermined",
    "RAMP": "Ground Handling",
    "BIRD": "Bird Strike",
    "ATM": "Air Traffic Management / Communication",
    "NAV": "Navigation",
    "RI": "Runway Incursion",
    "TURB": "Turbulence",
    "WSTRW": "Windshear or Thunderstorm",
    "ADRM": "Aerodrome",
    "SEC": "Security Related",
    "CABIN": "Cabin Safety",
    "EVAC": "Evacuation",
    "F-NI": "Fire/Smoke — Non-Impact",
    "F-POST": "Fire/Smoke — Post-Impact",
    "GCOL": "Ground Collision",
    "VMC": "Visual Meteorological Conditions",
    "IMC": "Instrument Meteorological Conditions",
    "CICTT": "CAST/ICAO Common Taxonomy Team",
    "NTSB": "National Transportation Safety Board",
    "HFACS": "Human Factors Analysis and Classification System",
    "UPRT": "Upset Prevention and Recovery Training",
    "TAWS": "Terrain Awareness and Warning System",
    "GPWS": "Ground Proximity Warning System",
    "ECE": "Expected Calibration Error",
    "IFR": "Instrument Flight Rules",
    "VFR": "Visual Flight Rules",
    "SOP": "Standard Operating Procedures",
}


def abbr(code: str) -> str:
    """Return HTML <abbr> tooltip for a known abbreviation, or plain text."""
    full = ABBREVIATIONS.get(code)
    if full:
        return f'<abbr title="{full}">{code}</abbr>'
    return code


def kpi_row(metrics: list[dict]):
    """
    Render a row of professional KPI cards.

    Each dict: {label: str, value: str|int, detail: str (optional)}
    Max 4 cards per row for readability.
    """
    cols = st.columns(len(metrics))
    for col, m in zip(cols, metrics):
        with col:
            detail_html = f'<div class="kpi-detail">{m.get("detail", "")}</div>' if m.get("detail") else ""
            st.markdown(f"""
            <div class="kpi-card">
                <div class="kpi-value">{m['value']}</div>
                <div class="kpi-label">{m['label']}</div>
                {detail_html}
            </div>
            """, unsafe_allow_html=True)


def page_header(title: str, subtitle: str):
    """Render page title and subtitle."""
    st.markdown(f"## {title}")
    st.markdown(f"*{subtitle}*")


def section_divider():
    """Render a clean section divider."""
    st.markdown('<hr class="section-divider">', unsafe_allow_html=True)


def insight(text: str, type: str = "default"):
    """
    Render an insight callout with left-border styling.

    type: "default" (blue), "warning" (amber), "critical" (red)
    """
    css_class = "insight-box"
    if type == "warning":
        css_class += " warning"
    elif type == "critical":
        css_class += " critical"
    st.markdown(f'<div class="{css_class}">{text}</div>', unsafe_allow_html=True)


def coverage_note(feature: str, coverage_pct: float, total: int):
    """
    Render a plain-English data quality disclaimer.

    Example output: "Weather data is available for 344 of 436 accident reports (78.9%).
    Results reflect the subset with known values."
    """
    n_covered = int(total * coverage_pct / 100)
    st.markdown(
        f'<div class="coverage-note">'
        f'{feature} data is available for {n_covered:,} of {total:,} accident reports '
        f'({coverage_pct:.1f}%). Results reflect the subset with known values.'
        f'</div>',
        unsafe_allow_html=True,
    )


def sample_note(n: int, description: str = "accident reports"):
    """Plain-English sample size note. No 'n=' notation."""
    st.markdown(
        f'<div class="coverage-note">Based on {n:,} {description}.</div>',
        unsafe_allow_html=True,
    )


def methodology_section(content: str):
    """Render methodology & limitations in an expander."""
    with st.expander("Methodology & Data Notes"):
        st.markdown(content)


def chart_with_insight(
    fig,
    insight_text: str,
    insight_type: str = "default",
    chart_key: str | None = None,
):
    """
    Render a chart followed by its narrative insight.

    Standard pattern: visual first, then the "so what?" below it.
    """
    st.plotly_chart(fig, use_container_width=True, key=chart_key)
    insight(insight_text, type=insight_type)
