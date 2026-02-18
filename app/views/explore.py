"""Explore page — taxonomy explorer (placeholder)."""

import streamlit as st


def render():
    st.markdown("## Taxonomy Explorer")
    st.markdown(
        "*Browse the CICTT occurrence category taxonomy — 27 Level 1 categories "
        "and 32 Level 2 subcategories — with report counts and distributions.*"
    )

    st.markdown("---")

    col1, col2 = st.columns([1, 2])

    with col1:
        st.markdown("#### Categories")
        st.markdown(
            "Select a category to explore its subcategories, associated reports, "
            "and feature distributions."
        )
        # Placeholder category list
        categories = [
            "LOC-I — Loss of Control (165)",
            "CFIT — Controlled Flight Into Terrain (99)",
            "SCF-PP — Powerplant Failure (69)",
            "SCF-NP — Non-Powerplant Failure (65)",
            "ICE — Icing (25)",
            "FUEL — Fuel Related (17)",
        ]
        for cat in categories:
            st.button(cat, use_container_width=True, key=f"cat_{cat[:5]}")

    with col2:
        st.markdown("#### Category Detail")
        st.info(
            "The taxonomy explorer will display subcategory breakdowns, "
            "report lists with PDF links, and feature distributions for the "
            "selected category. This view is being developed."
        )
