"""Glossary page — terminology reference for all reports."""

import streamlit as st
from app.components import data_loader as dl
from app.components.report_layout import page_header


def render():
    page_header(
        "Glossary & Terminology",
        "Reference guide for aviation safety and statistical terms used "
        "throughout RiskRADAR. Use the search box to find specific terms."
    )

    search = st.text_input(
        "Search terms...", placeholder="e.g. LOC-I, Bayesian, IMC"
    )
    search_lower = search.strip().lower()

    def matches(text: str) -> bool:
        return not search_lower or search_lower in str(text).lower()

    def filter_df(df, columns=None):
        if not search_lower:
            return df
        cols = columns or df.columns
        mask = df[cols].apply(
            lambda col: col.astype(str).str.lower().str.contains(search_lower, na=False)
        ).any(axis=1)
        return df[mask]

    # ── CICTT L1 Categories
    st.markdown("### CICTT Occurrence Categories")
    st.markdown(
        "The 27 standard categories defined by the CAST/ICAO Common Taxonomy "
        "Team for classifying aviation accidents."
    )
    l1 = dl.get_l1_glossary()
    l1_filtered = filter_df(l1)
    if not l1_filtered.empty:
        st.dataframe(
            l1_filtered.rename(columns={"code": "Code", "name": "Name", "description": "Description"}),
            use_container_width=True, hide_index=True,
            height=min(35 * len(l1_filtered) + 38, 500),
        )

    # ── L2 Subcategories
    st.markdown("### Subcategories (Level 2)")
    l2 = dl.get_l2_glossary()
    l2_filtered = filter_df(l2)
    parents = l2_filtered["parent_code"].dropna().unique()
    for parent in sorted(parents):
        group = l2_filtered[l2_filtered["parent_code"] == parent]
        with st.expander(f"{parent} subcategories ({len(group)})"):
            st.dataframe(
                group[["code", "name", "description"]].rename(
                    columns={"code": "Code", "name": "Name", "description": "Description"}
                ),
                use_container_width=True, hide_index=True,
            )
    hfacs = l2_filtered[l2_filtered["parent_code"].isna()]
    if not hfacs.empty:
        with st.expander(f"HFACS Human Factors ({len(hfacs)})"):
            st.dataframe(
                hfacs[["code", "name", "description"]].rename(
                    columns={"code": "Code", "name": "Name", "description": "Description"}
                ),
                use_container_width=True, hide_index=True,
            )

    # ── Aircraft Categories
    st.markdown("### Aircraft Categories")
    features = dl.get_feature_definitions()
    ac_filtered = filter_df(features["aircraft_categories"])
    if not ac_filtered.empty:
        st.dataframe(
            ac_filtered.rename(columns={"category": "Category", "description": "Description"}),
            use_container_width=True, hide_index=True,
        )

    # ── US Regions
    st.markdown("### US Census Regions")
    region_df = features["us_regions"]
    region_summary = region_df.groupby("region").agg(
        states=("state_code", lambda x: ", ".join(sorted(x))),
        count=("state_code", "count"),
    ).reset_index()
    region_filtered = filter_df(region_summary, ["region", "states"])
    if not region_filtered.empty:
        st.dataframe(
            region_filtered.rename(columns={
                "region": "Region", "states": "State Codes", "count": "States"
            }),
            use_container_width=True, hide_index=True,
        )

    # ── Aviation Terms
    st.markdown("### Aviation Terminology")
    for term in dl.get_aviation_terms():
        if not matches(str(term)):
            continue
        abbr = f" ({term['abbreviation']})" if term.get("abbreviation") else ""
        st.markdown(f"**{term['term']}{abbr}** — {term['definition']}")

    # ── Statistical Terms
    st.markdown("### Statistical & Modeling Terms")
    for term in dl.get_statistical_terms():
        if not matches(str(term)):
            continue
        st.markdown(f"**{term['term']}** — {term['definition']}")

    st.markdown("---")
    st.caption("All definitions are original content written for this project.")
