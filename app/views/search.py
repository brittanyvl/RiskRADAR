"""Search page — semantic search interface (placeholder)."""

import streamlit as st


def render():
    st.markdown("## Semantic Search")
    st.markdown(
        "*Search across 24,766 chunks from 510 NTSB accident reports using "
        "hybrid retrieval (BM25 + semantic embeddings).*"
    )

    query = st.text_input(
        "Search accident reports...",
        placeholder="e.g. engine failure during climb, icing on approach",
    )

    col1, col2, col3 = st.columns(3)
    with col1:
        st.selectbox("Aircraft Type", ["All Types", "Jet - Wide Body", "Jet - Narrow Body",
                      "Turboprop", "Single-Engine Piston", "Helicopter"])
    with col2:
        st.selectbox("Category Filter", ["All Categories", "LOC-I", "CFIT", "SCF-PP",
                      "SCF-NP", "ICE", "FUEL", "RE"])
    with col3:
        st.selectbox("Search Mode", ["Hybrid (Recommended)", "Semantic Only", "Keyword Only"])

    if query:
        st.info("Search functionality will be connected to the Qdrant vector database and BM25 index.")
    else:
        st.markdown("---")
        st.markdown(
            "#### How It Works\n\n"
            "RiskRADAR combines two search strategies for comprehensive retrieval:\n\n"
            "- **BM25 (keyword)** — Traditional term-frequency matching for exact phrases and report IDs\n"
            "- **Semantic embeddings** — MiniLM model captures meaning, finding relevant results even "
            "when wording differs\n"
            "- **Hybrid (RRF)** — Reciprocal Rank Fusion merges both result lists for the best of both worlds\n\n"
            "Results are filtered by taxonomy categories and enriched with PDF links to original NTSB reports."
        )
