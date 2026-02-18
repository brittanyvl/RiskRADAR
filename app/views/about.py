"""About page — project info, author, learnings."""

import streamlit as st


def render():
    st.markdown("## About RiskRADAR")

    st.markdown(
        "RiskRADAR (Retrieval and Discovery of Aviation Accident Reports) is an "
        "end-to-end data science portfolio project that transforms unstructured "
        "NTSB aviation accident PDFs into searchable, structured, explainable "
        "safety insights."
    )

    st.markdown("---")

    # ── Project Overview
    st.markdown("### Project Overview")

    st.markdown("""
**The Challenge:** The National Transportation Safety Board (NTSB) publishes detailed
accident investigation reports as PDFs — rich with safety insights but difficult to
search, compare, or analyze at scale.

**The Solution:** RiskRADAR processes 510 NTSB reports through an 8-phase pipeline:
PDF extraction, chunking, embedding, vector search, taxonomy classification, feature
extraction, Bayesian risk modeling, and interactive analytics.

**The Result:** A searchable knowledge base with 24,766 text chunks, 27 accident
categories, and a calibrated risk model that estimates category probabilities for
any flight profile.
    """)

    st.markdown("---")

    # ── Technical Pipeline
    st.markdown("### Technical Pipeline")

    pipeline_data = {
        "Phase": ["PDF Processing", "Text Chunking", "Embeddings", "Vector Search",
                   "Taxonomy", "Feature Extraction", "Risk Modeling", "Analytics"],
        "What": [
            "OCR + text extraction from 510 NTSB PDFs (30,602 pages)",
            "Section-aware chunking into 24,766 searchable segments",
            "MiniLM (search) + NASA MIKA (taxonomy) sentence embeddings",
            "BM25 + semantic + hybrid (RRF) retrieval via Qdrant Cloud",
            "27 CICTT L1 + 32 L2 categories via embedding similarity",
            "Aircraft type, region, season, weather, time of day per report",
            "Binary Relevance Naive Bayes (ECE = 0.021, Hit@5 = 86.8%)",
            "SQLite queries + Plotly dashboards for 3 stakeholder personas",
        ],
        "Tech": [
            "PyMuPDF, Tesseract OCR",
            "Custom section detector",
            "sentence-transformers",
            "Qdrant Cloud, rank_bm25",
            "MIKA cosine similarity",
            "Regex + LLM extraction",
            "Multi-label Naive Bayes",
            "Streamlit, Plotly, Pandas",
        ],
    }
    st.dataframe(pipeline_data, use_container_width=True, hide_index=True)

    st.markdown("---")

    # ── What I Learned
    st.markdown("### What I Learned")

    st.markdown("""
- **Embedding model selection matters more than expected.** MiniLM outperformed the
  aviation-specialized MIKA model on precision (75.5% vs 60.0%) despite lower MRR —
  because MIKA's broader recall introduced more false positives.

- **Multi-label classification requires different thinking.** Aviation accidents rarely
  have a single cause. The average report has 4.2 categories. Binary Relevance Naive Bayes
  handles this naturally, while softmax-based approaches systematically miscalibrate.

- **Calibration > accuracy for decision support.** An ECE of 0.021 means our predicted
  probabilities are trustworthy — a 30% prediction is correct ~30% of the time. This
  matters more than top-1 accuracy for risk assessment.

- **Data quality limits everything.** Weather data is only available for ~79% of reports,
  and ~57% of the dataset is from the 1960s-70s. Every insight carries these caveats.
    """)

    st.markdown("---")

    # ── Author
    st.markdown("### Author")

    st.markdown(
        "Built by **Brady Villamaa** as a portfolio project demonstrating "
        "data engineering, machine learning, and full-stack data application development."
    )

    st.markdown("---")
    st.caption("RiskRADAR | Data from NTSB accident reports | Not for operational use")
