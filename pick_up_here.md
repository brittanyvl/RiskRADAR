# RiskRADAR - Pick Up Here

**Last Updated:** 2026-02-25

## Completed

- [x] Fleet Safety Report — full overhaul (KPIs, multi-select aircraft comparison, manufacturer heatmap, human factors, LOC-I subtypes, IMC/VMC diverging chart, decade trends)
- [x] Underwriting Risk Report — full overhaul (6 KPIs, 10 sections, quadrant bubble scatter severity spectrum, interactive Bayesian profile builder, co-occurrence heatmap, decade trends, storytelling-driven section ordering)

## Roadmap (in order)

### 1. Operational Risk Report — UI review & polish
- [ ] Review current page (`app/views/operational_risk.py`), identify layout/UX issues
- [ ] Apply same polish standards as Fleet Safety and Underwriting (horizontal labels, percentage-based heatmaps, acronym tooltips, targeted KPIs)
- [ ] Evaluate whether new analytics/sections are needed

### 2. Semantic Search page
- [ ] Integrate `search/` module (BM25 + semantic + hybrid via RRF)
- [ ] Add taxonomy filters (L1/L2 category filtering)
- [ ] Include PDF links in results
- [ ] Build search results UI with chunk previews

### 3. Taxonomy Explorer page
- [ ] L1 category overview with counts and descriptions
- [ ] L1 → L2 drill-down with report lists
- [ ] Visual category distribution (treemap or sunburst)

### 4. Risk Profiler — review & finalize
- [ ] Review current page (`app/pages/4_Risk_Profiler.py`)
- [ ] Polish visualization (bar chart of posteriors, confidence indicators)
- [ ] Validate UX flow and output clarity

### 5. About page — review & finalize
- [ ] Review and fix current About/landing page
- [ ] Ensure project description, methodology, and data sourcing are clear
- [ ] Final layout and copy polish

## Future (post-deployment)
- [ ] FastAPI backend (if needed beyond Streamlit)
- [ ] Model improvements (feature interactions, hierarchical Bayes)
- [ ] Improve weather extraction coverage (currently 72.4%)
